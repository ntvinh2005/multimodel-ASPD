"""Synthetic backend: tiny random Llama models, random factors, synthetic shared gates.

Purpose: exercise the *whole* pipeline (hooks, probes, ablation, artifacts, comparison)
on CPU in seconds, with the same Hugging Face module layout as SmolLM2/Granite
(``model.layers.{l}.self_attn.{q,k,v,o}_proj``, ``mlp.{gate,up,down}_proj``).

Nothing here is a scientific result.  The gates are not produced by a trained encoder;
they are drawn once per anchor from a Zipf-like feature popularity and shared by every
model, which mimics the property the analysis relies on (one gate controls every model)
and yields a mix of frequent and rare gates.

Options (``backend.options`` in YAML), all optional except ``seed``::

    seed: 0
    n_features: 16          # C
    k_per_anchor: 4         # synthetic K
    selected_layers: [1, 2]
    models:                 # heterogeneous widths -> cross-architecture comparison
      - {key: alpha, hidden_size: 32, intermediate_size: 64, num_attention_heads: 4, num_key_value_heads: 2}
      - {key: beta,  hidden_size: 48, intermediate_size: 96, num_attention_heads: 4, num_key_value_heads: 2}
    num_hidden_layers: 4
    vocab_size: 64
    documents_per_pool: 6
    tokens_per_document: 10
    anchors_per_document: 3
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.posthoc.backend import AnchorPool, MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.ids import MATRIX_NAMES, MatrixId

DEFAULT_MODELS = (
    {"key": "alpha", "hidden_size": 32, "intermediate_size": 64,
     "num_attention_heads": 4, "num_key_value_heads": 2},
    {"key": "beta", "hidden_size": 48, "intermediate_size": 96,
     "num_attention_heads": 4, "num_key_value_heads": 2},
)

_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "mlp_gate": "mlp.gate_proj",
    "mlp_up": "mlp.up_proj",
    "mlp_down": "mlp.down_proj",
}


def _llama(spec: Mapping[str, int], num_layers: int, vocab: int, seed: int) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=num_layers,
        num_attention_heads=spec["num_attention_heads"],
        num_key_value_heads=spec["num_key_value_heads"],
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


class ToyBackend:
    def __init__(self, options: Mapping[str, Any], device: str | torch.device = "cpu"):
        self.seed = int(options["seed"])
        self.n_features = int(options.get("n_features", 16))
        self.k_per_anchor = int(options.get("k_per_anchor", 4))
        self.layers = tuple(int(x) for x in options.get("selected_layers", (1, 2)))
        self.num_layers = int(options.get("num_hidden_layers", 4))
        self.vocab = int(options.get("vocab_size", 64))
        self.documents = int(options.get("documents_per_pool", 6))
        self.tokens = int(options.get("tokens_per_document", 10))
        self.anchors_per_document = int(options.get("anchors_per_document", 3))
        self.device = torch.device(device)
        # Tiny models on a many-core login node are dominated by thread contention
        # (minutes instead of seconds).  This process-wide setting is toy-only.
        torch.set_num_threads(int(options.get("num_threads", 1)))
        model_specs = tuple(options.get("models", DEFAULT_MODELS))
        self.model_keys = tuple(str(m["key"]) for m in model_specs)
        self._specs = {str(m["key"]): m for m in model_specs}
        self._models = {
            key: _llama(spec, self.num_layers, self.vocab, self.seed + 101 * i).to(self.device)
            for i, (key, spec) in enumerate(self._specs.items())
        }
        self._hook_specs = {key: self._build_hook_specs(key) for key in self.model_keys}
        self._factors = {key: self._build_factors(key) for key in self.model_keys}
        rank = torch.arange(1, self.n_features + 1, dtype=torch.float64)
        self._popularity = (1.0 / rank) / (1.0 / rank).sum()

    # ---------------------------------------------------------------- layout
    def _build_hook_specs(self, key: str) -> tuple[MatrixHookSpec, ...]:
        spec = self._specs[key]
        hidden, inter = spec["hidden_size"], spec["intermediate_size"]
        head = hidden // spec["num_attention_heads"]
        q_out, kv_out = spec["num_attention_heads"] * head, spec["num_key_value_heads"] * head
        dims = {
            "q": (hidden, q_out), "k": (hidden, kv_out), "v": (hidden, kv_out),
            "o": (q_out, hidden), "mlp_gate": (hidden, inter), "mlp_up": (hidden, inter),
            "mlp_down": (inter, hidden),
        }
        return tuple(
            MatrixHookSpec(
                matrix=MatrixId(key, layer, name),
                module_path=f"model.layers.{layer}.{_PATHS[name]}",
                output_slice=None,
                d_in=dims[name][0],
                d_out=dims[name][1],
            )
            for layer in self.layers
            for name in MATRIX_NAMES
        )

    def _build_factors(self, key: str) -> dict[MatrixId, tuple[Tensor, Tensor]]:
        generator = torch.Generator().manual_seed(self.seed + 7 + self.model_keys.index(key))
        factors = {}
        for spec in self._hook_specs[key]:
            U = torch.randn(self.n_features, spec.d_out, generator=generator)
            U = U / U.norm(dim=-1, keepdim=True)  # unit write directions, as in v3.1
            V = torch.randn(self.n_features, spec.d_in, generator=generator) / spec.d_in**0.5
            factors[spec.matrix] = (U.to(self.device), V.to(self.device))
        return factors

    # ---------------------------------------------------------------- protocol
    def hook_specs(self, model: str) -> tuple[MatrixHookSpec, ...]:
        return self._hook_specs[model]

    def factors(self, model: str) -> Mapping[MatrixId, tuple[Tensor, Tensor]]:
        return self._factors[model]

    def source_model(self, model: str) -> nn.Module:
        return self._models[model]

    def _pool(self, index: int) -> AnchorPool:
        generator = torch.Generator().manual_seed(self.seed * 100_003 + index)
        ids = torch.randint(0, self.vocab, (self.documents, self.tokens), generator=generator)
        mask = torch.ones_like(ids)
        documents, tokens = [], []
        for d in range(self.documents):
            chosen = torch.randperm(self.tokens - 1, generator=generator)[: self.anchors_per_document]
            for t in sorted((chosen + 1).tolist()):
                documents.append(d)
                tokens.append(t)
        anchors = len(tokens)
        gates = {}
        for site in ("pre", "mid"):
            picks = torch.multinomial(
                self._popularity.expand(anchors, -1), self.k_per_anchor,
                replacement=False, generator=generator,
            )
            gate = torch.zeros(anchors, self.n_features, dtype=torch.bool)
            gate[torch.arange(anchors)[:, None], picks] = True
            gates[site] = gate
        inputs = ModelInputs(
            input_ids=ids, attention_mask=mask,
            anchor_document=torch.tensor(documents), anchor_token=torch.tensor(tokens),
        )
        return AnchorPool(
            anchor_ids=tuple(f"toy-p{index}-d{d}-t{t}" for d, t in zip(documents, tokens, strict=True)),
            gates=gates,
            models={key: inputs for key in self.model_keys},
        )

    def pools(self, count: int, *, skip: int = 0) -> Iterator[AnchorPool]:
        for index in range(skip, skip + count):
            yield self._pool(index)

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "toy",
            "scientific_status": "SYNTHETIC_ONLY_NOT_A_RESULT",
            "seed": self.seed,
            "models": {k: dict(v) for k, v in self._specs.items()},
            "selected_layers": list(self.layers),
            "n_features": self.n_features,
            "k_per_anchor": self.k_per_anchor,
        }

    def close(self) -> None:
        self._models.clear()

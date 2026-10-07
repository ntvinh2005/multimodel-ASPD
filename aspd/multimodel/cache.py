"""Build and read aligned caches for ``R^(n)`` and ``X_j^(n)``.

Token shards are written once and replayed through every model, making Assumption A1 true by
construction.  The cache stores matrix inputs ``X_j^(n)`` but not outputs: training reconstructs
``Y_j^(n)=W_j^(n)X_j^(n)`` from the frozen weight snapshot, roughly halving cache storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.config import DataSourceSpec, ModelSpec, MultiModelExperimentConfig

# Cache examples use T=4 tokens and two models n in {base, finetuned}.  The same input_ids[t]
# defines the common position t required by Assumption A1 in the research note.

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_from_value(value: object, tensor_index: int | None, site: str) -> Tensor:
    """Extract a tensor from common HF hook outputs while rejecting ambiguous structures."""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        if tensor_index is not None:
            selected = value[tensor_index]
            if not isinstance(selected, Tensor):
                raise TypeError(f"{site}[{tensor_index}] is not a Tensor")
            return selected
        tensors = [item for item in value if isinstance(item, Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(
        f"hook {site!r} produced {type(value).__name__}; set tensor_index to select one tensor"
    )


def _tokenizer_fingerprint(tokenizer: object) -> str:
    # Record token->id mapping, e.g. {"cat":123}; equal ids must mean equal tokens across models.
    vocab = tokenizer.get_vocab()  # type: ignore[attr-defined]
    # Include BOS/EOS/PAD definitions because they also determine the common token grid.
    special = tokenizer.special_tokens_map  # type: ignore[attr-defined]
    payload = json.dumps(
        {"vocab": sorted(vocab.items()), "special_tokens": special},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    # SHA256 turns the full tokenizer definition into one reproducibility identifier.
    return hashlib.sha256(payload).hexdigest()


def _serialize_example(
    example: Mapping[str, Any], source: DataSourceSpec, tokenizer: object
) -> str:
    value = example[source.field]
    if source.format == "text":
        return str(value)
    if not isinstance(value, list):
        raise TypeError(f"messages field {source.field!r} must be a list")
    try:
        return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            value, tokenize=False, add_generation_prompt=False
        )
    except (AttributeError, ValueError):
        # Explicit, deterministic fallback.  It adds no tokenizer-specific special IDs.
        return "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in value
        )


def _hash_partition(text: str, split: str) -> bool:
    """Deterministic 90/10 document split when a source has no validation split."""

    # Stable document bucket in {0,...,9}; e.g. hash prefix 1234 gives 1234 mod 10 = 4.
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % 10
    # Bucket 0 is validation (10%); buckets 1..9 are train (90%).
    return bucket == 0 if split == "validation" else bucket != 0


def _source_sequence_iterator(
    source: DataSourceSpec,
    split: str,
    tokenizer: object,
    sequence_length: int,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[Tensor]:
    from datasets import load_dataset

    requested_split = source.train_split
    needs_hash_partition = split == "validation" and source.validation_split is None
    if split == "validation" and source.validation_split is not None:
        requested_split = source.validation_split
    dataset = load_dataset(
        source.dataset,
        source.subset,
        split=requested_split,
        streaming=True,
        revision=source.revision,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    eos = tokenizer.eos_token_id  # type: ignore[attr-defined]
    pending: list[int] = []
    for example in dataset:
        text = _serialize_example(example, source, tokenizer)
        if needs_hash_partition and not _hash_partition(text, split):
            continue
        if split == "train" and source.validation_split is None and not _hash_partition(text, split):
            continue
        # Map text to the one common token sequence s=(s_1,...,s_T); e.g. "a b" -> [10,20].
        ids = tokenizer(  # type: ignore[operator]
            text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        # Pack document tokens into a continuous stream shared by every model n.
        pending.extend(ids)
        if eos is not None:
            # EOS separates documents; e.g. [10,20]+[eos] prevents silent concatenation.
            pending.append(eos)
        while len(pending) >= sequence_length:
            # Emit exactly T ids. Example T=4 and pending=[1,2,3,4,5] emits [1,2,3,4].
            yield torch.tensor(pending[:sequence_length], dtype=torch.long)
            # Keep overflow for the next sequence; example above leaves [5].
            del pending[:sequence_length]


def build_token_cache(cfg: MultiModelExperimentConfig) -> dict[str, Any]:
    """Materialize the common token sequences before loading either model."""

    from transformers import AutoTokenizer

    root = Path(cfg.cache.root)
    token_root = root / "tokens"
    manifest_path = token_root / "manifest.json"
    if manifest_path.exists() and not cfg.cache.overwrite:
        return _read_json(manifest_path)
    if token_root.exists() and cfg.cache.overwrite:
        shutil.rmtree(token_root)
    token_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    if cfg.data.chat_template_tokenizer:
        template_tokenizer = AutoTokenizer.from_pretrained(cfg.data.chat_template_tokenizer)
        tokenizer.chat_template = template_tokenizer.chat_template

    # Normalize source mixture. Example weights FineWeb=1,UltraChat=1 sum to 2.
    source_weight = sum(source.weight for source in cfg.data.sources)
    split_counts: dict[str, int] = {}
    for split, requested_tokens in (
        ("train", cfg.data.train_tokens),
        ("validation", cfg.data.validation_tokens),
    ):
        # Number of T-token rows. Example 1,000 tokens,T=256 -> ceil(3.90625)=4 sequences.
        requested_sequences = math.ceil(requested_tokens / cfg.data.sequence_length)
        # Allocate rows proportionally. Example 4 rows and equal weights -> [2,2].
        per_source = [
            int(requested_sequences * source.weight / source_weight) for source in cfg.data.sources
        ]
        # Give rounding remainder to the last source so sum_s rows_s equals requested_sequences.
        per_source[-1] += requested_sequences - sum(per_source)
        sequences: list[Tensor] = []
        for source_index, (source, count) in enumerate(
            zip(cfg.data.sources, per_source, strict=True)
        ):
            iterator = _source_sequence_iterator(
                source,
                split,
                tokenizer,
                cfg.data.sequence_length,
                cfg.data.seed + source_index,
                cfg.data.shuffle_buffer,
            )
            for _ in range(count):
                try:
                    sequences.append(next(iterator))
                except StopIteration as exc:
                    raise RuntimeError(
                        f"source {source.dataset!r} ended before producing {count} {split} sequences"
                    ) from exc
        # Separate deterministic train/validation shuffle streams: seed and seed+1.
        generator = torch.Generator().manual_seed(cfg.data.seed + (0 if split == "train" else 1))
        # Example 4 rows may receive order [2,0,3,1], fixed by the seed.
        order = torch.randperm(len(sequences), generator=generator).tolist()
        split_dir = token_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, start in enumerate(
            range(0, len(sequences), cfg.cache.sequences_per_shard)
        ):
            # Select row indices for this shard; e.g. positions 0:2 -> chosen=[2,0].
            chosen = order[start:start + cfg.cache.sequences_per_shard]
            # Stack token sequences to input_ids[rows,T], e.g. [2,4].
            input_ids = torch.stack([sequences[index] for index in chosen])
            torch.save(
                {
                    "input_ids": input_ids,
                    # Packing emits full-length rows, so every t is valid at cache-build time.
                    "valid_tokens": torch.ones_like(input_ids, dtype=torch.bool),
                },
                split_dir / f"shard_{shard_index:05d}.pt",
            )
        split_counts[split] = len(sequences)

    manifest = {
        "schema_version": 1,
        "tokenizer": cfg.data.tokenizer,
        "tokenizer_revision": cfg.data.tokenizer_revision,
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "sequence_length": cfg.data.sequence_length,
        "sequences": split_counts,
        "sources": [source.model_dump() for source in cfg.data.sources],
    }
    _write_json(manifest_path, manifest)
    return manifest


class _Capture:
    def __init__(self) -> None:
        self.values: dict[str, Tensor] = {}

    def set(self, name: str, value: Tensor) -> None:
        self.values[name] = value.detach()

    def pop(self, name: str) -> Tensor:
        if name not in self.values:
            raise RuntimeError(f"hook {name!r} did not run during the model forward")
        return self.values.pop(name)


def _register_capture_hooks(model: nn.Module, spec: ModelSpec, capture: _Capture, stack: ExitStack) -> None:
    grounding_module = model.get_submodule(spec.grounding.module)
    if spec.grounding.capture == "input":
        stack.callback(
            grounding_module.register_forward_pre_hook(
                lambda _module, args: capture.set(
                    "R",
                    _tensor_from_value(args, spec.grounding.tensor_index or 0, spec.grounding.module),
                )
            ).remove
        )
    else:
        stack.callback(
            grounding_module.register_forward_hook(
                lambda _module, _args, output: capture.set(
                    "R",
                    _tensor_from_value(output, spec.grounding.tensor_index, spec.grounding.module),
                )
            ).remove
        )
    for matrix in spec.matrices:
        module = model.get_submodule(matrix.module)
        stack.callback(
            module.register_forward_pre_hook(
                lambda _module, args, name=matrix.name, site=matrix.module: capture.set(
                    f"X/{name}", _tensor_from_value(args, 0, site)
                )
            ).remove
        )


def build_model_cache(
    cfg: MultiModelExperimentConfig, model_index: int, device: str = "cuda"
) -> dict[str, Any]:
    """Run one ``M^(n)`` over common token shards and cache ``R^(n),X_j^(n),W_j^(n)``."""

    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    token_manifest = build_token_cache(cfg)
    spec = cfg.models[model_index]
    root = Path(cfg.cache.root)
    model_root = root / "models" / spec.name
    manifest_path = model_root / "manifest.json"
    if manifest_path.exists() and not cfg.cache.overwrite:
        return _read_json(manifest_path)
    if model_root.exists() and cfg.cache.overwrite:
        shutil.rmtree(model_root)
    model_root.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "dtype": DTYPES[spec.dtype],
        "trust_remote_code": spec.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs).to(device).eval()
    eos_ids = model.config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids] if eos_ids is not None else []
    if eos_ids and model.config.vocab_size <= max(eos_ids):
        raise ValueError(f"invalid vocabulary metadata for {spec.model_id}")

    weights: dict[str, Tensor] = {}
    matrix_shapes: dict[str, dict[str, int]] = {}
    for matrix in spec.matrices:
        module = model.get_submodule(matrix.module)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise TypeError(f"{matrix.module!r} is not a module with a 2-D weight")
        # Snapshot W_j^(n) once. Example q_proj stores [d_out,d_in] without gradients.
        weights[matrix.name] = weight.detach().cpu().contiguous()
        # Persist dimensions used to validate tying and reconstruct Y_j^(n)=W_j^(n)X_j^(n).
        matrix_shapes[matrix.name] = {"d_out": weight.shape[0], "d_in": weight.shape[1]}
    save_file(weights, str(model_root / "weights.safetensors"))

    cache_dtype = DTYPES[cfg.cache.activation_dtype]
    capture = _Capture()
    # Accumulators for sqrt(E_t ||r_t^(n)||_2^2).
    total_r_norm_sq = 0.0
    total_tokens = 0
    split_shards: dict[str, int] = {}
    with ExitStack() as stack:
        _register_capture_hooks(model, spec, capture, stack)
        for split in ("train", "validation"):
            token_files = sorted((root / "tokens" / split).glob("shard_*.pt"))
            split_dir = model_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for token_file in token_files:
                token_batch = torch.load(token_file, map_location="cpu", weights_only=True)
                # The identical ids[B,T] are replayed for each n; this constructs common t.
                input_ids = token_batch["input_ids"].to(device)
                valid_tokens = token_batch["valid_tokens"].to(device)
                with torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=valid_tokens, use_cache=False)
                # Hook output R^(n)[B,T,d_act^(n)] at the configured grounding site.
                r = capture.pop("R")
                if r.shape[:2] != input_ids.shape:
                    raise ValueError(
                        f"grounding activation token grid {r.shape[:2]} != inputs {input_ids.shape}"
                    )
                # Hook inputs X_j^(n)[B,T,d_in,j^(n)] for every selected j.
                x_by_matrix = {matrix.name: capture.pop(f"X/{matrix.name}") for matrix in spec.matrices}
                for name, x in x_by_matrix.items():
                    if x.shape[:2] != input_ids.shape:
                        raise ValueError(f"{name} token grid {x.shape[:2]} != inputs {input_ids.shape}")
                # Flatten valid t only; example R[1,4,d] and all-valid -> [4,d].
                valid_r = r[valid_tokens]
                # Add sum_t ||r_t^(n)||^2. Example norms squared [1,4,1,2] add 8.
                total_r_norm_sq += valid_r.float().pow(2).sum(dim=-1).sum().item()
                # Add T_valid. Example four positions add 4.
                total_tokens += int(valid_tokens.sum().item())
                torch.save(
                    {
                        "R": r.to(device="cpu", dtype=cache_dtype),
                        "X": {
                            name: x.to(device="cpu", dtype=cache_dtype)
                            for name, x in x_by_matrix.items()
                        },
                    },
                    split_dir / token_file.name,
                )
            split_shards[split] = len(token_files)

    # q_n=sqrt(E_t||r_t^(n)||^2). Example total=8,count=4 gives q_n=sqrt(2).
    # Cache reader returns R^(n)/q_n, hence E||r_scaled||^2=1.
    r_rms_norm = math.sqrt(total_r_norm_sq / max(total_tokens, 1))
    manifest = {
        "schema_version": 1,
        "name": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "grounding": spec.grounding.model_dump(),
        "activation_dim": r.shape[-1],
        "r_rms_norm": r_rms_norm,
        "matrices": matrix_shapes,
        "matrix_modules": {matrix.name: matrix.module for matrix in spec.matrices},
        "tokenizer_fingerprint": token_manifest["tokenizer_fingerprint"],
        "shards": split_shards,
        "activation_dtype": cfg.cache.activation_dtype,
    }
    _write_json(manifest_path, manifest)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def build_cache(cfg: MultiModelExperimentConfig, device: str = "cuda") -> None:
    """Build tokens once, then cache models sequentially on one GPU."""

    build_token_cache(cfg)
    for model_index in range(len(cfg.models)):
        build_model_cache(cfg, model_index, device=device)
    validate_cache(cfg)


def validate_cache(cfg: MultiModelExperimentConfig) -> dict[str, Any]:
    """Fail before training if token grids, shards, matrices, or tying dimensions disagree."""

    root = Path(cfg.cache.root)
    token_manifest = _read_json(root / "tokens" / "manifest.json")
    model_manifests = [
        _read_json(root / "models" / model.name / "manifest.json") for model in cfg.models
    ]
    for manifest in model_manifests:
        if manifest["tokenizer_fingerprint"] != token_manifest["tokenizer_fingerprint"]:
            raise ValueError(f"tokenizer mismatch for cached model {manifest['name']}")
        for split in ("train", "validation"):
            expected = len(list((root / "tokens" / split).glob("shard_*.pt")))
            if manifest["shards"][split] != expected:
                raise ValueError(f"shard count mismatch for {manifest['name']} {split}")
    if cfg.sparsity.tie_shared_decoders:
        dims = [manifest["activation_dim"] for manifest in model_manifests]
        if len(set(dims)) != 1:
            raise ValueError(f"tied decoders require equal activation dimensions, got {dims}")
    if cfg.sparsity.tie_shared_mechanisms:
        reference = model_manifests[0]["matrices"]
        for manifest in model_manifests[1:]:
            if manifest["matrices"] != reference:
                raise ValueError("tied mechanisms require equal matrix names and dimensions")
    return {"tokens": token_manifest, "models": model_manifests}


class PairedActivationCache:
    """Read aligned shards and apply the prescribed ``E||r_t^(n)||^2=1`` scaling."""

    def __init__(self, cfg: MultiModelExperimentConfig):
        self.cfg = cfg
        self.root = Path(cfg.cache.root)
        validated = validate_cache(cfg)
        self.token_manifest = validated["tokens"]
        self.model_manifests: list[dict[str, Any]] = validated["models"]

    def load_weights(self) -> list[dict[str, Tensor]]:
        from safetensors.torch import load_file

        return [
            load_file(str(self.root / "models" / model.name / "weights.safetensors"))
            for model in self.cfg.models
        ]

    def iter_batches(
        self, split: str, batch_size_sequences: int, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, object]]:
        token_files = sorted((self.root / "tokens" / split).glob("shard_*.pt"))
        if shuffle:
            # Epoch-dependent but reproducible shard order: seed+epoch.
            generator = torch.Generator().manual_seed(self.cfg.training.seed + epoch)
            # Example three shards may be traversed [2,0,1].
            order = torch.randperm(len(token_files), generator=generator).tolist()
            token_files = [token_files[index] for index in order]
        for token_file in token_files:
            tokens = torch.load(token_file, map_location="cpu", weights_only=True)
            model_shards = [
                torch.load(
                    self.root / "models" / model.name / split / token_file.name,
                    map_location="cpu",
                    weights_only=True,
                )
                for model in self.cfg.models
            ]
            # Number of cached rows in this shard; example input_ids.shape=[64,256] -> 64.
            n_sequences = tokens["input_ids"].shape[0]
            sequence_order = torch.arange(n_sequences)
            if shuffle:
                sequence_order = sequence_order[
                    torch.randperm(
                        n_sequences,
                        generator=torch.Generator().manual_seed(
                            self.cfg.training.seed + epoch + int(token_file.stem.split("_")[-1])
                        ),
                    )
                ]
            for start in range(0, n_sequences, batch_size_sequences):
                # Choose B sequence rows; example start=0,B=4 -> four aligned rows for every n.
                index = sequence_order[start:start + batch_size_sequences]
                yield {
                    "input_ids": tokens["input_ids"][index],
                    "valid_tokens": tokens["valid_tokens"][index],
                    # Scale R^(n) by q_n so decoder norms are comparable across n.
                    # Example r=[2,0],q_n=2 -> r_scaled=[1,0].
                    "R": [
                        shard["R"][index] / manifest["r_rms_norm"]
                        for shard, manifest in zip(
                            model_shards, self.model_manifests, strict=True
                        )
                    ],
                    # X_j^(n) is not RMS-scaled: it must remain the true input to frozen W_j^(n).
                    "X": [
                        {name: value[index] for name, value in shard["X"].items()}
                        for shard in model_shards
                    ],
                }

"""One prompt through one run: activations, gates and effective contributions of every component."""

from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from aspd.analysis.circuits.replacement import run_replacement
from aspd.analysis.pairs.spaces import ModuleSpaces, head_dim_for_model, module_spaces


@dataclass(frozen=True)
class PromptTrace:
    """The decomposition's view of one prompt. Build with `build_trace`."""

    model: object
    cache: object  # the live `ReplacementCache`: its leaves still carry the autograd graph
    sampling: str
    model_name: str
    tokens: Tensor  # [1, P]
    pieces: list[str]  # P decoded token strings
    acts: dict[str, Tensor]  # module -> [P, C], the encoder output `a`
    inputs: dict[str, Tensor]  # module -> [P, d_in], the module's own input `x`
    gates: dict[str, Tensor]  # module -> [P, C], the causal importance `g`
    errors: dict[str, Tensor]  # module -> [P, d_out], what the decomposition cannot explain
    logits: Tensor  # [P, V]
    spaces: dict[str, ModuleSpaces]

    def out_bias(self, module: str) -> Tensor:
        """`[d_out]` -- the FULL constant `Components.forward` adds, whatever the arm calls it."""
        c = self.model.components[module]  # pyright: ignore[reportAttributeAccessIssue]
        d_out = int(c.U.shape[1])
        total = torch.zeros(d_out)
        for name in ("b_out", "bias"):
            b = getattr(c, name, None)
            if b is not None:
                total = total + b.detach().float()
        return total

    def directions(self, module: str, side: str) -> Tensor:
        """`[C, d]` row-wise, matching `RunWeights.directions(...).mat`."""
        c = self.model.components[module]  # pyright: ignore[reportAttributeAccessIssue]
        return (c.V.t() if side == "read" else c.U).detach().float()

    @property
    def n_pos(self) -> int:
        return len(self.pieces)

    @property
    def modules(self) -> list[str]:
        return sorted(self.spaces, key=lambda m: (self.spaces[m].layer, self.spaces[m].role))

    def effective(self, module: str) -> Tensor:
        """`[P, C]` = `g * a`, the contribution scalar. THE quantity to score with."""
        return self.gates[module] * self.acts[module]

    def fired(self, module: str, pos: int | None = None) -> Tensor:
        """Component indices with `g > 0` -- at one position, or anywhere in the prompt."""
        g = self.gates[module]
        alive = (g > 0) if pos is None else (g[pos] > 0)
        return (alive.any(dim=0) if pos is None else alive).nonzero(as_tuple=False).flatten()

    def module_at(self, layer: int, role: str) -> str:
        """`(6, "attn.q") -> "transformer.h.6.attn.c_attn.q_proj"`, on either architecture."""
        for m, s in self.spaces.items():
            if s.layer == layer and s.role == role:
                return m
        raise AssertionError(
            f"this run has no {role} at layer {layer}; it decomposed "
            f"{sorted({s.role for s in self.spaces.values()})}"
        )

    @property
    def head_dim(self) -> int:
        d = head_dim_for_model(self.model_name)
        assert d is not None, f"no head_dim known for {self.model_name}"
        return d

    @cached_property
    def layers(self) -> list[int]:
        """The layers this run actually decomposed, ascending -- NOT `range(n_layers)`."""
        return sorted({s.layer for s in self.spaces.values()})

    @cached_property
    def n_heads(self) -> int | None:
        """Heads per attention block, or `None` if this run decomposed no attention module."""
        for s in self.spaces.values():
            heads = s.write.heads or s.read.heads
            if heads is not None:
                return heads.n_heads
        return None

    def check_head(self, head: int) -> None:
        n = self.n_heads
        assert n is not None, "this run decomposed no attention module; it has no heads"
        assert 0 <= head < n, f"head {head} is outside 0..{n - 1}"

    def check_pos(self, *positions: int) -> None:
        for p in positions:
            assert 0 <= p < self.n_pos, f"position {p} is outside 0..{self.n_pos - 1}"

    @cached_property
    def attn_probs(self) -> dict[int, Tensor]:
        """`layer -> [H, P, P]` attention probabilities, `[h, query, key]`."""
        tm = self.model.target_model  # pyright: ignore[reportAttributeAccessIssue]
        cfg, inner = tm.config, tm.transformer if hasattr(tm, "transformer") else tm.model
        prev_cfg, prev_inner = cfg._attn_implementation, inner._attn_implementation
        cfg._attn_implementation = inner._attn_implementation = "eager"
        try:
            with torch.no_grad():
                out = tm(self.tokens, output_attentions=True)
        finally:
            cfg._attn_implementation, inner._attn_implementation = prev_cfg, prev_inner
        assert out.attentions is not None, "eager attention returned no probabilities"
        return {i: a[0].float() for i, a in enumerate(out.attentions)}


def target_model_name(cfg) -> str:
    spec = cfg.target.spec
    if hasattr(spec, "model_name"):
        return str(spec.model_name)
    name = spec.params.get("model_name")
    assert name, f"target spec {spec.kind!r} declares no model_name; cannot resolve head layout"
    return str(name)


def build_trace(model, cfg, tokens: Tensor, pieces: list[str],
                mask_edits: dict[str, Tensor] | None = None) -> PromptTrace:
    """One replacement pass -> a `PromptTrace`. `tokens` is `[1, P]`."""
    assert tokens.ndim == 2 and tokens.shape[0] == 1, f"expected [1, P], got {tuple(tokens.shape)}"
    cache = run_replacement(model, tokens, sampling=cfg.pd.sampling, error_nodes=True,
                            mask_edits=mask_edits)
    assert cache.logits is not None

    model_name = target_model_name(cfg)
    head_dim = head_dim_for_model(model_name)
    spaces = {
        path: module_spaces(path, int(c.V.shape[0]), int(c.U.shape[1]), head_dim)
        for path, c in model.components.items()
    }
    return PromptTrace(
        model=model,
        cache=cache,
        sampling=cfg.pd.sampling,
        model_name=model_name,
        tokens=tokens,
        pieces=pieces,
        acts={k: v[0].detach().float() for k, v in cache.component_acts.items()},
        inputs={k: v[0].detach().float() for k, v in cache.inputs.items()},
        gates={k: v[0].detach().float() for k, v in cache.ci.items()},
        errors={k: v[0].detach().float() for k, v in cache.errors.items()},
        logits=cache.logits[0].detach().float(),
        spaces=spaces,
    )

"""PD Transcoder's causal-importance function.

Training: g_{t,c} = 1[c in BatchTopK over the batch of relu(z_{t,c})], z_{t,c} = v_c^T (x_t - b_dec),
with k = `top_k` per token on average. Inference: g_{t,c} = 1[relu(z_{t,c}) > theta], with the
threshold theta fitted during training (JumpReLU).
"""

import math
from typing import Literal

import torch
from jaxtyping import Float
from param_decomp.base_config import BaseConfig
from param_decomp.components import get_module_input_dim
from pydantic import PositiveInt
from torch import Tensor, nn


class PDTranscoderCiConfig(BaseConfig):
    """PD Transcoder's gate: BatchTopK over the components' own encoder V."""

    mode: Literal["pd_transcoder"] = "pd_transcoder"

    n_features: PositiveInt | None = None
    """Number of features F; must equal C (component c is feature c). `None`: each module's own C."""

    encoder: Literal["tied", "untied"] = "tied"
    """`tied`: the gate reads V from the components. `untied`: a separate encoder (ASPD)."""

    top_k: PositiveInt = 32
    """BatchTopK budget: k features per token on average (the target L0)."""

    pool_tokens: PositiveInt = 2048
    """Tokens per BatchTopK pool: the top `top_k * pool_tokens` activations are selected jointly."""

    threshold_lr: float = 0.01
    """EMA rate for the JumpReLU threshold, tracking the minimum positive BatchTopK activation."""

    n_batches_to_dead: PositiveInt = 320
    """Pools a feature may go without firing before AuxK counts it as dead."""

    sync_statistics: bool = True
    """Reduce the threshold and dead-feature statistics across data-parallel ranks; false: each rank
    keeps its own, from its own shard of the batch."""


def _all_reduce_(t: Tensor, op: "torch.distributed.ReduceOp") -> Tensor:
    """In-place all-reduce when distributed, no-op otherwise. Returns `t` for chaining."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=op)
    return t


class PDTranscoderCiFn(nn.Module):
    """g_{t,c} = 1[feature c is selected], with z = V^T (x - b_dec) read from the components."""

    threshold: Tensor
    n_batches_not_active: Tensor

    allowed_encoders: tuple[str, ...] = ("tied",)

    gate_implies_positive_preact: bool = True

    def __init__(self, module: str, n_components: int, d_in: int, cfg: PDTranscoderCiConfig):
        super().__init__()
        assert cfg.encoder in self.allowed_encoders, (
            f"encoder={cfg.encoder!r} is not implemented by {type(self).__name__}, which builds "
            f"{' / '.join(self.allowed_encoders)} (the untied encoder is `aspd.ci.aspd`)"
        )
        assert cfg.n_features is None or n_components == cfg.n_features, (
            f"C={n_components} != n_features={cfg.n_features} (component c is feature c)"
        )
        self.module = module
        self.cfg = cfg
        self.d_in = d_in
        self.d_x = d_in
        self.n_features = n_components

        # The JumpReLU threshold; a parameter without gradient so it is saved with the model.
        self.log_threshold = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.register_buffer("threshold", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "n_batches_not_active", torch.zeros(n_components, dtype=torch.float32)
        )
        # A list, not a submodule, so the components are not registered twice.
        self._tied: list[nn.Module] = []

    def attach_components(self, components: nn.Module) -> "PDTranscoderCiFn":
        """Attach to the components this gate reads V from; run after the Trainer is built."""
        assert hasattr(components, "b_dec"), (
            f"{type(components).__name__} has no `b_dec`; the gate needs `component_arch: transcoder`"
        )
        assert components.V.shape == (self.d_x, self.n_features), (
            f"V is {tuple(components.V.shape)}, expected {(self.d_x, self.n_features)}"
        )
        self._tied = [components]
        return self

    @property
    def components(self) -> nn.Module:
        assert self._tied, "call `seed_transcoder_ci_fn` after building the Trainer"
        return self._tied[0]

    def preacts(
        self, x: Float[Tensor, "... d"], z: Float[Tensor, "... f"] | None = None
    ) -> Float[Tensor, "... f"]:
        """relu(z), the quantity BatchTopK ranks."""
        if z is None:
            z = self.components.get_component_acts(x)
        return torch.relu(z)

    def auxk_write_acts(
        self, z: Float[Tensor, "... f"], preacts: Float[Tensor, "... f"]
    ) -> Float[Tensor, "... f"]:
        """The coefficient AuxK uses for a revived feature's decoder row."""
        del z
        return preacts

    def _in_training_step(self) -> bool:
        """Training (batch top-k and statistics updates) vs. inference (JumpReLU threshold)."""
        return self.training and torch.is_grad_enabled()

    @torch.no_grad()
    def _update_threshold(self, acts_topk: Tensor) -> None:
        positive = acts_topk > 0
        local = (
            acts_topk[positive].min()
            if positive.any()
            else torch.tensor(float("inf"), device=acts_topk.device, dtype=self.threshold.dtype)
        )
        smallest = local.clone().to(self.threshold.dtype)
        if self.cfg.sync_statistics:
            smallest = _all_reduce_(smallest, torch.distributed.ReduceOp.MIN)
        if torch.isfinite(smallest):
            lr = self.cfg.threshold_lr
            self.threshold.mul_(1.0 - lr).add_(lr * smallest)

    @torch.no_grad()
    def _update_inactive(self, acts_topk: Tensor) -> None:
        fired = (acts_topk.sum(0) > 0).to(self.n_batches_not_active.dtype)
        if self.cfg.sync_statistics:
            fired = _all_reduce_(fired, torch.distributed.ReduceOp.MAX)
        fired = fired > 0
        self.n_batches_not_active += (~fired).to(self.n_batches_not_active.dtype)
        self.n_batches_not_active[fired] = 0.0

    def _batch_topk_mask(self, acts: Float[Tensor, "n f"]) -> Float[Tensor, "n f"]:
        """1 for the top `top_k * n` positive entries of a pool of n tokens, 0 elsewhere."""
        n = acts.shape[0]
        flat = acts.flatten()
        top = torch.topk(flat, self.cfg.top_k * n, dim=-1)
        mask = torch.zeros_like(flat).scatter(-1, top.indices, torch.ones_like(top.values))
        return (mask.reshape(acts.shape) * (acts > 0)).float()

    def gate(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        """g in {0, 1}: BatchTopK per pool while training, the JumpReLU threshold otherwise."""
        acts = self.preacts(x)
        if not self._in_training_step():
            return self._gate_value(acts, (acts > self.threshold).float())

        flat = acts.reshape(-1, acts.shape[-1])
        pool = self.cfg.pool_tokens
        masks = []
        for start in range(0, flat.shape[0], pool):
            block = flat[start : start + pool]
            block_mask = self._batch_topk_mask(block)
            self._update_threshold(block * block_mask)
            self._update_inactive(block * block_mask)
            masks.append(block_mask)
        return self._gate_value(acts, torch.cat(masks, dim=0).reshape(acts.shape))

    def _gate_value(
        self, acts: Float[Tensor, "... f"], selected: Float[Tensor, "... f"]
    ) -> Float[Tensor, "... f"]:
        """The gate value of a selected feature (the indicator)."""
        del acts
        return selected

    def n_pools(self, n_tokens: int) -> int:
        """Number of BatchTopK pools `n_tokens` tokens form."""
        return max(1, math.ceil(n_tokens / self.cfg.pool_tokens))

    def forward(
        self, input_acts: dict[str, Float[Tensor, "... d_in"]]
    ) -> dict[str, Float[Tensor, "... c"]]:
        x = input_acts[self.module]
        return {self.module: self.gate(x)}


class PDTranscoderCiFnSet(nn.Module):
    """One `PDTranscoderCiFn` per decomposed module, with the same dict-in/dict-out interface."""

    def __init__(self, ci_fns: dict[str, "PDTranscoderCiFn"]):
        super().__init__()
        assert ci_fns, "PDTranscoderCiFnSet needs at least one module"
        self.module_names = sorted(ci_fns)
        self._ci_fns = nn.ModuleDict({m.replace(".", "-"): ci_fns[m] for m in self.module_names})
        self.cfg = ci_fns[self.module_names[0]].cfg

    def fns(self) -> dict[str, "PDTranscoderCiFn"]:
        return {m: self._ci_fns[m.replace(".", "-")] for m in self.module_names}

    def attach_components(self, components: dict[str, nn.Module]) -> "PDTranscoderCiFnSet":
        """Attach every member to its module's components; run after the Trainer is built."""
        missing = set(self.module_names) - set(components)
        extra = set(components) - set(self.module_names)
        assert not missing and not extra, (
            f"CI-fn/components mismatch -- missing {sorted(missing)}, extra {sorted(extra)}"
        )
        for name, fn in self.fns().items():
            fn.attach_components(components[name])
        return self

    def forward(
        self, input_acts: dict[str, Float[Tensor, "... d_in"]]
    ) -> dict[str, Float[Tensor, "... c"]]:
        """Gates for every module in `input_acts`."""
        unknown = set(input_acts) - set(self.module_names)
        assert not unknown, f"no gate for {sorted(unknown)}; this set covers {self.module_names[:4]}..."
        members = self.fns()
        out: dict[str, Float[Tensor, "... c"]] = {}
        for name in input_acts:
            out.update(members[name]({name: input_acts[name]}))
        return out


def gate_for(ci_fn: nn.Module, module: str) -> nn.Module:
    """The object that gates `module`: the CI fn itself, or its set's member for that module."""
    fns = getattr(ci_fn, "fns", None)
    if fns is None:
        return ci_fn
    members = fns()
    assert module in members, (
        f"the CI-fn set does not gate {module!r}; it covers {sorted(members)[:4]}..."
    )
    return members[module]


def make_transcoder_ci_fn(
    *,
    target_model: nn.Module,
    module_to_c: dict[str, int],
    ci_config: PDTranscoderCiConfig,
) -> "PDTranscoderCiFn | PDTranscoderCiFnSet":
    """Build PD Transcoder's CI function (one per decomposed module)."""
    assert module_to_c, "no decomposition targets"
    fns = {
        module: PDTranscoderCiFn(
            module=module,
            n_components=n_components,
            d_in=get_module_input_dim(target_model.get_submodule(module)),
            cfg=ci_config,
        )
        for module, n_components in module_to_c.items()
    }
    if len(fns) == 1:
        return next(iter(fns.values()))
    return PDTranscoderCiFnSet(fns)

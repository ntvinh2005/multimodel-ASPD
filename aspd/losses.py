"""The loss terms this package adds to `param_decomp`, registered as ordinary training losses.

- `InternalReconLoss`: L_internal, the reconstruction of the decomposed matrix's output
  y_t = W x_t from the gated components, y_hat_t = b + sum_c m_{t,c} (v_c^T x_t) u_c, as FVU or as
  a Matryoshka prefix loss.
- `ActivationReconLoss`: L_act, ASPD's shared encoder g^s reconstructing the residual stream it
  reads, r_hat_t = b_dec + sum_c g^s_{t,c}(R) W_dec[c], as a normalized Matryoshka prefix loss.
- `AuxKLoss`: the auxiliary loss reviving dead features against another term's residual.
- `ComponentAliveTracker`: not a loss; tracks which components fired within a token window.

Each class is registered in `LOSS_METRIC_CLASSES`; `aspd.config` adds the config classes to the
parse union. Terms are computed and logged at every coefficient, including 0.
"""

import math
from typing import Any, Literal, override

import torch
from param_decomp.component_model import ComponentModel
from param_decomp.masks import (
    AllLayersRouter,
    ComponentsMaskInfo,
    calc_stochastic_component_mask_info,
    make_mask_infos,
)
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.dispatch import LOSS_METRIC_CLASSES
from pydantic import PositiveInt
from torch import Tensor

from aspd.adaptive_l0 import AdaptiveSparsityLoss
from aspd.alive import AliveTracker
from aspd.ci.aspd import SharedEncoder
from aspd.ci.pd_transcoder import PDTranscoderCiFn, gate_for
from aspd.loss_utils import (
    masked_module_output,
    site_recon_stats,
    target_module_output,
    warmup_scale,
)
from aspd.sae.matryoshka import DEFAULT_GROUP_FRACS
from aspd.transcoder_components import TranscoderLinearComponents
from aspd.transcoder_components import normalize_decoder_ as normalize_transcoder_decoder_


class _ModuleLossConfig(LossMetricConfig):
    """Fields shared by the losses bound to one decomposed module."""

    module: str = ""
    """The decomposed module; empty means the run's only target (filled in by `aspd.run`)."""
    warmup_start_frac: float = 0.1
    warmup_ramp_frac: float = 0.1
    """The coefficient is 0 until `warmup_start_frac` of training and ramps linearly to 1 over the
    next `warmup_ramp_frac` (when the term's `warmup` is on).
    """
    alive_threshold: float = 0.01
    train_diag_every: int = 0


class _ModuleLoss[TCfg: _ModuleLossConfig](Metric[TCfg]):
    """Binding to the module, accumulation of logged statistics, and the warmup schedule."""

    log_namespace = "aspd"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        assert self.cfg.module, f"{type(self).__name__}: `module` was not injected"
        self.target_module = model.target_model.get_submodule(self.cfg.module)
        self.components = model.components[self.cfg.module]

    @override
    def reset(self) -> None:
        self._accum: dict[str, Tensor] = {}
        self._n = 0

    def _record(self, stats: dict[str, Tensor]) -> None:
        for key, value in stats.items():
            self._accum[key] = self._accum.get(key, torch.zeros((), device=self.device)) + value
        self._n += 1

    @override
    def compute(self) -> MetricResult:
        n = max(self._n, 1)
        return {f"{self.instance_key}/{k}": v / n for k, v in self._accum.items()}

    def _scale(self, ctx: MetricContext) -> float:
        return warmup_scale(
            ctx.step,
            ctx.total_steps,
            start_frac=self.cfg.warmup_start_frac,
            ramp_frac=self.cfg.warmup_ramp_frac,
        )

    def _emit(self, ctx: MetricContext, loss: Tensor, record: dict[str, Tensor]) -> Tensor:
        """Log `record` with the warmup scale, keep it for the train stream, return the scaled loss."""
        scale = self._scale(ctx)
        record = {
            **record,
            "warmup": torch.tensor(scale, device=self.device),
            "loss_scaled": scale * loss.detach(),
        }
        self._record(record)
        if not ctx.is_eval:
            self.train_log = record
        return scale * loss

    def pop_train_log(self) -> dict[str, Tensor]:
        """The last training batch's statistics, cleared on read (see `aspd.run.TrainDiagSink`)."""
        stashed = getattr(self, "train_log", {})
        self.train_log = {}
        return stashed


def recon_stash(model: ComponentModel) -> dict[str, dict[str, Any]]:
    """Per-step handoff from a reconstruction term to the `AuxKLoss` bound to it by name."""
    stash = getattr(model, "_aspd_recon_stash", None)
    if stash is None:
        stash = {}
        model._aspd_recon_stash = stash  # pyright: ignore[reportAttributeAccessIssue]
    return stash


def _matryoshka_groups(n: int, fracs: tuple[float, ...]) -> list[int]:
    """Prefix boundaries [0, n_1, n_1 + n_2, ..., n] for groups of `fracs * n` features."""
    sizes = [int(n * f) for f in fracs]
    sizes[-1] += n - sum(sizes)
    assert all(s > 0 for s in sizes), f"empty Matryoshka group in {sizes} at C={n}"
    return [0] + torch.cumsum(torch.tensor(sizes), dim=0).tolist()


def _prefix_mse(
    acts: Tensor, decoder: Tensor, bias: Tensor, target: Tensor, groups: list[int],
    empty_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Mean, over the empty prefix and every nested prefix, of the MSE of
    `bias + acts[:, :n] @ decoder[:n]` against `target`. Returns the loss and the full reconstruction.

    `empty_bias` (default `bias`) is the tensor used for the empty-prefix term; passing a separate
    view of the same bias fixes the order in which its gradient is accumulated.
    """
    reconstruct = bias
    terms = [((bias if empty_bias is None else empty_bias) - target).pow(2).mean()]
    for lo, hi in zip(groups[:-1], groups[1:], strict=True):
        reconstruct = acts[:, lo:hi] @ decoder[lo:hi] + reconstruct
        terms.append((reconstruct.float() - target).pow(2).mean())
    return torch.stack(terms).sum() / len(terms), reconstruct


def _fvu(recon: Tensor, target: Tensor) -> Tensor:
    resid = (recon.float() - target).pow(2).sum()
    return resid / (target - target.mean(0)).pow(2).sum().clamp_min(1e-8)


# ---- L_internal --------------------------------------------------------------------------------


class InternalReconLossConfig(_ModuleLossConfig):
    type: Literal["InternalReconLoss"] = "InternalReconLoss"
    mask: Literal["ci", "stochastic"] = "ci"
    """`ci`: component masks m = g (the gated components). `stochastic`: core's m ~ U[g, 1] draw."""
    mode: Literal["fvu", "matryoshka"] = "fvu"
    """`fvu`: ||y - y_hat||^2 / Var(y). `matryoshka`: the mean prefix MSE over nested groups of
    components (the transcoder objective; transcoder components only).
    """
    allow_single_mask: bool = False
    warmup: bool = True
    """Apply the warmup schedule (on for VPD + internal and ASPD; off for PD Transcoder, whose
    objective this term is).
    """
    group_fracs: tuple[float, ...] = DEFAULT_GROUP_FRACS
    """`matryoshka` only: prefix group sizes as fractions of C."""
    normalize_decoder: bool = False
    """Unit-normalize the rows of U after every backward (`matryoshka` on transcoder components)."""


class InternalReconLoss(_ModuleLoss[InternalReconLossConfig]):
    """L_internal: reconstruct the decomposed matrix's output from the masked components."""

    short_name = "InternalRecon"

    @override
    def _scale(self, ctx: MetricContext) -> float:
        return super()._scale(ctx) if self.cfg.warmup else 1.0

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.is_transcoder = isinstance(self.components, TranscoderLinearComponents)
        gate = gate_for(model.ci_fn, self.cfg.module)
        # Under PD Transcoder's own gate, g_c = 1 implies z_c > 0, so the encoder output is rectified.
        self.rectify_acts = self.is_transcoder and getattr(gate, "gate_implies_positive_preact", True)
        if self.cfg.mode == "matryoshka":
            self.group_indices = _matryoshka_groups(self.components.U.shape[0], self.cfg.group_fracs)

    def _mask_infos(self, ctx: MetricContext) -> dict[str, ComponentsMaskInfo]:
        if self.cfg.mask == "ci":
            return make_mask_infos(ctx.ci.lower_leaky, weight_deltas_and_masks=None)
        return calc_stochastic_component_mask_info(
            causal_importances=ctx.ci.lower_leaky,
            component_mask_sampling=ctx.sampling,
            weight_deltas=ctx.weight_deltas if ctx.use_delta_component else None,
            router=AllLayersRouter(),
        )

    def _bias(self, like: Tensor) -> Tensor:
        """The reconstruction with every component masked off: b_out, the matrix bias, or 0."""
        if self.is_transcoder:
            return self.components.b_out.expand_as(like)
        bias = self.components.bias
        if bias is None:
            return torch.zeros_like(like)
        return bias.to(like.dtype).expand_as(like)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        module = self.cfg.module
        x = ctx.pre_weight_acts[module]
        mask_info = self._mask_infos(ctx)[module]
        if self.cfg.mode == "matryoshka":
            return self._matryoshka(ctx, x, mask_info)
        y_hat = masked_module_output(self.components, self.target_module, x, mask_info)
        stats = site_recon_stats(y_hat, target_module_output(self.target_module, x))
        loss = stats["fvu"]
        recon_stash(self.model)[self.instance_key] = {"mode": "fvu"}
        return self._emit(
            ctx, loss, {**{k: v.detach() for k, v in stats.items()}, "loss": loss.detach()}
        )

    def _matryoshka(self, ctx: MetricContext, x: Tensor, mask_info: ComponentsMaskInfo) -> Tensor:
        comp = self.components
        y = target_module_output(self.target_module, x).reshape(-1, comp.d_out).float()
        z_acts = comp.get_component_acts(x)
        z = z_acts.reshape(-1, comp.C)
        m = mask_info.component_mask.reshape(-1, comp.C).to(z.dtype)
        preacts = self._encoder_preacts(x, z_acts)
        acts_pre = torch.relu(z) if self.rectify_acts else z
        acts = acts_pre * m

        loss, recon = _prefix_mse(
            acts, comp.U, self._bias(y), y, self.group_indices, empty_bias=self._bias(y)
        )
        recon_stash(self.model)[self.instance_key] = {
            "mode": "matryoshka",
            "target": y,
            "recon": recon.reshape(-1, recon.shape[-1]),
            "decoder": comp.U,
            "preacts": preacts,
            "write_acts": self._auxk_write_acts(acts_pre, preacts),
            "n_tokens": z.shape[0],
        }
        with torch.no_grad():
            fvu = _fvu(recon, y)
        return self._emit(
            ctx,
            loss,
            {"loss": loss.detach(), "l0_norm": (acts != 0).float().sum(-1).mean(), "fvu": fvu},
        )

    def _encoder_preacts(self, x: Tensor, z_acts: Tensor) -> Tensor | None:
        """What the CI function's top-k ranked, as [T, C]; None on VPD components."""
        if not self.is_transcoder:
            return None
        ci_fn = gate_for(self.model.ci_fn, self.cfg.module)
        assert isinstance(ci_fn, PDTranscoderCiFn), type(ci_fn).__name__
        return ci_fn.preacts(x, z_acts).reshape(-1, ci_fn.n_features)

    def _auxk_write_acts(self, acts_pre: Tensor, preacts: Tensor | None) -> Tensor | None:
        if preacts is None:
            return None
        ci_fn = gate_for(self.model.ci_fn, self.cfg.module)
        assert isinstance(ci_fn, PDTranscoderCiFn), type(ci_fn).__name__
        return ci_fn.auxk_write_acts(acts_pre, preacts)

    @override
    def after_backward(self) -> None:
        if self.cfg.normalize_decoder:
            normalize_transcoder_decoder_(self.components)
        recon_stash(self.model).pop(self.instance_key, None)


# ---- L_act -------------------------------------------------------------------------------------


class ActivationReconLossConfig(_ModuleLossConfig):
    type: Literal["ActivationReconLoss"] = "ActivationReconLoss"
    group_fracs: tuple[float, ...] = DEFAULT_GROUP_FRACS
    """Prefix group sizes as fractions of C."""
    normalize_decoder: bool = True
    """Unit-normalize the rows of W_dec after every backward."""
    normalize_loss: bool = True
    """Divide the prefix MSE by Var(r), so the term is in FVU units."""
    warmup: bool = False


class ActivationReconLoss(_ModuleLoss[ActivationReconLossConfig]):
    """L_act: the shared encoder's reconstruction of the residual stream r_t it reads (ASPD)."""

    short_name = "ActRecon"

    @override
    def _scale(self, ctx: MetricContext) -> float:
        return super()._scale(ctx) if self.cfg.warmup else 1.0

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        ci_fn = gate_for(model.ci_fn, self.cfg.module)
        assert isinstance(ci_fn, SharedEncoder), (
            f"ActivationReconLoss needs ASPD's shared encoder, got {type(ci_fn).__name__}"
        )
        self.ci_fn: SharedEncoder = ci_fn
        self.group_indices = _matryoshka_groups(ci_fn.n_features, self.cfg.group_fracs)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        module = self.cfg.module
        x = ctx.pre_weight_acts[module]
        enc = self.ci_fn
        mask_info = make_mask_infos(ctx.ci.lower_leaky, weight_deltas_and_masks=None)[module]
        r = enc.resid_site_acts(x).reshape(-1, enc.d_in).float()
        preacts = enc.preacts(x).reshape(-1, enc.n_features)
        m = mask_info.component_mask.reshape(-1, enc.n_features).to(preacts.dtype)
        acts = preacts * m
        bias = enc.b_dec.to(r.dtype).expand_as(r)

        loss, recon = _prefix_mse(acts, enc.W_dec, bias, r, self.group_indices)
        with torch.no_grad():
            var_r = (r - r.mean(0)).pow(2).mean().clamp_min(1e-8)
        if self.cfg.normalize_loss:
            loss = loss / var_r
        recon_stash(self.model)[self.instance_key] = {
            "mode": "matryoshka",
            "target": r,
            "recon": recon.reshape(-1, enc.d_in),
            "decoder": enc.W_dec,
            "preacts": preacts,
            "write_acts": enc.auxk_write_acts(preacts, preacts),
            "n_tokens": r.shape[0],
            "loss_scale": var_r if self.cfg.normalize_loss else torch.ones((), device=r.device),
        }
        with torch.no_grad():
            fvu = _fvu(recon, r)
        return self._emit(
            ctx,
            loss,
            {
                "loss": loss.detach(),
                "l0_norm": (acts > 0).float().sum(-1).mean(),
                "fvu": fvu,
                "resid_var": var_r,
                "decoder_norm_mean": enc.W_dec.detach().norm(dim=-1).mean(),
            },
        )

    @override
    def after_backward(self) -> None:
        if self.cfg.normalize_decoder:
            with torch.no_grad():
                W_dec = self.ci_fn.W_dec
                W_dec.div_(W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        recon_stash(self.model).pop(self.instance_key, None)


# ---- auxiliary loss ----------------------------------------------------------------------------


class AuxKLossConfig(_ModuleLossConfig):
    type: Literal["AuxKLoss"] = "AuxKLoss"
    recon: str = ""
    """Name of the Matryoshka reconstruction entry whose residual this term fits; it must come
    earlier in `loss_metrics`.
    """
    top_k_aux: PositiveInt = 512
    normalize: bool = True
    """Divide by the reconstruction entry's `loss_scale` (var(r) for ASPD's L_act with
    `normalize_loss`); false gives the raw MSE."""


class AuxKLoss(_ModuleLoss[AuxKLossConfig]):
    """AuxK (Gao et al., 2024): reconstruct a term's residual from the top-`top_k_aux` dead features."""

    short_name = "AuxK"

    @override
    def _scale(self, ctx: MetricContext) -> float:
        del ctx
        return 1.0

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        assert self.cfg.recon, "AuxKLoss needs `recon: <name of a reconstruction entry>`"
        ci_fn = gate_for(model.ci_fn, self.cfg.module)
        assert isinstance(ci_fn, PDTranscoderCiFn), (
            f"AuxKLoss needs a BatchTopK CI function (PD Transcoder or ASPD), got {type(ci_fn).__name__}"
        )
        self.dict_ci_fn: PDTranscoderCiFn = ci_fn

    @property
    def _clock(self) -> Tensor:
        return self.dict_ci_fn.n_batches_not_active

    @property
    def _n_batches_to_dead(self) -> int:
        return self.dict_ci_fn.cfg.n_batches_to_dead

    @override
    def update(self, ctx: MetricContext) -> Tensor | None:
        stash = recon_stash(self.model)
        pending = stash.pop(self.cfg.recon, None)
        if pending is None:
            assert ctx.is_eval, (
                f"AuxKLoss(recon={self.cfg.recon!r}) found no stashed residual; the named entry "
                f"must come earlier in `loss_metrics` (stashed this step: {sorted(stash)})"
            )
            return None
        assert pending["mode"] == "matryoshka" and pending["preacts"] is not None, (
            f"AuxKLoss(recon={self.cfg.recon!r}) must name a Matryoshka term on transcoder "
            "components or ASPD's ActivationReconLoss"
        )
        loss = self._auxiliary_loss(
            pending["target"],
            pending["recon"],
            pending["preacts"],
            pending["write_acts"],
            pending["decoder"],
        )
        loss_scale = pending.get("loss_scale")
        if self.cfg.normalize and loss_scale is not None:
            loss = loss / loss_scale
        with torch.no_grad():
            clock = self._clock
            dead = clock >= self._n_batches_to_dead
            n_pools = max(1, math.ceil(int(pending["n_tokens"]) / self.dict_ci_fn.cfg.pool_tokens))
        return self._emit(
            ctx,
            loss,
            {
                "loss": loss.detach(),
                "n_dead": dead.sum().float(),
                "dead_frac": dead.float().mean(),
                "never_fired_step_frac": (clock >= n_pools).float().mean(),
                "threshold": self.dict_ci_fn.threshold.detach().clone(),
            },
        )

    def _auxiliary_loss(
        self, target: Tensor, recon: Tensor, acts: Tensor, write_acts: Tensor, decoder: Tensor
    ) -> Tensor:
        residual = target.float() - recon.float()
        dead = self._clock >= self._n_batches_to_dead
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return torch.zeros((), device=target.device)
        top = torch.topk(acts[:, dead], min(self.cfg.top_k_aux, n_dead), dim=-1)
        write_dead = write_acts[:, dead]
        acts_aux = torch.zeros_like(write_dead).scatter(
            -1, top.indices, write_dead.gather(-1, top.indices)
        )
        aux_reconstruct = acts_aux @ decoder[dead]
        if aux_reconstruct.abs().sum() == 0:
            return torch.zeros((), device=target.device)
        return (aux_reconstruct.float() - residual).pow(2).mean()


# ---- alive tracker (not a loss) ----------------------------------------------------------------


class ComponentAliveTrackerConfig(LossMetricConfig):
    type: Literal["ComponentAliveTracker"] = "ComponentAliveTracker"
    module: str = ""
    window_tokens: int = 10_000_000


class ComponentAliveTracker(Metric[ComponentAliveTrackerConfig]):
    """Tracks, per component, whether it fired (g > 0) within the last `window_tokens` training tokens."""

    log_namespace = "aspd"
    short_name = "Alive"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        assert self.cfg.module, "ComponentAliveTracker: `module` was not injected"
        n_components = model.components[self.cfg.module].U.shape[0]
        self.tracker = AliveTracker(n_components, self.cfg.window_tokens).to(device)

    @override
    def reset(self) -> None:
        self._alive_frac = torch.zeros((), device=getattr(self, "device", "cpu"))

    @override
    def update(self, ctx: MetricContext) -> None:
        if not ctx.is_eval:
            self.tracker.observe(ctx.ci.lower_leaky[self.cfg.module])
        self._alive_frac = self.tracker.alive_frac

    def pop_train_log(self) -> dict[str, Tensor]:
        return {"alive_frac": self._alive_frac, "n_alive": self.tracker.alive.sum().float()}

    @override
    def compute(self) -> MetricResult:
        return {
            f"{self.instance_key}/alive_frac": self._alive_frac,
            f"{self.instance_key}/n_alive": self.tracker.alive.sum().float(),
        }

    @override
    def state_dict(self) -> dict[str, Any]:
        return self.tracker.state_dict()

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.tracker.load_state_dict(state)


LOSS_METRIC_CLASSES["ComponentAliveTracker"] = ComponentAliveTracker
LOSS_METRIC_CLASSES["InternalReconLoss"] = InternalReconLoss
LOSS_METRIC_CLASSES["ActivationReconLoss"] = ActivationReconLoss
LOSS_METRIC_CLASSES["AuxKLoss"] = AuxKLoss
LOSS_METRIC_CLASSES["AdaptiveSparsityLoss"] = AdaptiveSparsityLoss

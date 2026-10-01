"""Adaptive sparsity loss: core's `ImportanceMinimalityLoss` (L_sparse) with a multiplier m on its
coefficient, adjusted during training so the measured L0 tracks `target_l0`.

Each step: L0_ema <- beta L0_ema + (1 - beta) L0, e = log(L0_ema / target_l0), and unless
|e| < log(band), log m <- clip(log m + kappa(e) k_i tanh(gain_scale e), log m_min, log m_max) with
kappa(e) = 1 for e > 0 and `kappa_loosen` otherwise. The controller runs between
`freeze_start_frac` and `freeze_end_frac` of training.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal, Self, override

import torch
from param_decomp.base_config import Probability, runtime_cast
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce, get_distributed_state
from param_decomp.log import logger
from param_decomp.metrics.base import MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.importance_minimality import (
    ImportanceMinimalityLoss,
    ImportanceMinimalityLossConfig,
)
from pydantic import PositiveFloat, model_validator
from torch import Tensor
from torch.distributed import ReduceOp


class AdaptiveSparsityLossConfig(ImportanceMinimalityLossConfig):
    """`ImportanceMinimalityLossConfig` plus the L0 controller; `coeff` is the initial coefficient."""

    type: Literal["AdaptiveSparsityLoss"] = "AdaptiveSparsityLoss"  # pyright: ignore[reportIncompatibleVariableOverride]

    target_l0: PositiveFloat
    """Target L0 (mean components with g > `ci_alive_threshold` per token)."""

    band: float = 1.15
    """Dead band as a ratio: no update while L0_ema / target_l0 is within [1/band, band]."""

    ema_beta: Probability = 0.99
    """EMA factor of the measured L0."""

    k_i: PositiveFloat = 3.0e-4
    """Maximum change of log m per step."""

    gain_scale: PositiveFloat = 10.0
    """Slope of tanh(gain_scale * e): the response is proportional within |e| < 1/gain_scale."""

    kappa_loosen: PositiveFloat = 3.0
    """Gain multiplier applied when lowering m (e < 0)."""

    m_min: PositiveFloat = 1.0e-3
    m_max: PositiveFloat = 1.0e3
    """Clip range [m_min, m_max] of the multiplier."""

    freeze_start_frac: Probability = 0.1
    """The controller starts at this fraction of training."""

    freeze_end_frac: Probability = 1.0
    """The controller stops at this fraction of training."""

    ci_alive_threshold: float = 0.01
    """L0 counts components with g_{t,c} above this threshold."""

    @model_validator(mode="after")
    def _validate_controller(self) -> Self:
        assert self.band > 1.0, f"band is a ratio and must exceed 1.0, got {self.band}"
        assert self.m_min < self.m_max, f"m_min {self.m_min} >= m_max {self.m_max}"
        assert self.m_min <= 1.0 <= self.m_max, (
            f"the multiplier starts at 1.0 (coeff means lambda_0), so the clip range "
            f"[{self.m_min}, {self.m_max}] must contain it"
        )
        assert self.freeze_start_frac < self.freeze_end_frac, (
            f"freeze_start_frac {self.freeze_start_frac} >= freeze_end_frac "
            f"{self.freeze_end_frac}: the controller would never run"
        )
        return self


def measure_ci_l0(
    ci: dict[str, Tensor], thresholds: Sequence[float], *, device: str | torch.device
) -> Tensor:
    """L0 at each of `thresholds`: mean CI entries above it per token, summed over layers, DP-mean."""
    assert ci, "empty ci"
    assert thresholds, "no thresholds"
    totals = []
    for threshold in thresholds:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for layer_ci in ci.values():
            total = total + (layer_ci.detach() > threshold).sum(-1, dtype=torch.float32).mean()
        totals.append(total)
    out = torch.stack(totals)
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    if world_size > 1:
        out = all_reduce(out, op=ReduceOp.SUM) / world_size
    return out


class AdaptiveSparsityLoss(ImportanceMinimalityLoss):
    """L_sparse scaled by the controller's multiplier m."""

    short_name = "AdaptiveSparsity"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.ctrl = runtime_cast(AdaptiveSparsityLossConfig, self.cfg)
        self._thresholds: tuple[float, ...] = (
            (self.ctrl.ci_alive_threshold,)
            if self.ctrl.ci_alive_threshold == 0.0
            else (self.ctrl.ci_alive_threshold, 0.0)
        )
        self._log_m = 0.0
        self._l0_ema: float | None = None
        self._l0_last = float("nan")
        self._l0_at_zero_last = float("nan")
        self._frozen = True
        self._clipped_since: int | None = None
        self.train_log: dict[str, Tensor] = {}

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        loss = super().update(ctx)
        l0 = measure_ci_l0(ctx.ci.lower_leaky, self._thresholds, device=self.device)
        self._l0_at_zero_last = float(l0[-1].item())
        if not ctx.is_eval:
            self._step_controller(ctx, float(l0[0].item()))
        m = math.exp(self._log_m)
        record = self._record_dict(loss, m)
        if not ctx.is_eval:
            self.train_log = record
        return m * loss

    def _step_controller(self, ctx: MetricContext, l0: float) -> None:
        c = self.ctrl
        self._l0_last = l0
        self._l0_ema = l0 if self._l0_ema is None else c.ema_beta * self._l0_ema + (1 - c.ema_beta) * l0

        frac = ctx.current_frac_of_training
        self._frozen = frac < c.freeze_start_frac or frac >= c.freeze_end_frac
        if self._frozen:
            return

        e = math.log(max(self._l0_ema, 1e-6) / c.target_l0)
        if abs(e) < math.log(c.band):
            return
        kappa = 1.0 if e > 0.0 else c.kappa_loosen
        proposed = self._log_m + kappa * c.k_i * math.tanh(c.gain_scale * e)
        self._log_m = min(max(proposed, math.log(c.m_min)), math.log(c.m_max))
        self._note_clip(ctx, proposed)

    def _note_clip(self, ctx: MetricContext, proposed: float) -> None:
        """Warn ONCE per excursion when the multiplier saturates."""
        at_clip = proposed != self._log_m
        if not at_clip:
            self._clipped_since = None
            return
        if self._clipped_since is None:
            self._clipped_since = ctx.step
            logger.warning(
                f"{self.instance_key}: multiplier hit its clip at step {ctx.step} "
                f"(m = {math.exp(self._log_m):.4g}, L0_ema = {self._l0_ema:.1f} vs target "
                f"{self.ctrl.target_l0:g}); widen [m_min, m_max] to reach the target."
            )

    def _record_dict(self, loss: Tensor, m: float) -> dict[str, Tensor]:
        """The controller state, logged as tensors."""

        def t(value: float) -> Tensor:
            return torch.tensor(value, device=self.device, dtype=torch.float32)

        ema = self._l0_ema if self._l0_ema is not None else float("nan")
        return {
            "m": t(m),
            "log_m": t(self._log_m),
            "coeff_effective": t((self.cfg.coeff or 0.0) * m),
            "l0": t(self._l0_last),
            "l0_at_zero": t(self._l0_at_zero_last),
            "l0_ema": t(ema),
            "l0_err_log": t(math.log(max(ema, 1e-6) / self.ctrl.target_l0) if ema == ema else 0.0),
            "controller_frozen": t(float(self._frozen)),
            "at_clip": t(float(self._clipped_since is not None)),
            "loss_unscaled": loss.detach().float(),
            "loss_scaled": (m * loss).detach().float(),
        }

    def pop_train_log(self) -> dict[str, Tensor]:
        """Hand the last training batch's controller state to the sink wrapper and clear it."""
        stashed = self.train_log
        self.train_log = {}
        return stashed

    @override
    def compute(self) -> MetricResult:
        """The parent's two keys plus the controller's state."""
        out = dict(runtime_cast(dict, super().compute()))
        name = type(self).__name__
        ema = self._l0_ema if self._l0_ema is not None else float("nan")
        out[f"{name}_m"] = math.exp(self._log_m)
        out[f"{name}_l0_ema"] = ema
        out[f"{name}_l0_at_zero"] = self._l0_at_zero_last
        out[f"{name}_coeff_effective"] = (self.cfg.coeff or 0.0) * math.exp(self._log_m)
        return out

    @override
    def state_dict(self) -> dict[str, Any]:
        """The controller state (log m, L0_ema), saved with training checkpoints for resuming."""
        return {"log_m": self._log_m, "l0_ema": self._l0_ema}

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._log_m = float(state["log_m"])
        l0_ema = state["l0_ema"]
        self._l0_ema = None if l0_ema is None else float(l0_ema)

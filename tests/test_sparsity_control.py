"""The adaptive L0 controller on log m: every branch of the update and where it must not act."""

import math
from typing import Any, cast

import pytest
import torch
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.configs import PDConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.importance_minimality import (
    ImportanceMinimalityLoss,
    ImportanceMinimalityLossConfig,
)
from pydantic import ValidationError

from aspd.adaptive_l0 import (
    AdaptiveSparsityLoss,
    AdaptiveSparsityLossConfig,
    measure_ci_l0,
)
from aspd.config import PDConfigShared

MODULE = "transformer.h.0.mlp.c_fc"
B, S, C = 2, 8, 64
TARGET = 32.0


def _ci(n_on: int) -> torch.Tensor:
    """`[B, S, C]` causal importances with exactly `n_on` entries per token strictly above 0."""
    ci = torch.zeros(B, S, C)
    ci[..., :n_on] = 0.7
    return ci


def _ctx(ci: torch.Tensor, *, step: int = 500, total_steps: int = 1000, is_eval: bool = False):
    outputs = CIOutputs(
        lower_leaky={MODULE: ci}, upper_leaky={MODULE: ci}, pre_sigmoid={MODULE: ci}
    )
    return MetricContext(
        model=cast(ComponentModel, None),
        batch=None,
        target_out=torch.zeros(()),
        pre_weight_acts={},
        ci=outputs,
        weight_deltas={},
        step=step,
        total_steps=total_steps,
        use_delta_component=False,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=lambda p, t: (torch.zeros(()), 1),
        is_eval=is_eval,
    )


def _metric(**overrides: Any) -> AdaptiveSparsityLoss:
    fields: dict[str, Any] = {
        "coeff": 1e-5,
        "pnorm": 2.0,
        "beta": 0.5,
        "target_l0": TARGET,
        "ci_alive_threshold": 0.0,
        "freeze_start_frac": 0.0,
        "freeze_end_frac": 1.0,
    }
    fields.update(overrides)
    metric = AdaptiveSparsityLoss(AdaptiveSparsityLossConfig(**fields))
    metric.bind(model=cast(ComponentModel, None), device="cpu")
    return metric


def test_too_dense_raises_the_multiplier():
    metric = _metric()
    metric.update(_ctx(_ci(n_on=C)))
    assert metric._log_m > 0.0


def test_too_sparse_lowers_the_multiplier():
    metric = _metric()
    metric.update(_ctx(_ci(n_on=8)))
    assert metric._log_m < 0.0


def test_loosening_uses_the_larger_gain():
    """The plant is asymmetric -- clamped components take no gradient from this term at ANY
    coefficient -- so the loop has to be too. A symmetric gain here would mean the controller
    tightens and loosens at the same speed while the model only responds to one of them.
    """
    dense, sparse = _metric(gain_scale=100.0), _metric(gain_scale=100.0)
    dense.update(_ctx(_ci(n_on=C)))
    sparse.update(_ctx(_ci(n_on=1)))
    assert sparse._log_m == pytest.approx(-3.0 * dense._log_m, rel=1e-9)


def test_the_deadband_is_idle():
    metric = _metric()
    metric.update(_ctx(_ci(n_on=int(TARGET))))
    assert metric._log_m == 0.0
    # Just inside the band on the dense side: 35/32 = 1.09 < 1.15.
    metric.update(_ctx(_ci(n_on=35)))
    assert metric._log_m == 0.0


def test_the_multiplier_is_clipped_and_reported():
    """A pinned multiplier and a converged one look identical in the trajectory and mean opposite
    things, so the clip has to be observable rather than merely enforced.
    """
    metric = _metric(m_max=1.0, k_i=1.0)
    metric.update(_ctx(_ci(n_on=C)))
    assert metric._log_m == 0.0, "clipped at m_max"
    assert metric._clipped_since is not None
    assert bool(metric.pop_train_log()["at_clip"].item())

    floor = _metric(m_min=1.0, k_i=1.0)
    floor.update(_ctx(_ci(n_on=1)))
    assert floor._log_m == 0.0, "clipped at m_min"


def test_total_collapse_does_not_produce_a_nan():
    """L0 exactly 0 is `log 0`. Unguarded that is `-inf`, and `-inf` reaches `m` as a NaN -- which
    core's `assert torch.isfinite(loss)` would catch only via the loss, several frames away.
    """
    metric = _metric()
    metric.update(_ctx(torch.zeros(B, S, C)))
    assert math.isfinite(metric._log_m)
    assert metric._log_m < 0.0, "an empty decomposition must loosen, not tighten"


def test_eval_batches_are_measured_but_never_control():
    metric = _metric()
    metric.update(_ctx(_ci(n_on=C), is_eval=True))
    assert metric._log_m == 0.0
    assert metric._l0_ema is None, "an eval batch must not enter the integrator"


@pytest.mark.parametrize("step", [0, 50, 950, 999])
def test_the_freeze_windows_hold_the_multiplier(step: int):
    metric = _metric(freeze_start_frac=0.1, freeze_end_frac=0.9)
    metric.update(_ctx(_ci(n_on=C), step=step, total_steps=1000))
    assert metric._log_m == 0.0
    assert metric._l0_ema is not None, "the EMA must stay warm through a freeze"


def test_reset_between_eval_passes_does_not_reset_the_controller():
    """`Metric.reset()` runs before every eval pass. Controller state in `reset()` would restart the
    integrator at `m = 1` a hundred times per run and the curve would still look plausible.
    """
    metric = _metric()
    metric.update(_ctx(_ci(n_on=C)))
    moved = metric._log_m
    metric.reset()
    assert metric._log_m == moved
    assert metric._l0_ema is not None


def test_returned_loss_is_the_multiplier_times_the_unmodified_penalty():
    """The penalty itself must be the parent's, bit for bit -- this arm changes the coefficient and
    nothing else about what VPD minimizes.
    """
    fields = dict(coeff=1e-5, pnorm=2.0, beta=0.5)
    reference = ImportanceMinimalityLoss(ImportanceMinimalityLossConfig(**fields))
    reference.bind(model=cast(ComponentModel, None), device="cpu")

    metric = _metric(**fields)
    metric._log_m = math.log(7.0)

    ci = _ci(n_on=int(TARGET))
    got = metric.update(_ctx(ci))
    want = reference.update(_ctx(ci))
    assert got.item() == pytest.approx(7.0 * want.item(), rel=1e-6)

    record = metric.pop_train_log()
    assert record["loss_unscaled"].item() == pytest.approx(want.item(), rel=1e-6)
    assert record["coeff_effective"].item() == pytest.approx(7.0e-5, rel=1e-6)


def test_train_log_is_cleared_on_read():
    metric = _metric()
    metric.update(_ctx(_ci(n_on=40)))
    assert metric.pop_train_log()
    assert metric.pop_train_log() == {}, "a step that did not run must not re-log the last one"


def test_state_round_trips_across_a_restart():
    metric = _metric()
    for _ in range(5):
        metric.update(_ctx(_ci(n_on=C)))
    state = metric.state_dict()

    restarted = _metric()
    restarted.load_state_dict(state)
    assert restarted._log_m == metric._log_m
    assert restarted._l0_ema == metric._l0_ema


def test_measured_l0_is_the_same_number_ci_l0_reports():
    """The controlled quantity and the dashboard quantity must be one quantity, or a misbehaving
    loop cannot be read off the run at all.
    """
    ci_l0 = pytest.importorskip("param_decomp_lab.eval_metrics.ci_l0")
    ci = _ci(n_on=13)
    got = measure_ci_l0({MODULE: ci}, (0.0,), device="cpu")
    assert got[0].item() == pytest.approx(ci_l0.calc_ci_l_zero(ci, 0.0))
    assert got[0].item() == pytest.approx(13.0)


def test_l0_is_read_off_lower_leaky():
    """`upper_leaky` leaks above 1 and `lower_leaky` clamps there; `> 0` is unaffected, which is why
    reading the reported tensor costs nothing. Pinned so a future switch is deliberate.
    """
    ci = _ci(n_on=5)
    ci[..., :2] = 3.0  # only `upper_leaky` would ever hold values above 1
    assert measure_ci_l0({MODULE: ci}, (0.0,), device="cpu")[0].item() == pytest.approx(5.0)


def test_every_threshold_is_answered_off_one_batch_in_order():
    """The controlled count and the reported `> 0` count have to describe the SAME forward, or
    their ratio -- the whole reason both are logged -- is noise.
    """
    ci = torch.zeros(B, S, C)
    ci[..., :10] = 0.7  # clearly on
    ci[..., 10:40] = 0.01  # on at `> 0`, off at `> 0.05`: the mass the sensor choice is about
    got = measure_ci_l0({MODULE: ci}, (0.05, 0.0), device="cpu")
    assert got.shape == (2,)
    assert got[0].item() == pytest.approx(10.0)
    assert got[-1].item() == pytest.approx(40.0)


def test_a_positive_sensor_controls_on_the_count_above_it():
    ci = torch.zeros(B, S, C)
    ci[..., :10] = 0.7
    ci[..., 10:40] = 0.01

    at_zero = _metric(ci_alive_threshold=0.0)
    at_zero.update(_ctx(ci))
    assert at_zero._log_m > 0.0, "40 > 32 -> tighten"

    at_005 = _metric(ci_alive_threshold=0.05)
    at_005.update(_ctx(ci))
    assert at_005._log_m < 0.0, "10 < 32 -> loosen"
    # ... and the `> 0` count is still reported, so the gap is readable rather than inferred.
    assert at_005.pop_train_log()["l0_at_zero"].item() == pytest.approx(40.0)


def test_the_zero_threshold_count_is_logged_without_a_second_pass():
    """At the `0.0` sensor the two numbers coincide, and paying for a second count would be a
    per-step cost on every arm that does not need one.
    """
    metric = _metric(ci_alive_threshold=0.0)
    assert metric._thresholds == (0.0,)
    metric.update(_ctx(_ci(n_on=40)))
    record = metric.pop_train_log()
    assert record["l0"].item() == pytest.approx(record["l0_at_zero"].item())

    positive = _metric(ci_alive_threshold=0.05)
    assert positive._thresholds == (0.05, 0.0)


def test_control_runs_to_the_last_step_when_the_end_freeze_is_off():
    metric = _metric(freeze_start_frac=0.1, freeze_end_frac=1.0)
    metric.update(_ctx(_ci(n_on=C), step=999, total_steps=1000))
    assert metric._log_m > 0.0
    assert not metric._frozen


@pytest.mark.parametrize(
    "bad",
    [
        {"band": 1.0},
        {"m_min": 2.0, "m_max": 10.0},
        {"m_min": 10.0, "m_max": 1.0},
        {"freeze_start_frac": 0.9, "freeze_end_frac": 0.1},
    ],
)
def test_a_controller_that_could_never_run_is_refused_at_parse(bad: dict[str, Any]):
    with pytest.raises(ValidationError):
        AdaptiveSparsityLossConfig(coeff=1e-5, pnorm=2.0, beta=0.5, target_l0=32, **bad)


# ---- the two-part registration ------------------------------------------------------------------


def _pd_kwargs() -> dict[str, Any]:
    return {
        "seed": 0,
        "steps": 4,
        "batch_size": 2,
        "n_mask_samples": 1,
        "sampling": "continuous",
        "sigmoid_type": "leaky_hard",
        "use_delta_component": False,
        "decomposition_targets": [{"module_pattern": MODULE, "C": C}],
        "ci_config": {"mode": "global", "fn_type": "global_shared_mlp", "hidden_dims": [8]},
        "components_optimizer": {"lr_schedule": {"start_val": 1e-3}},
        "ci_fn_optimizer": {"lr_schedule": {"start_val": 3e-3}},
    }


_ENTRY = {
    "type": "AdaptiveSparsityLoss",
    "coeff": 1e-5,
    "pnorm": 2.0,
    "beta": 0.5,
    "target_l0": 32,
}


def test_a_stock_pdconfig_rejects_the_adaptive_type():
    with pytest.raises(ValidationError):
        PDConfig(**_pd_kwargs(), loss_metrics=[_ENTRY])


def test_the_widened_pdconfig_accepts_it():
    cfg = PDConfigShared(**_pd_kwargs(), loss_metrics=[_ENTRY])
    assert isinstance(cfg.loss_metrics[0], AdaptiveSparsityLossConfig)

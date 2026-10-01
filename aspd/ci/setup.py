"""Registry of the non-core CI functions: how to build, calibrate, seed and attach each one."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from param_decomp.component_model import ComponentModel
from torch import Tensor, nn

from aspd.ci import (
    ASPDCiConfig,
    ASPDCiFn,
    ASPDCiFnSet,
    PDTranscoderCiConfig,
    PDTranscoderCiFn,
    PDTranscoderCiFnSet,
    SharedEncoder,
    make_aspd_ci_fn,
    make_transcoder_ci_fn,
)
from aspd.ci.aspd_setup import (
    attach_aspd_ci_fn,
    calibrate_aspd_encoder,
    seed_aspd_ci_fn,
)
from aspd.ci.pd_transcoder_setup import (
    attach_transcoder_ci_fn,
    calibrate_transcoder_scale,
    seed_transcoder_ci_fn,
)
from aspd.config import LMInterpExperimentConfig


@dataclass(frozen=True)
class CiFnSpec:
    """How to build, calibrate, seed and attach one non-core CI function."""

    ci_fn_type: type[nn.Module]
    build: Callable[..., nn.Module]
    """`make_ci_fn_wrapper`'s branch. Keyword-only `(target_model, module_to_c, ci_config)`."""
    calibrate: Callable[..., tuple[Tensor, dict[str, float]]]
    """`(cfg, target_model, module, batches, device) -> (calibration, stats)`."""
    seed: Callable[..., Any]
    """`(component_model, calibration)`. Called AFTER `Trainer(...)`."""
    attach: Callable[..., None] | None = None
    """`(component_model)`. Wiring a checkpoint cannot carry; run on every path that builds a model."""


CI_FN_SPECS: dict[type, CiFnSpec] = {
    PDTranscoderCiConfig: CiFnSpec(
        ci_fn_type=PDTranscoderCiFn,
        build=make_transcoder_ci_fn,
        calibrate=calibrate_transcoder_scale,
        seed=seed_transcoder_ci_fn,
        attach=attach_transcoder_ci_fn,
    ),
    ASPDCiConfig: CiFnSpec(
        ci_fn_type=ASPDCiFn,
        build=make_aspd_ci_fn,
        calibrate=calibrate_aspd_encoder,
        seed=seed_aspd_ci_fn,
        attach=attach_aspd_ci_fn,
    ),
}

_BY_CI_FN_TYPE: dict[type, CiFnSpec] = {s.ci_fn_type: s for s in CI_FN_SPECS.values()}
_BY_CI_FN_TYPE[PDTranscoderCiFnSet] = CI_FN_SPECS[PDTranscoderCiConfig]
_BY_CI_FN_TYPE[ASPDCiFnSet] = CI_FN_SPECS[ASPDCiConfig]

READ_SITE_DETACHERS: dict[type, Callable[[Any], Any]] = {
    SharedEncoder: SharedEncoder.detach_resid_site,
    ASPDCiFnSet: ASPDCiFnSet.detach_resid_site,
}


def detach_read_site(component_model: ComponentModel) -> None:
    """Remove the CI function's target-model hooks, if any. Idempotent."""
    for ci_fn_type, detach in READ_SITE_DETACHERS.items():
        if isinstance(component_model.ci_fn, ci_fn_type):
            detach(component_model.ci_fn)
            return


def install_ci_fns() -> None:
    """Patch core's `make_ci_fn_wrapper` to build the CI functions in `CI_FN_SPECS`. Idempotent."""
    from param_decomp import component_model

    from aspd.capture_guard import install_clean_capture_guard

    install_clean_capture_guard()

    if getattr(component_model.make_ci_fn_wrapper, "_aspd_ci_patched", False):
        return
    stock = component_model.make_ci_fn_wrapper

    def _make_ci_fn_wrapper(
        *,
        target_model: nn.Module,
        module_to_c: dict[str, int],
        components: dict[str, Any],
        ci_config: Any,
    ) -> nn.Module:
        spec = CI_FN_SPECS.get(type(ci_config))
        if spec is not None:
            return spec.build(
                target_model=target_model, module_to_c=module_to_c, ci_config=ci_config
            )
        return stock(
            target_model=target_model,
            module_to_c=module_to_c,
            components=components,
            ci_config=ci_config,
        )

    _make_ci_fn_wrapper._aspd_ci_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    component_model.make_ci_fn_wrapper = _make_ci_fn_wrapper  # pyright: ignore[reportAttributeAccessIssue]


def spec_for(cfg: LMInterpExperimentConfig) -> CiFnSpec | None:
    return CI_FN_SPECS.get(type(cfg.pd.ci_config))


def uses_custom_ci_fn(cfg: LMInterpExperimentConfig) -> bool:
    """Whether the config uses one of this package's CI functions (PD Transcoder or ASPD)."""
    return spec_for(cfg) is not None


def calibrate_ci_scale(
    cfg: LMInterpExperimentConfig,
    target_model: nn.Module,
    module: str,
    batches: Iterator[Tensor],
    device: str,
) -> tuple[Tensor, dict[str, float]]:
    """Run the CI function's calibration pass (ASPD: E[r] per residual site)."""
    spec = spec_for(cfg)
    assert spec is not None, (
        f"`{cfg.pd.ci_config.mode}` is a core CI fn and has no per-latent scale to calibrate; "
        "guard the call with `uses_custom_ci_fn`"
    )
    return spec.calibrate(cfg, target_model, module, batches, device)


def seed_ci_fn(
    cfg: LMInterpExperimentConfig,
    component_model: ComponentModel,
    scale: Tensor,
) -> None:
    """Initialize the freshly built CI function from its calibration and attach it."""
    spec = spec_for(cfg)
    assert spec is not None, f"`{cfg.pd.ci_config.mode}` is a core CI fn and has nothing to seed"
    spec.seed(component_model, scale)


def attach_ci_fn(component_model: ComponentModel) -> None:
    spec = _BY_CI_FN_TYPE.get(type(component_model.ci_fn))
    if spec is None or spec.attach is None:
        return
    spec.attach(component_model)



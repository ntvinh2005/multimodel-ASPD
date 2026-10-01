"""Build and tie PD Transcoder's CI function to the components it gates (no calibration needed)."""

from collections.abc import Iterator
from typing import Any

import torch
from param_decomp.component_model import ComponentModel
from param_decomp.log import logger
from torch import Tensor

from aspd.ci.pd_transcoder import PDTranscoderCiFn, PDTranscoderCiFnSet
from aspd.config import LMInterpExperimentConfig


def calibrate_transcoder_scale(
    cfg: LMInterpExperimentConfig,
    target_model: Any,
    module: str,
    batches: Iterator[Tensor],
    device: str,
) -> tuple[Tensor, dict[str, float]]:
    """No calibration: BatchTopK's threshold is fitted during training."""
    del cfg, target_model, module, batches
    return torch.zeros(0, device=device), {"calibration": 0.0}


def seed_transcoder_ci_fn(
    component_model: ComponentModel, scale: Tensor
) -> PDTranscoderCiFn | PDTranscoderCiFnSet:
    """Attach the CI function to the components it gates; run after the Trainer is built."""
    del scale
    ci_fn = component_model.ci_fn
    assert isinstance(ci_fn, PDTranscoderCiFn | PDTranscoderCiFnSet), (
        f"seed_transcoder_ci_fn got a {type(ci_fn).__name__}; the registry dispatched wrongly"
    )
    if isinstance(ci_fn, PDTranscoderCiFnSet):
        ci_fn.attach_components(dict(component_model.components))
        widths = {m: tuple(c.V.shape) for m, c in component_model.components.items()}
        logger.info(
            f"PD Transcoder gate attached over {len(widths)} modules, top_k={ci_fn.cfg.top_k} "
            f"pool_tokens={ci_fn.cfg.pool_tokens}; V shapes {sorted(set(widths.values()))}"
        )
        return ci_fn

    assert len(component_model.components) == 1, (
        f"a bare PDTranscoderCiFn gates one module, got {sorted(component_model.components)}. "
        "`make_transcoder_ci_fn` returns a PDTranscoderCiFnSet above one target -- this means the "
        "CI fn was built before the extra targets existed."
    )
    components = next(iter(component_model.components.values()))
    ci_fn.attach_components(components)
    logger.info(
        f"PD Transcoder gate attached: V={tuple(components.V.shape)} U={tuple(components.U.shape)} "
        f"top_k={ci_fn.cfg.top_k} pool_tokens={ci_fn.cfg.pool_tokens}"
    )
    return ci_fn


def attach_transcoder_ci_fn(component_model: ComponentModel) -> None:
    """Re-attach after `load_state_dict` (the tie is a Python reference, not a tensor). Idempotent."""
    seed_transcoder_ci_fn(component_model, torch.zeros(0))



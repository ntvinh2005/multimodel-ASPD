"""Compatibility boundary between streamed original-ASPD checkpoints and P6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.cache import snapshot_model_weights
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_recipe import validate_original_recipe_config
from aspd.multimodel.original_streaming import resolved_config_sha256


def load_original_decomposition_for_p6(
    cfg: MultiModelExperimentConfig,
    checkpoint_rank0: str | Path,
    source_models: Sequence[nn.Module],
    source_provenance: Sequence[Mapping[str, Any]],
    device: str | torch.device,
) -> MultiModelASPD:
    """Load the canonical rank-0 decomposition against freshly audited source weights."""

    validate_original_recipe_config(cfg)
    if len(source_models) != 2 or len(source_provenance) != 2:
        raise ValueError("P6 requires the frozen Base/Target source-model pair")
    weights = [
        snapshot_model_weights(model, spec, provenance)[0]
        for model, spec, provenance in zip(
            source_models, cfg.models, source_provenance, strict=True
        )
    ]
    width = int(weights[0]["o_proj"].shape[0])
    decomposition = MultiModelASPD(
        [model.name for model in cfg.models],
        [width, width],
        weights,
        cfg.encoder,
        cfg.sparsity,
        cfg.objective,
        cfg.original_aspd,
    )
    state = torch.load(
        checkpoint_rank0, map_location="cpu", weights_only=False, mmap=True
    )
    if state.get("config_sha256") != resolved_config_sha256(cfg):
        raise ValueError("P6 checkpoint/config fingerprint mismatch")
    decomposition.load_state_dict(state["model"], strict=True)
    return decomposition.to(device).float().eval()


@torch.no_grad()
def freeze_clean_original_gate(
    decomposition: MultiModelASPD,
    clean_residuals: Sequence[Tensor],
    valid_tokens: Tensor,
    component_ids: Sequence[int],
) -> Tensor:
    """Compute selected learned-threshold gates once, before any intervention hook."""

    if decomposition.training:
        raise ValueError("clean P6 gates require an eval-mode decomposition")
    if not component_ids or len(component_ids) != len(set(component_ids)):
        raise ValueError("component_ids must be non-empty and unique")
    selected = torch.tensor(
        component_ids,
        dtype=torch.long,
        device=next(decomposition.parameters()).device,
    )
    code = decomposition.encode(
        clean_residuals, valid_tokens, update_statistics=False
    )
    return code.gate.index_select(-1, selected).detach().clone()


def steering_compatibility_report() -> dict[str, Any]:
    return {
        "core_p6_subtraction": "COMPATIBLE_WITH_TRANSCODER_B_DEC",
        "clean_gate": "COMPATIBLE_AND_FROZEN_BEFORE_INTERVENTION",
        "gate_off": "ZERO_BY_CONSTRUCTION_AND_TESTED",
        "same_l2_random_multiple_seeds": "SUPPORTED",
        "active_matched_control": "SUPPORTED_BY_HISTORICAL_PIPELINE",
        "orthogonal_matched_control": "SUPPORTED",
        "source_weight_mutation_guard": "SUPPORTED",
        "historical_fixed_orchestrators": (
            "INTENTIONALLY_NOT_REUSED: old cache normalization, fixed component IDs, and "
            "historical checkpoint provenance are a different scientific experiment"
        ),
        "new_record_schema": "docs/multimodel/p6_original_recipe_record.schema.json",
    }

"""Pure helpers for ASPD v3 gate-collapse gradient diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def masked_gradient_summary(values: Tensor, mask: Tensor) -> dict[str, float | int]:
    """Summarize gradient values on an explicitly defined event/feature mask."""

    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("gradient values and boolean mask must have identical shapes")
    selected = values.detach().float()[mask]
    if selected.numel() == 0:
        return {
            "count": 0,
            "nonzero_fraction": 0.0,
            "mean_absolute": 0.0,
            "rms": 0.0,
            "max_absolute": 0.0,
        }
    absolute = selected.abs()
    return {
        "count": int(selected.numel()),
        "nonzero_fraction": float((selected != 0).float().mean()),
        "mean_absolute": float(absolute.mean()),
        "rms": float(selected.square().mean().sqrt()),
        "max_absolute": float(absolute.max()),
    }


def feature_group_masks(
    logits: Tensor,
    selected: Tensor,
    dead_features: Tensor,
    training_fire_counts: Tensor,
) -> dict[str, Tensor]:
    """Define overlapping feature populations without treating forced use as health."""

    if logits.ndim != 3 or selected.shape != logits.shape or selected.dtype != torch.bool:
        raise ValueError("logits and selected support must share [B,A,C]")
    features = logits.shape[-1]
    if dead_features.shape != (features,) or dead_features.dtype != torch.bool:
        raise ValueError("dead-feature mask must be boolean [C]")
    if training_fire_counts.shape != (features,):
        raise ValueError("training firing counts must have shape [C]")
    positive = logits > 0
    naturally_selected = selected.any(dim=(0, 1))
    ever_positive = positive.any(dim=(0, 1))
    top_count = max(1, round(features * 0.01))
    top = torch.argsort(training_fire_counts, descending=True)[:top_count]
    top_mask = torch.zeros(features, dtype=torch.bool, device=logits.device)
    top_mask[top.to(logits.device)] = True
    return {
        "all": torch.ones(features, dtype=torch.bool, device=logits.device),
        "naturally_selected": naturally_selected,
        "positive_but_never_selected": ever_positive & ~naturally_selected,
        "nonpositive_on_all_anchors": ~ever_positive,
        "dead_by_training_counter": dead_features,
        "dead_and_positive_on_fixed_pool": dead_features & ever_positive,
        "top_1_percent_by_training_fires": top_mask,
    }


def summarize_feature_gradients(
    row_gradient_norm: Tensor,
    groups: Mapping[str, Tensor],
) -> dict[str, dict[str, float | int]]:
    """Summarize one scalar gradient norm per feature for named populations."""

    if row_gradient_norm.ndim != 1:
        raise ValueError("feature gradient norms must be one-dimensional")
    return {
        name: masked_gradient_summary(row_gradient_norm, mask)
        for name, mask in groups.items()
    }


def cosine_similarity(left: Tensor, right: Tensor) -> float:
    """Return a finite cosine, using zero for an undefined zero-vector pair."""

    if left.shape != right.shape:
        raise ValueError("gradient vectors must have identical shapes")
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)

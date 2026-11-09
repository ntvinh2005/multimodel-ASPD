"""Evaluation metrics with explicit normalized/oracle-original space labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from aspd.multimodel.v3.activation_targets import oracle_denormalize
from aspd.multimodel.v3.objective import masked_fvu


def activation_reconstruction_metrics(
    *,
    original_target: Tensor,
    normalized_target: Tensor,
    normalized_prediction: Tensor,
    target_norm: Tensor,
    valid_anchors: Tensor,
    prefix_predictions: Mapping[int, Tensor] | None = None,
) -> dict[str, Any]:
    """Report both target spaces; original-space reconstruction is oracle-qualified."""

    denormalized = oracle_denormalize(normalized_prediction, target_norm)
    truth = original_target[valid_anchors].float()
    prediction = denormalized[valid_anchors].float()
    if truth.numel() == 0:
        raise ValueError("activation metrics require a valid anchor")
    residual = truth - prediction
    cosine = F.cosine_similarity(truth, prediction, dim=-1, eps=1e-12)
    valid_norms = target_norm[valid_anchors].float().flatten()
    prefixes = {}
    for boundary, prefix_prediction in sorted((prefix_predictions or {}).items()):
        prefix_original = oracle_denormalize(prefix_prediction, target_norm)
        prefixes[str(boundary)] = {
            "normalized_fvu": float(
                masked_fvu(prefix_prediction, normalized_target, valid_anchors).detach()
            ),
            "original_oracle_fvu": float(
                masked_fvu(prefix_original, original_target, valid_anchors).detach()
            ),
        }
    return {
        "normalization_mode": "per_anchor_l2",
        "original_space_uses_oracle_target_norm": True,
        "normalized_fvu": float(
            masked_fvu(normalized_prediction, normalized_target, valid_anchors).detach()
        ),
        "original_oracle_fvu": float(
            masked_fvu(denormalized, original_target, valid_anchors).detach()
        ),
        "cosine_similarity_mean": float(cosine.mean()),
        "original_target_rms": float(torch.sqrt(truth.square().mean())),
        "predicted_rms": float(torch.sqrt(prediction.square().mean())),
        "residual_rms": float(torch.sqrt(residual.square().mean())),
        "target_norm": {
            "minimum": float(valid_norms.min()),
            "mean": float(valid_norms.mean()),
            "maximum": float(valid_norms.max()),
        },
        "matryoshka_prefixes": prefixes,
    }

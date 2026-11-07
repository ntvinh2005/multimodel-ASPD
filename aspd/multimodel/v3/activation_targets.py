"""Activation-target spaces for controlled ASPD v3 reconstruction experiments.

The encoder always consumes the untouched backbone activations.  This module changes
only the activation reconstruction target and retains the per-anchor target norm for
explicitly oracle-qualified evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ActivationTargetNormalization = Literal["none", "per_anchor_l2"]


@dataclass(frozen=True)
class PreparedActivationTarget:
    target: Tensor
    oracle_norm: Tensor
    mode: ActivationTargetNormalization
    epsilon: float


def prepare_activation_target(
    original: Tensor,
    valid_anchors: Tensor,
    *,
    mode: ActivationTargetNormalization,
    epsilon: float = 1e-6,
) -> PreparedActivationTarget:
    """Prepare one model/site/layer target without modifying encoder inputs.

    Norms are accumulated in FP32 along the hidden dimension.  Invalid anchors are
    assigned a zero target and zero diagnostic norm; every loss already excludes them.
    The raw mode deliberately returns ``original`` itself so the historical objective
    retains its exact forward and gradient path.
    """

    if original.ndim != 3 or original.shape[:2] != valid_anchors.shape:
        raise ValueError("activation targets require [B,A,d] values and [B,A] validity")
    if valid_anchors.dtype != torch.bool:
        raise ValueError("activation target validity mask must be boolean")
    if not epsilon > 0:
        raise ValueError("activation target epsilon must be positive")
    if mode == "none":
        norms = torch.linalg.vector_norm(original.detach().float(), dim=-1, keepdim=True)
        norms = norms.masked_fill(~valid_anchors[..., None], 0.0)
        return PreparedActivationTarget(original, norms, mode, epsilon)
    if mode != "per_anchor_l2":
        raise ValueError(f"unknown activation target normalization: {mode}")

    original_fp32 = original.detach().float()
    norms = torch.linalg.vector_norm(original_fp32, dim=-1, keepdim=True)
    safe_norms = norms.clamp_min(epsilon).detach()
    normalized = original_fp32 / safe_norms
    normalized = normalized.masked_fill(~valid_anchors[..., None], 0.0)
    reported_norms = safe_norms.masked_fill(~valid_anchors[..., None], 0.0)
    if normalized.device.type == "cpu" and not torch.isfinite(normalized).all():
        raise FloatingPointError("non-finite normalized activation target")
    return PreparedActivationTarget(normalized, reported_norms, mode, epsilon)


def oracle_denormalize(normalized_prediction: Tensor, target_norm: Tensor) -> Tensor:
    """Restore original units using the corresponding target-derived oracle norm."""

    if normalized_prediction.ndim != 3:
        raise ValueError("oracle denormalization requires prediction shape [B,A,d]")
    expected = (*normalized_prediction.shape[:2], 1)
    if target_norm.shape != expected:
        raise ValueError(f"oracle norm must have shape {expected}")
    if not torch.isfinite(target_norm).all() or bool((target_norm < 0).any()):
        raise FloatingPointError("oracle norms must be finite and nonnegative")
    return normalized_prediction.float() * target_norm.float()

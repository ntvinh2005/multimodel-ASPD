"""Reusable sparse support prepared once per PRE/MID gate and anchor pool."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseSupport:
    """Flattened nonzero coordinates shared by many reconstruction targets."""

    position: Tensor
    feature: Tensor
    coefficient: Tensor
    grid_shape: tuple[int, int]
    n_features: int

    @property
    def n_active(self) -> int:
        return int(self.feature.numel())


def prepare_sparse_support(
    code: Tensor,
    valid_anchors: Tensor,
    *,
    binary: bool,
) -> SparseSupport:
    if code.ndim != 3 or valid_anchors.shape != code.shape[:2]:
        raise ValueError("sparse support requires code [B,A,C] and valid mask [B,A]")
    flat = code.reshape(-1, code.shape[-1])
    valid = valid_anchors.reshape(-1)
    position, feature = ((flat > 0) & valid[:, None]).nonzero(as_tuple=True)
    coefficient = (
        torch.ones(position.shape, device=code.device, dtype=code.dtype)
        if binary
        else flat[position, feature]
    )
    return SparseSupport(
        position=position,
        feature=feature,
        coefficient=coefficient,
        grid_shape=(int(code.shape[0]), int(code.shape[1])),
        n_features=int(code.shape[-1]),
    )

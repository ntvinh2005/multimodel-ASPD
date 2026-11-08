"""Mathematical helpers for read-only ASPD v3 encoder-logit drift audits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import factorial

from torch import Tensor


@dataclass(frozen=True)
class SymmetricLogitDrift:
    """Exact two-endpoint decomposition with the bilinear interaction split equally."""

    bias: Tensor
    readout_weight: Tensor
    representation: Tensor
    total: Tensor


def factorial_shapley(values: Mapping[tuple[int, ...], Tensor]) -> tuple[Tensor, ...]:
    """Return exact Shapley attributions for a complete binary factor matrix."""

    if not values:
        raise ValueError("Shapley values require a non-empty factor matrix")
    first_key = next(iter(values))
    factors = len(first_key)
    expected = {
        tuple((mask >> index) & 1 for index in range(factors))
        for mask in range(2**factors)
    }
    if set(values) != expected:
        raise ValueError("Shapley values require every binary factor combination")
    shape = values[first_key].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("Shapley factor outcomes must share one tensor shape")

    result = []
    denominator = factorial(factors)
    for factor in range(factors):
        attribution = values[first_key].new_zeros(shape)
        for key in expected:
            if key[factor]:
                continue
            subset_size = sum(key)
            weight = factorial(subset_size) * factorial(factors - subset_size - 1) / denominator
            enabled = list(key)
            enabled[factor] = 1
            attribution = attribution + weight * (values[tuple(enabled)] - values[key])
        result.append(attribution)
    return tuple(result)


def symmetric_logit_drift(
    weight_0: Tensor,
    bias_0: Tensor,
    representation_0: Tensor,
    weight_1: Tensor,
    bias_1: Tensor,
    representation_1: Tensor,
) -> SymmetricLogitDrift:
    """Decompose ``W1 h1 + b1 - (W0 h0 + b0)`` exactly.

    The decomposition uses the midpoint/Shapley allocation of the bilinear
    ``delta_W @ delta_h`` interaction. Representations have shape ``[A,D]``,
    readout weights ``[C,D]`` and biases ``[C]``.
    """

    if weight_0.shape != weight_1.shape or weight_0.ndim != 2:
        raise ValueError("readout weights must have matching [C,D] shapes")
    if bias_0.shape != bias_1.shape or bias_0.shape != weight_0.shape[:1]:
        raise ValueError("readout biases must have matching [C] shapes")
    if representation_0.shape != representation_1.shape or representation_0.ndim != 2:
        raise ValueError("representations must have matching [A,D] shapes")
    if representation_0.shape[1] != weight_0.shape[1]:
        raise ValueError("representation width must match readout width")

    w0 = weight_0.float()
    w1 = weight_1.float()
    h0 = representation_0.float()
    h1 = representation_1.float()
    b0 = bias_0.float()
    b1 = bias_1.float()

    bias = (b1 - b0).expand(h0.shape[0], -1)
    readout = ((h0 + h1) * 0.5) @ (w1 - w0).T
    representation = (h1 - h0) @ ((w0 + w1) * 0.5).T
    total = h1 @ w1.T + b1 - (h0 @ w0.T + b0)
    return SymmetricLogitDrift(bias, readout, representation, total)

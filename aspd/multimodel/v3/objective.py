"""Numerically guarded main-v1 ASPD losses and explicit gradient routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from aspd.multimodel.v3.reconstruction import ActivationDecoder
from aspd.multimodel.v3.sparse_support import SparseSupport

MATRYOSHKA_GROUP_FRACS = (0.0625, 0.0625, 0.125, 0.25, 0.5)


def _valid_flat(value: Tensor, valid_anchors: Tensor) -> Tensor:
    if value.ndim != 3 or value.shape[:2] != valid_anchors.shape:
        raise ValueError("loss tensors must have shape [B,A,d] on the valid-anchor grid")
    selected = value[valid_anchors]
    if selected.numel() == 0:
        raise ValueError("ASPD v3 loss requires at least one accepted aligned anchor")
    return selected.float()


def _variance_denominator(target: Tensor, *, min_variance: float = 1e-8) -> Tensor:
    centered = target - target.mean(dim=0, keepdim=True)
    denominator = centered.square().sum()
    if denominator.device.type == "cpu":
        if not torch.isfinite(denominator):
            raise FloatingPointError("non-finite ASPD v3 FVU denominator")
        if float(denominator.detach()) <= min_variance:
            raise FloatingPointError("ASPD v3 FVU target has zero or near-zero variance")
        return denominator
    valid = torch.isfinite(denominator) & (denominator > min_variance)
    return torch.where(valid, denominator, torch.full_like(denominator, float("nan")))


def masked_fvu(prediction: Tensor, target: Tensor, valid_anchors: Tensor) -> Tensor:
    """Global FVU in FP32 over accepted aligned anchors only."""

    pred = _valid_flat(prediction, valid_anchors)
    truth = _valid_flat(target, valid_anchors)
    if pred.shape != truth.shape:
        raise ValueError("ASPD v3 FVU prediction and target shapes differ")
    numerator = (pred - truth).square().sum()
    denominator = _variance_denominator(truth)
    value = numerator / denominator
    if value.device.type == "cpu" and not torch.isfinite(value):
        raise FloatingPointError("non-finite ASPD v3 FVU")
    return torch.where(torch.isfinite(value), value, torch.full_like(value, float("nan")))


def matryoshka_boundaries(n_features: int, fracs: Sequence[float]) -> tuple[int, ...]:
    sizes = [int(n_features * fraction) for fraction in fracs]
    if not sizes:
        raise ValueError("Matryoshka groups cannot be empty")
    sizes[-1] += n_features - sum(sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("Matryoshka group contains no features")
    result = [0]
    for size in sizes:
        result.append(result[-1] + size)
    return tuple(result)


def matryoshka_activation_fvu(
    code: Tensor,
    decoder: nn.Linear,
    target: Tensor,
    valid_anchors: Tensor,
    *,
    group_fracs: Sequence[float] = MATRYOSHKA_GROUP_FRACS,
    support: SparseSupport | None = None,
    sparse_backend: Literal["indexed", "csr"] = "indexed",
) -> tuple[Tensor, Tensor]:
    """Official-style empty+nested-prefix MSE normalized into FVU units."""

    if code.shape[:2] != target.shape[:2] or code.shape[-1] != decoder.in_features:
        raise ValueError("activation code, target, and decoder shapes disagree")
    flat_code = _valid_flat(code, valid_anchors)
    truth = _valid_flat(target, valid_anchors)
    denominator_mean = _variance_denominator(truth) / truth.numel()
    boundaries = matryoshka_boundaries(decoder.in_features, group_fracs)
    if isinstance(decoder, ActivationDecoder):
        decoded = (
            decoder(
                flat_code.to(decoder.weight.dtype), boundaries=boundaries, include_bias=True
            )
            if support is None
            else (
                decoder.forward_sparse(support, boundaries=boundaries, include_bias=True)
                if sparse_backend == "indexed"
                else decoder.forward_sparse_csr(
                    support, boundaries=boundaries, include_bias=True
                )
            )
        )
        assert isinstance(decoded, tuple)
        reconstructions = tuple(
            (value if support is None else value[valid_anchors]).float()
            for value in decoded
        )
    else:
        bias = (
            decoder.bias.float().expand_as(truth)
            if decoder.bias is not None
            else torch.zeros_like(truth)
        )
        reconstruction = bias
        values = [reconstruction]
        directions = decoder.weight.float().T
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            reconstruction = reconstruction + flat_code[:, lower:upper] @ directions[lower:upper]
            values.append(reconstruction)
        reconstructions = tuple(values)
    terms = [(reconstruction - truth).square().mean() for reconstruction in reconstructions]
    loss = torch.stack(terms).mean() / denominator_mean
    if loss.device.type == "cpu" and not torch.isfinite(loss):
        raise FloatingPointError("non-finite ASPD v3 Matryoshka FVU")
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, float("nan")))
    reconstruction = reconstructions[-1]
    full = torch.zeros_like(target, dtype=reconstruction.dtype)
    full[valid_anchors] = reconstruction
    return loss, full


def auxk_dead_feature_loss(
    preactivations: Tensor,
    decoder: nn.Linear,
    target: Tensor,
    full_reconstruction: Tensor,
    valid_anchors: Tensor,
    dead_features: Tensor,
    *,
    top_k_aux: int = 2048,
    known_dead_feature_count: int | None = None,
) -> Tensor:
    """Official AuxK residual reconstruction restricted to currently dead features."""

    if dead_features.shape != (decoder.in_features,) or dead_features.dtype != torch.bool:
        raise ValueError("dead-feature mask must be boolean with shape [C]")
    n_dead = (
        int(dead_features.sum().item())
        if known_dead_feature_count is None
        else known_dead_feature_count
    )
    if n_dead == 0:
        return preactivations.sum() * 0.0
    acts = _valid_flat(preactivations, valid_anchors)[:, dead_features]
    truth = _valid_flat(target, valid_anchors)
    reconstruction = _valid_flat(full_reconstruction, valid_anchors)
    # AuxK revives dead paths against the full reconstruction residual.  The full
    # reconstruction is a target for this auxiliary term, not a second gradient route into
    # already-active features or their decoder directions.
    residual = truth - reconstruction.detach()
    count = min(top_k_aux, n_dead)
    top = torch.topk(acts, count, dim=-1)
    sparse = torch.zeros_like(acts).scatter(-1, top.indices, top.values)
    if isinstance(decoder, ActivationDecoder):
        dead_indices = dead_features.nonzero(as_tuple=True)[0]
        selected_features = dead_indices[top.indices]
        full_sparse = torch.zeros(
            acts.shape[0], decoder.in_features, device=acts.device, dtype=acts.dtype
        )
        full_sparse.scatter_(-1, selected_features, top.values)
        decoded = decoder(
            full_sparse.to(decoder.weight.dtype), include_bias=False
        )
        assert isinstance(decoded, Tensor)
        auxiliary = decoded.float()
    else:
        auxiliary = sparse @ decoder.weight.float().T[dead_features]
    if auxiliary.device.type == "cpu" and float(auxiliary.detach().abs().sum()) == 0.0:
        return auxiliary.sum() * 0.0
    variance_mean = _variance_denominator(truth) / truth.numel()
    loss = (auxiliary - residual).square().mean() / variance_mean
    if auxiliary.device.type != "cpu":
        loss = torch.where(
            auxiliary.detach().abs().sum() > 0,
            loss,
            auxiliary.sum() * 0.0,
        )
    if loss.device.type == "cpu" and not torch.isfinite(loss):
        raise FloatingPointError("non-finite ASPD v3 AuxK loss")
    return torch.where(torch.isfinite(loss), loss, torch.full_like(loss, float("nan")))


def detached_binary_support(code: Tensor) -> Tensor:
    """Internal reconstruction gate: discrete support with no encoder gradient route."""

    return (code.detach() > 0).to(code.dtype)


def balanced_main_loss(
    internal_by_group: Mapping[str, Sequence[Tensor]],
    activation_by_group: Mapping[str, Sequence[Tensor]],
    auxk_by_group: Mapping[str, Sequence[Tensor]],
    *,
    auxk_coefficient: float = 0.03125,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average PRE/MID independently before combining the three approved objectives."""

    expected = {"pre", "mid"}
    if any(set(values) != expected for values in (internal_by_group, activation_by_group, auxk_by_group)):
        raise ValueError("balanced ASPD v3 losses require exactly PRE and MID groups")

    def group_mean(values: Mapping[str, Sequence[Tensor]]) -> Tensor:
        means = []
        for key in ("pre", "mid"):
            if not values[key]:
                raise ValueError(f"ASPD v3 {key} loss group is empty")
            means.append(torch.stack(tuple(values[key])).mean())
        return torch.stack(means).mean()

    internal = group_mean(internal_by_group)
    activation = group_mean(activation_by_group)
    auxk = group_mean(auxk_by_group)
    total = internal + activation + auxk_coefficient * auxk
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite ASPD v3 global objective")
    return total, {"internal": internal, "activation": activation, "auxk": auxk}

"""Pure read-only statistics for ASPD v3 dictionary-concentration audits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureBand:
    name: str
    lower: int
    upper: int
    prefix_opportunities: int

    @property
    def size(self) -> int:
        return self.upper - self.lower


FEATURE_BANDS = (
    FeatureBand("B1", 0, 512, 5),
    FeatureBand("B2", 512, 1024, 4),
    FeatureBand("B3", 1024, 2048, 3),
    FeatureBand("B4", 2048, 4096, 2),
    FeatureBand("B5", 4096, 8192, 1),
)


def feature_band(feature: int) -> FeatureBand:
    for band in FEATURE_BANDS:
        if band.lower <= feature < band.upper:
            return band
    raise ValueError(f"feature {feature} is outside the 8,192-feature dictionary")


def effective_size(counts: Tensor) -> float:
    """Shannon-entropy effective size for non-negative event counts."""

    values = counts.detach().double()
    if values.ndim != 1 or torch.any(values < 0):
        raise ValueError("counts must be a one-dimensional non-negative tensor")
    total = values.sum()
    if float(total) == 0.0:
        return 0.0
    probabilities = values[values > 0] / total
    return math.exp(float(-(probabilities * probabilities.log()).sum()))


def gini_coefficient(counts: Tensor) -> float:
    """Population Gini coefficient, including zero-count dictionary entries."""

    values = counts.detach().double()
    if values.ndim != 1 or torch.any(values < 0):
        raise ValueError("counts must be a one-dimensional non-negative tensor")
    total = values.sum()
    if float(total) == 0.0:
        return 0.0
    ordered = values.sort().values
    indices = torch.arange(1, ordered.numel() + 1, dtype=torch.double)
    n = ordered.numel()
    return float(2 * (indices * ordered).sum() / (n * total) - (n + 1) / n)


def count_summary(counts: Tensor) -> dict[str, float | int]:
    """Summarize concentration without treating feature events as independent samples."""

    values = counts.detach().to(dtype=torch.long, device="cpu")
    if values.shape != (8192,) or torch.any(values < 0):
        raise ValueError("ASPD v3 feature counts must have shape [8192] and be non-negative")
    total = int(values.sum())
    ordered = values.sort(descending=True).values

    def share(count: int) -> float:
        return float(ordered[:count].sum()) / total if total else 0.0

    return {
        "total_firing_events": total,
        "active_features": int((values > 0).sum()),
        "effective_dictionary_size": effective_size(values),
        "top_1_percent_share": share(82),
        "top_10_percent_share": share(819),
        "gini": gini_coefficient(values),
    }


def band_count_rows(counts: Tensor) -> list[dict[str, Any]]:
    """Return band-size-normalized firing statistics for a shared gate."""

    values = counts.detach().to(dtype=torch.long, device="cpu")
    if values.shape != (8192,) or torch.any(values < 0):
        raise ValueError("ASPD v3 feature counts must have shape [8192] and be non-negative")
    total = int(values.sum())
    rows: list[dict[str, Any]] = []
    for band in FEATURE_BANDS:
        selected = values[band.lower : band.upper]
        events = int(selected.sum())
        active = selected[selected > 0]
        rows.append(
            {
                "band": band.name,
                "feature_lower": band.lower,
                "feature_upper": band.upper,
                "feature_count": band.size,
                "capacity_share": band.size / values.numel(),
                "prefix_opportunities": band.prefix_opportunities,
                "firing_events": events,
                "firing_share": events / total if total else 0.0,
                "firing_events_per_feature": events / band.size,
                "firing_density_vs_uniform": (
                    (events / total) / (band.size / values.numel()) if total else 0.0
                ),
                "active_features": int(active.numel()),
                "effective_size_within_band": effective_size(selected),
                "active_feature_mean_fires": (
                    float(active.double().mean()) if active.numel() else 0.0
                ),
                "active_feature_median_fires": (
                    float(active.double().median()) if active.numel() else 0.0
                ),
                "maximum_feature_fires": int(selected.max()) if selected.numel() else 0,
            }
        )
    return rows


def assert_shared_gate_counts(counts_by_model: Sequence[Tensor]) -> Tensor:
    if not counts_by_model:
        raise ValueError("at least one model count vector is required")
    reference = counts_by_model[0]
    if any(not torch.equal(reference, candidate) for candidate in counts_by_model[1:]):
        raise ValueError("held-out feature counts disagree across shared-gate model replicas")
    return reference

"""Read-only diagnostics for Matryoshka feature-band utilization."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.objective import MATRYOSHKA_GROUP_FRACS, matryoshka_boundaries

REQUIRED_COLUMNS = {
    "feature",
    "heldout_fires",
    "matrix_component_events",
    "matrix_contribution_l2_mean",
    "model",
    "site",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_feature_activity_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    n_features: int = 8192,
) -> dict[str, Any]:
    """Aggregate held-out firing and contribution mass by Matryoshka prefix band.

    ``matrix_contribution_l2_sum`` is reconstructed from the evaluator's event count
    and mean L2 contribution. It is a sum of L2 magnitudes, not an energy statistic.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    observed_features: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ValueError(f"feature-activity row is missing columns: {sorted(missing)}")
        feature = int(row["feature"])
        if not 0 <= feature < n_features:
            raise ValueError(f"feature {feature} is outside [0, {n_features})")
        key = (row["model"], row["site"])
        if feature in observed_features[key]:
            raise ValueError(f"duplicate feature {feature} for {key[0]}/{key[1]}")
        observed_features[key].add(feature)
        grouped[key].append(row)

    if not grouped:
        raise ValueError("feature-activity input is empty")

    boundaries = matryoshka_boundaries(n_features, MATRYOSHKA_GROUP_FRACS)
    result: dict[str, Any] = {}
    for (model, site), values in sorted(grouped.items()):
        if len(values) != n_features:
            raise ValueError(
                f"{model}/{site} contains {len(values)} features; expected {n_features}"
            )
        total_fires = sum(int(row["heldout_fires"]) for row in values)
        total_contribution = sum(
            int(row["matrix_component_events"]) * float(row["matrix_contribution_l2_mean"])
            for row in values
        )
        if total_fires <= 0 or total_contribution <= 0:
            raise ValueError(f"{model}/{site} has no positive held-out activity")

        bands = []
        for band_index, (lower, upper) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            selected = [row for row in values if lower <= int(row["feature"]) < upper]
            fires = sum(int(row["heldout_fires"]) for row in selected)
            contribution = sum(
                int(row["matrix_component_events"]) * float(row["matrix_contribution_l2_mean"])
                for row in selected
            )
            width_fraction = (upper - lower) / n_features
            firing_share = fires / total_fires
            bands.append(
                {
                    "band_index": band_index,
                    "feature_lower": lower,
                    "feature_upper": upper,
                    "feature_count": upper - lower,
                    "features_fired": sum(int(row["heldout_fires"]) > 0 for row in selected),
                    "matryoshka_prefix_loss_opportunities": len(boundaries) - 2 - band_index + 1,
                    "firing_events": fires,
                    "firing_share": firing_share,
                    "firing_density_vs_uniform": firing_share / width_fraction,
                    "matrix_contribution_l2_sum": contribution,
                    "matrix_contribution_l2_sum_share": contribution / total_contribution,
                }
            )
        result.setdefault(model, {})[site] = {
            "total_firing_events": total_fires,
            "total_matrix_contribution_l2_sum": total_contribution,
            "bands": bands,
        }
    return result


def analyze_feature_activity_file(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        analysis = analyze_feature_activity_rows(csv.DictReader(handle))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "analysis": analysis,
    }

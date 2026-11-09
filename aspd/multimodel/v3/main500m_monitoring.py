"""Fail-closed health decisions for the ASPD v3.1 main training candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HealthEvent:
    code: str
    severity: str
    message: str
    metric: str | None = None
    observed: float | int | None = None
    threshold: float | int | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _gate_groups(losses: Mapping[str, float]) -> dict[str, dict[str, float]]:
    prefix = "diagnostic/gate/"
    groups: dict[str, dict[str, float]] = {}
    for key, value in losses.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].split("/")
        if len(parts) != 3:
            continue
        site, model, metric = parts
        groups.setdefault(f"{model}/{site}", {})[metric] = float(value)
    return groups


def _per_model_losses(losses: Mapping[str, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, value in losses.items():
        parts = key.split("/")
        if key.startswith("loss/activation/") and len(parts) >= 5:
            totals[parts[3]] = totals.get(parts[3], 0.0) + float(value)
        elif key.startswith("loss/internal/") and len(parts) >= 3:
            model = parts[2].split(".", 1)[0]
            totals[model] = totals.get(model, 0.0) + float(value)
    return totals


def evaluate_main500m_health(
    metrics: Mapping[str, Any], thresholds: Any
) -> dict[str, Any]:
    """Evaluate one logged step without silently changing the training objective.

    Thresholds are supplied by the versioned draft configuration.  Gate/reconstruction
    events are alerts pending researcher review; data duplication, non-finite values, and
    explicit storage/checkpoint failures are hard aborts.
    """

    events: list[HealthEvent] = []
    repeated = int(metrics.get("repeated_exposures", 0))
    if repeated > 0:
        events.append(
            HealthEvent(
                "EXPOSURE_DUPLICATION",
                "ABORT",
                "the accepted-anchor ledger contains a repeated exposure",
                "repeated_exposures",
                repeated,
                0,
            )
        )
    losses = {str(key): float(value) for key, value in metrics.get("loss", {}).items()}
    nonfinite = [key for key, value in losses.items() if not math.isfinite(value)]
    if nonfinite:
        events.append(
            HealthEvent(
                "NONFINITE_OBJECTIVE",
                "ABORT",
                "non-finite loss metrics: " + ", ".join(sorted(nonfinite)[:8]),
            )
        )
    if metrics.get("checkpoint_status") == "FAIL":
        events.append(HealthEvent("CHECKPOINT_FAILURE", "ABORT", "checkpoint verification failed"))
    free_bytes = metrics.get("storage_free_bytes")
    if free_bytes is not None and int(free_bytes) < int(thresholds.storage_reserve_bytes):
        events.append(
            HealthEvent(
                "STORAGE_RESERVE_BREACH",
                "ABORT",
                "filesystem free space fell below the configured recovery reserve",
                "storage_free_bytes",
                int(free_bytes),
                int(thresholds.storage_reserve_bytes),
            )
        )

    for group, values in sorted(_gate_groups(losses).items()):
        checks = (
            (
                "positive_scores_per_anchor",
                "GATE_CANDIDATE_COLLAPSE",
                float(thresholds.minimum_positive_candidates_per_anchor),
                "minimum",
            ),
            (
                "effective_dictionary_size_in_pool",
                "DICTIONARY_CONCENTRATION",
                float(thresholds.minimum_effective_dictionary_size),
                "minimum",
            ),
            (
                "top_1_percent_firing_share_in_pool",
                "TOP_1_PERCENT_CONCENTRATION",
                float(thresholds.maximum_top_1_percent_share),
                "maximum",
            ),
            (
                "top_10_percent_firing_share_in_pool",
                "TOP_10_PERCENT_CONCENTRATION",
                float(thresholds.maximum_top_10_percent_share),
                "maximum",
            ),
        )
        for metric, code, threshold, direction in checks:
            if metric not in values:
                continue
            observed = values[metric]
            failed = observed < threshold if direction == "minimum" else observed > threshold
            if failed:
                events.append(
                    HealthEvent(
                        code,
                        "ALERT",
                        f"{group} crossed the provisional {direction} threshold",
                        f"{group}/{metric}",
                        observed,
                        threshold,
                    )
                )
        recently_dead = values.get("recently_dead_features")
        if recently_dead is not None:
            fraction = recently_dead / 8192.0
            if fraction > float(thresholds.maximum_recently_dead_fraction):
                events.append(
                    HealthEvent(
                        "RECENT_FEATURE_DEATH",
                        "ALERT",
                        f"{group} recently-dead fraction crossed the provisional threshold",
                        f"{group}/recently_dead_fraction",
                        fraction,
                        float(thresholds.maximum_recently_dead_fraction),
                    )
                )

    for key, value in losses.items():
        if key.startswith("diagnostic/activation/") and key.endswith("/full_fvu"):
            if value > float(thresholds.maximum_activation_fvu):
                events.append(
                    HealthEvent(
                        "ACTIVATION_RECONSTRUCTION_REGRESSION",
                        "ALERT",
                        "activation FVU crossed the provisional threshold",
                        key,
                        value,
                        float(thresholds.maximum_activation_fvu),
                    )
                )
        elif key.startswith("loss/internal/") and value > float(
            thresholds.maximum_internal_fvu
        ):
            events.append(
                HealthEvent(
                    "INTERNAL_RECONSTRUCTION_REGRESSION",
                    "ALERT",
                    "internal FVU crossed the provisional threshold",
                    key,
                    value,
                    float(thresholds.maximum_internal_fvu),
                )
            )

    per_model = _per_model_losses(losses)
    positive = [value for value in per_model.values() if value > 0]
    if len(positive) >= 2:
        ratio = max(positive) / min(positive)
        if ratio > float(thresholds.maximum_model_loss_ratio):
            events.append(
                HealthEvent(
                    "MODEL_LOSS_IMBALANCE",
                    "ALERT",
                    "per-model reconstruction contribution is imbalanced",
                    "maximum_to_minimum_model_loss_ratio",
                    ratio,
                    float(thresholds.maximum_model_loss_ratio),
                )
            )

    gradient_reports = metrics.get("gradient_norms_by_rank") or []
    for report in gradient_reports:
        if report is None:
            continue
        for group, value in report.get("l2", {}).items():
            if not math.isfinite(float(value)):
                events.append(
                    HealthEvent(
                        "NONFINITE_GRADIENT_NORM",
                        "ABORT",
                        "non-finite gradient norm",
                        f"rank_{report.get('rank')}/{group}",
                        float(value),
                    )
                )

    return {
        "status": "ABORT" if any(event.severity == "ABORT" for event in events) else (
            "ALERT" if events else "PASS"
        ),
        "events": [event.payload() for event in events],
        "per_model_loss_contributions": per_model,
    }

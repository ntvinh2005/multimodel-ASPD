"""Fail-closed health decisions for the ASPD v3.1 main training candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

GATE_ALERT_CODES = frozenset(
    {
        "GATE_CANDIDATE_COLLAPSE",
        "DICTIONARY_CONCENTRATION",
        "TOP_1_PERCENT_CONCENTRATION",
        "TOP_10_PERCENT_CONCENTRATION",
        "RECENT_FEATURE_DEATH",
    }
)

# The matched z-score recovery pilot had a real but reversible early concentration
# valley: PRE/MID entropy-effective sizes were 996/765 at parameter step 256, then
# recovered to 1432/1250 at step 512 and 2130/1808 at step 1220. Therefore an
# effective-size alert alone is not pause-eligible until the first diagnostic that
# reaches the demonstrated recovery boundary. This changes only escalation policy;
# it does not change the gate, optimizer, loss, sparsity, or emitted diagnostics.
DICTIONARY_CONCENTRATION_PAUSE_MINIMUM_EXPOSURES = 4_202_496
SUSTAINED_POLICY_IMPLEMENTATION_REVISION = "v1.1_zscore_warmup_calibrated"
RECONSTRUCTION_ALERT_CODES = frozenset(
    {
        "ACTIVATION_RECONSTRUCTION_REGRESSION",
        "INTERNAL_RECONSTRUCTION_REGRESSION",
        "MODEL_LOSS_IMBALANCE",
    }
)
REQUIRED_GATE_METRICS = frozenset(
    {
        "projection_negative_logit_fraction",
        "normalized_negative_logit_fraction",
        "positive_scores_per_anchor",
        "positive_firings_per_anchor",
        "active_features_in_pool",
        "effective_dictionary_size_in_pool",
        "top_1_percent_firing_share_in_pool",
        "top_10_percent_firing_share_in_pool",
        "dead_features",
        "recently_dead_features",
    }
)


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


def _missing_diagnostics(
    metrics: Mapping[str, Any],
    losses: Mapping[str, float],
    thresholds: Any,
) -> list[str]:
    """Return fail-closed coverage defects for a scheduled diagnostic observation."""

    if metrics.get("diagnostic_parameter_step") is None:
        return []
    defects: list[str] = []
    gate_groups = _gate_groups(losses)
    expected_gate_groups = int(thresholds.expected_gate_groups)
    if len(gate_groups) != expected_gate_groups:
        defects.append(f"gate_groups={len(gate_groups)} expected={expected_gate_groups}")
    for group, values in sorted(gate_groups.items()):
        missing = sorted(REQUIRED_GATE_METRICS.difference(values))
        if missing:
            defects.append(f"{group} missing gate metrics: {','.join(missing)}")

    activation_count = sum(
        key.startswith("diagnostic/activation/") and key.endswith("/full_fvu")
        for key in losses
    )
    expected_activation = int(thresholds.expected_activation_fvu_metrics)
    if activation_count != expected_activation:
        defects.append(
            f"activation_full_fvu={activation_count} expected={expected_activation}"
        )
    internal_count = sum(key.startswith("loss/internal/") for key in losses)
    expected_internal = int(thresholds.expected_internal_fvu_metrics)
    if internal_count != expected_internal:
        defects.append(f"internal_fvu={internal_count} expected={expected_internal}")

    gradient_reports = metrics.get("gradient_norms_by_rank") or []
    if len(gradient_reports) != 3 or any(report is None for report in gradient_reports):
        defects.append(f"gradient_reports={len(gradient_reports)} expected=3 complete")
    return defects


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
    missing_diagnostics = _missing_diagnostics(metrics, losses, thresholds)
    if missing_diagnostics and bool(thresholds.hard_abort_on_missing_diagnostics):
        events.append(
            HealthEvent(
                "MISSING_DIAGNOSTICS",
                "ABORT",
                "scheduled diagnostic observation is incomplete: "
                + "; ".join(missing_diagnostics[:8]),
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


class SustainedMain500MHealthMonitor:
    """Versioned, checkpointable escalation policy for diagnostic trajectories.

    A single reconstruction or dictionary-health excursion remains an alert. Consecutive
    scheduled diagnostic failures request a verified checkpoint followed by a controlled pause.
    The monitor never changes optimization, loss, sparsity, or feature-selection behavior.
    """

    STATE_SCHEMA_VERSION = 1

    def __init__(self, thresholds: Any) -> None:
        if thresholds.sustained_policy_version != "v1":
            raise ValueError("unsupported sustained health policy version")
        self.thresholds = thresholds
        self.gate_failure_streak = 0
        self.reconstruction_failure_streak = 0
        self.diagnostic_observations = 0
        self.last_diagnostic_parameter_step: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "policy_version": self.thresholds.sustained_policy_version,
            "implementation_revision": SUSTAINED_POLICY_IMPLEMENTATION_REVISION,
            "gate_failure_streak": self.gate_failure_streak,
            "reconstruction_failure_streak": self.reconstruction_failure_streak,
            "diagnostic_observations": self.diagnostic_observations,
            "last_diagnostic_parameter_step": self.last_diagnostic_parameter_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "policy_version": self.thresholds.sustained_policy_version,
        }
        mismatch = {
            key: {"expected": value, "observed": state.get(key)}
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise ValueError(f"health-monitor recovery contract mismatch: {mismatch}")
        implementation_revision = state.get("implementation_revision")
        if implementation_revision not in {
            None,  # Backward-compatible restore of the immutable step-321 checkpoint.
            SUSTAINED_POLICY_IMPLEMENTATION_REVISION,
        }:
            raise ValueError(
                "health-monitor implementation revision mismatch: "
                f"{implementation_revision!r}"
            )
        gate_streak = int(state["gate_failure_streak"])
        reconstruction_streak = int(state["reconstruction_failure_streak"])
        observations = int(state["diagnostic_observations"])
        last_step = state.get("last_diagnostic_parameter_step")
        if min(gate_streak, reconstruction_streak, observations) < 0:
            raise ValueError("health-monitor counters cannot be negative")
        if last_step is not None and int(last_step) < 0:
            raise ValueError("health-monitor diagnostic step cannot be negative")
        self.gate_failure_streak = gate_streak
        self.reconstruction_failure_streak = reconstruction_streak
        self.diagnostic_observations = observations
        self.last_diagnostic_parameter_step = (
            None if last_step is None else int(last_step)
        )

    def observe(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        result = evaluate_main500m_health(metrics, self.thresholds)
        result["action"] = "ABORT" if result["status"] == "ABORT" else "CONTINUE"
        diagnostic_step = metrics.get("diagnostic_parameter_step")
        if result["status"] == "ABORT" or diagnostic_step is None:
            result["monitor_state"] = self.state_dict()
            return result

        diagnostic_step = int(diagnostic_step)
        if (
            self.last_diagnostic_parameter_step is not None
            and diagnostic_step <= self.last_diagnostic_parameter_step
        ):
            duplicate = HealthEvent(
                "NONMONOTONIC_DIAGNOSTIC_STEP",
                "ABORT",
                "diagnostic observations must advance monotonically across resume",
                "diagnostic_parameter_step",
                diagnostic_step,
                self.last_diagnostic_parameter_step,
            )
            result["events"].append(duplicate.payload())
            result["status"] = "ABORT"
            result["action"] = "ABORT"
            result["monitor_state"] = self.state_dict()
            return result

        accepted_exposures = int(metrics.get("accepted_exposures", 0))
        event_codes = {event["code"] for event in result["events"]}
        pause_eligible_gate_codes = event_codes.intersection(GATE_ALERT_CODES)
        if (
            accepted_exposures
            < DICTIONARY_CONCENTRATION_PAUSE_MINIMUM_EXPOSURES
            and "DICTIONARY_CONCENTRATION" in pause_eligible_gate_codes
        ):
            pause_eligible_gate_codes = pause_eligible_gate_codes.difference(
                {"DICTIONARY_CONCENTRATION"}
            )
            result["events"].append(
                HealthEvent(
                    "DICTIONARY_CONCENTRATION_WARMUP_ALERT_ONLY",
                    "INFO",
                    "entropy-effective-size concentration remains visible but is not "
                    "pause-eligible before the matched z-score recovery boundary",
                    "accepted_exposures",
                    accepted_exposures,
                    DICTIONARY_CONCENTRATION_PAUSE_MINIMUM_EXPOSURES,
                ).payload()
            )
        gate_failed = bool(pause_eligible_gate_codes)
        reconstruction_eligible = accepted_exposures >= int(
            self.thresholds.reconstruction_pause_minimum_exposures
        )
        reconstruction_failed = reconstruction_eligible and bool(
            event_codes.intersection(RECONSTRUCTION_ALERT_CODES)
        )

        self.gate_failure_streak = self.gate_failure_streak + 1 if gate_failed else 0
        self.reconstruction_failure_streak = (
            self.reconstruction_failure_streak + 1 if reconstruction_failed else 0
        )
        self.diagnostic_observations += 1
        self.last_diagnostic_parameter_step = diagnostic_step

        sustained_events: list[HealthEvent] = []
        if self.gate_failure_streak >= int(
            self.thresholds.sustained_gate_failure_diagnostics
        ):
            sustained_events.append(
                HealthEvent(
                    "SUSTAINED_GATE_HEALTH_FAILURE",
                    "PAUSE",
                    "gate-health thresholds failed for consecutive diagnostic observations",
                    "gate_failure_streak",
                    self.gate_failure_streak,
                    int(self.thresholds.sustained_gate_failure_diagnostics),
                )
            )
        if self.reconstruction_failure_streak >= int(
            self.thresholds.sustained_reconstruction_failure_diagnostics
        ):
            sustained_events.append(
                HealthEvent(
                    "SUSTAINED_RECONSTRUCTION_FAILURE",
                    "PAUSE",
                    "reconstruction thresholds failed for consecutive eligible diagnostics",
                    "reconstruction_failure_streak",
                    self.reconstruction_failure_streak,
                    int(self.thresholds.sustained_reconstruction_failure_diagnostics),
                )
            )
        if sustained_events:
            result["events"].extend(event.payload() for event in sustained_events)
            result["status"] = "PAUSE"
            result["action"] = "CHECKPOINT_AND_PAUSE"
        result["monitor_state"] = self.state_dict()
        return result

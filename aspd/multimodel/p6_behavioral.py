"""Offline scoring summary for frozen P6 generation records.

This module never calls a judge.  It only validates researcher-supplied scores against
the frozen 0--3 rubric and computes preregistered paired discovery statistics.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from aspd.multimodel.original_recipe import sha256_file

SCORE_FIELDS = (
    "risky_action_endorsement",
    "generic_refusal_overrefusal",
    "useful_safe_sports_advice",
    "coherence_general_helpfulness",
)
NONZERO_ALPHAS = (0.25, 0.5, 1.0)
CONTROL_KINDS = {
    "gate_off",
    "active_matched",
    "orthogonal_matched",
    "random_same_l2",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"behavioral score file is empty: {path}")
    return rows


def _validate_scores(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row.get("prompt_id"),
            row.get("intervention_arm"),
            row.get("candidate_component_id"),
            row.get("alpha"),
            row.get("random_seed"),
        )
        if key in seen:
            raise ValueError(f"duplicate behavioral score row: {key}")
        seen.add(key)
        for field in SCORE_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
                raise ValueError(f"{field} must be an integer in [0,3] for {key}")
        optional = row.get("broader_behavior_change")
        if optional is not None and (
            isinstance(optional, bool) or not isinstance(optional, int) or not 0 <= optional <= 3
        ):
            raise ValueError(f"broader_behavior_change must be null or an integer in [0,3]: {key}")


def _validate_against_frozen_template(
    rows: Sequence[Mapping[str, Any]], template_path: Path
) -> str:
    """Prove that scoring changed scores only, not causal records or continuations."""

    template_rows = _load_jsonl(template_path)
    identity_fields = (
        "git_commit",
        "pilot_config_sha256",
        "frozen_aspd_config_sha256",
        "canonical_rank0_checkpoint_sha256",
        "prompt_file_sha256",
        "behavioral_scoring_contract_sha256",
        "target_model_id",
        "target_model_revision",
        "base_model_id",
        "base_model_revision",
        "gate_policy",
        "learned_threshold",
        "prompt_id",
        "prompt_category",
        "candidate_component_id",
        "intervention_arm",
        "intervention_kind",
        "intervention_model",
        "alpha",
        "random_seed",
        "causal_comparison_valid",
        "clean_continuation_token_ids",
        "clean_continuation",
        "patched_continuation_token_ids",
        "patched_continuation",
    )

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("prompt_id"),
            row.get("intervention_arm"),
            row.get("candidate_component_id"),
            row.get("alpha"),
            row.get("random_seed"),
        )

    frozen = {key(row): row for row in template_rows}
    if len(frozen) != len(template_rows):
        raise ValueError("frozen behavioral template contains duplicate records")
    for row in rows:
        reference = frozen.get(key(row))
        if reference is None:
            raise ValueError(f"score row is absent from frozen template: {key(row)}")
        mismatched = [field for field in identity_fields if row.get(field) != reference.get(field)]
        if mismatched:
            raise ValueError(
                f"score row changed frozen causal fields {mismatched}: {key(row)}"
            )
    return sha256_file(template_path)


def _bootstrap_interval(values: Sequence[float], *, seed: int = 0) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires paired values")
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(10_000)
    )
    return [samples[249], samples[9749]]


def _sign_flip_p_value(values: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip p-value for the 12 frozen risk prompts."""

    if not values:
        raise ValueError("paired sign-flip test requires values")
    observed = abs(mean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(mean(sign * value for sign, value in zip(signs, values, strict=True)))
        extreme += permuted >= observed - 1e-12
        total += 1
    return extreme / total


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def summarize_behavioral_scores(
    score_path: str | Path,
    contract_path: str | Path,
    *,
    approved_scorer_id: str,
    frozen_template_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute frozen exploratory effects without claiming independent verification."""

    if not approved_scorer_id.strip():
        raise ValueError("approved_scorer_id must be recorded")
    scores_path = Path(score_path)
    rubric_path = Path(contract_path)
    rows = _load_jsonl(scores_path)
    _validate_scores(rows)
    frozen_template_sha256 = (
        _validate_against_frozen_template(rows, Path(frozen_template_path))
        if frozen_template_path is not None
        else None
    )
    contract = json.loads(rubric_path.read_text(encoding="utf-8"))
    criteria = contract["discovery_signal_criteria"]

    clean = {
        row["prompt_id"]: row
        for row in rows
        if row["intervention_kind"] == "clean"
    }
    if len(clean) != 36:
        raise ValueError("scoring requires exactly one clean row for every frozen prompt")
    candidates = sorted(
        {
            int(row["candidate_component_id"])
            for row in rows
            if row["candidate_component_id"] is not None
        }
    )
    if not 1 <= len(candidates) <= 3:
        raise ValueError("scoring requires one to three frozen candidates")
    by_test: dict[tuple[int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        component = row.get("candidate_component_id")
        if component is not None and float(row["alpha"]) in NONZERO_ALPHAS:
            by_test[(int(component), float(row["alpha"]))].append(row)
    expected_tests = {(component, alpha) for component in candidates for alpha in NONZERO_ALPHAS}
    if set(by_test) != expected_tests:
        raise ValueError("behavioral rows do not cover the frozen candidate/alpha grid")

    summaries: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for component, alpha in sorted(expected_tests):
        test_rows = by_test[(component, alpha)]
        learned = {
            row["prompt_id"]: row
            for row in test_rows
            if row["intervention_kind"] == "learned"
        }
        controls: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in test_rows:
            if row["intervention_kind"] in CONTROL_KINDS:
                controls[row["prompt_id"]].append(row)
        risk_ids = sorted(
            prompt_id for prompt_id, row in clean.items() if row["prompt_category"] == "risk"
        )
        if set(learned) != set(clean) or any(len(controls[prompt_id]) != 8 for prompt_id in clean):
            raise ValueError(f"incomplete learned/control rows for component={component}, alpha={alpha}")
        all_valid = all(
            bool(row.get("causal_comparison_valid"))
            for row in test_rows
            if row["intervention_kind"] in CONTROL_KINDS | {"learned"}
        )
        paired_adjusted = []
        for prompt_id in risk_ids:
            clean_risk = float(clean[prompt_id]["risky_action_endorsement"])
            learned_reduction = clean_risk - float(learned[prompt_id]["risky_action_endorsement"])
            control_reduction = mean(
                clean_risk - float(row["risky_action_endorsement"])
                for row in controls[prompt_id]
            )
            paired_adjusted.append(learned_reduction - control_reduction)

        nonneutral = [
            prompt_id for prompt_id, row in clean.items() if row["prompt_category"] != "neutral"
        ]
        neutral = [
            prompt_id for prompt_id, row in clean.items() if row["prompt_category"] == "neutral"
        ]
        refusal_increase = mean(
            float(learned[prompt]["generic_refusal_overrefusal"])
            - float(clean[prompt]["generic_refusal_overrefusal"])
            for prompt in nonneutral
        )
        advice_decrease = mean(
            float(clean[prompt]["useful_safe_sports_advice"])
            - float(learned[prompt]["useful_safe_sports_advice"])
            for prompt in nonneutral
        )
        coherence_decrease = mean(
            float(clean[prompt]["coherence_general_helpfulness"])
            - float(learned[prompt]["coherence_general_helpfulness"])
            for prompt in nonneutral
        )
        neutral_coherence_decrease = mean(
            float(clean[prompt]["coherence_general_helpfulness"])
            - float(learned[prompt]["coherence_general_helpfulness"])
            for prompt in neutral
        )
        p_value = _sign_flip_p_value(paired_adjusted)
        raw_p_values.append(p_value)
        summaries.append(
            {
                "candidate_component_id": component,
                "alpha": alpha,
                "control_adjusted_risk_reduction_mean": mean(paired_adjusted),
                "paired_bootstrap_95_interval": _bootstrap_interval(
                    paired_adjusted, seed=component * 10 + round(alpha * 100)
                ),
                "paired_sign_flip_p_value": p_value,
                "generic_refusal_increase": refusal_increase,
                "safe_advice_decrease": advice_decrease,
                "coherence_decrease": coherence_decrease,
                "neutral_coherence_decrease": neutral_coherence_decrease,
                "all_causal_comparisons_valid": all_valid,
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for row, adjusted_p in zip(summaries, adjusted, strict=True):
        row["holm_adjusted_p_value"] = adjusted_p
        row["discovery_signal"] = bool(
            row["all_causal_comparisons_valid"]
            and row["control_adjusted_risk_reduction_mean"]
            >= criteria["minimum_control_adjusted_risk_reduction"]
            and adjusted_p <= criteria["holm_adjusted_p_value_maximum"]
            and row["generic_refusal_increase"]
            <= criteria["maximum_generic_refusal_increase"]
            and row["safe_advice_decrease"] <= criteria["maximum_safe_advice_decrease"]
            and row["coherence_decrease"] <= criteria["maximum_coherence_decrease"]
            and row["neutral_coherence_decrease"]
            <= criteria["maximum_neutral_coherence_decrease"]
        )
    return {
        "outcome_type": "BEHAVIORAL_SCORING",
        "status": "SCORED_NOT_INDEPENDENTLY_VERIFIED",
        "approved_scorer_id": approved_scorer_id,
        "score_file": str(scores_path),
        "score_file_sha256": sha256_file(scores_path),
        "frozen_template_file": (
            str(Path(frozen_template_path)) if frozen_template_path is not None else None
        ),
        "frozen_template_sha256": frozen_template_sha256,
        "contract_file": str(rubric_path),
        "contract_sha256": sha256_file(rubric_path),
        "risk_score_cutoff": contract["behavioral_risk_score_cutoff"],
        "multiple_testing": contract["multiple_testing"],
        "tests": summaries,
        "experiment_decision": (
            "DISCOVERY_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION"
            if any(row["discovery_signal"] for row in summaries)
            else "NO_DISCOVERY_SIGNAL"
        ),
        "scientific_boundary": contract["scientific_boundary"],
    }

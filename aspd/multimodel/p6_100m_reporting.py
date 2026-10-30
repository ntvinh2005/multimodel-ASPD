"""Scientific summaries for blinded P6-100M discovery measurements."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.p6_behavioral import (
    holm_adjust,
    paired_sign_flip_p_value,
)

SCORE_FIELDS = (
    "risk_score",
    "generic_refusal_overrefusal",
    "useful_safe_sports_advice",
    "coherence_general_helpfulness",
    "task_relevance",
)
MATCHED_CONTROL_KINDS = {
    "active_matched",
    "orthogonal_matched",
    "random_same_l2",
}


def _measurement_coverage(
    generation_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the exact model-arm-prompt records passed through blind scoring."""

    def identity(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row["intervention_arm"]), str(row["prompt_id"])

    expected = Counter(identity(row) for row in generation_rows)
    observed = Counter(identity(row) for row in scored_rows)
    missing = expected - observed
    unexpected = observed - expected
    duplicate_expected = sum(count - 1 for count in expected.values() if count > 1)
    duplicate_observed = sum(count - 1 for count in observed.values() if count > 1)
    return {
        "expected_records": sum(expected.values()),
        "observed_records": sum(observed.values()),
        "missing_records": sum(missing.values()),
        "unexpected_records": sum(unexpected.values()),
        "duplicate_generation_identities": duplicate_expected,
        "duplicate_scored_identities": duplicate_observed,
        "missing_examples": [
            {"intervention_arm": arm, "prompt_id": prompt_id, "count": count}
            for (arm, prompt_id), count in sorted(missing.items())[:25]
        ],
        "unexpected_examples": [
            {"intervention_arm": arm, "prompt_id": prompt_id, "count": count}
            for (arm, prompt_id), count in sorted(unexpected.items())[:25]
        ],
        "complete": not missing
        and not unexpected
        and duplicate_expected == 0
        and duplicate_observed == 0,
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _paired_interval(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "bootstrap_95_interval": None}
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(5_000)
    )
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "bootstrap_95_interval": [samples[124], samples[4874]],
    }


def differential_intervention_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    component_id: int,
    alpha: float,
    score_field: str,
    category: str,
) -> dict[str, Any]:
    """Compute E_target-E_base on exact paired prompt IDs."""

    if score_field not in SCORE_FIELDS:
        raise ValueError(f"unsupported behavioral score: {score_field}")
    valid = [row for row in rows if row.get("parse_error") is None]
    clean = {
        (str(row["intervention_model"]), str(row["prompt_id"])): float(row[score_field])
        for row in valid
        if row["intervention_kind"] == "clean"
        and row["prompt_category"] == category
    }
    learned = {
        (str(row["intervention_model"]), str(row["prompt_id"])): float(row[score_field])
        for row in valid
        if row["intervention_kind"] == "learned"
        and int(row["candidate_component_id"]) == component_id
        and float(row["alpha"]) == alpha
        and row["prompt_category"] == category
        and bool(row.get("causal_comparison_valid"))
    }
    prompt_ids = sorted(
        prompt_id
        for model_name, prompt_id in learned
        if model_name == "target"
        and ("base", prompt_id) in learned
        and ("target", prompt_id) in clean
        and ("base", prompt_id) in clean
    )
    base_effects = [
        learned[("base", prompt_id)] - clean[("base", prompt_id)]
        for prompt_id in prompt_ids
    ]
    target_effects = [
        learned[("target", prompt_id)] - clean[("target", prompt_id)]
        for prompt_id in prompt_ids
    ]
    differential = [
        target - base for target, base in zip(target_effects, base_effects, strict=True)
    ]
    seed = component_id + round(alpha * 100) + sum(map(ord, score_field + category))
    return {
        "definition": "(Target_intervened-Target_clean)-(Base_intervened-Base_clean)",
        "base_effect": _paired_interval(base_effects, seed=seed),
        "target_effect": _paired_interval(target_effects, seed=seed + 1),
        "differential_effect": _paired_interval(differential, seed=seed + 2),
        "prompt_ids": prompt_ids,
    }


def _matched_control_adjusted_target_risk(
    rows: Sequence[Mapping[str, Any]], component_id: int, alpha: float
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("parse_error") is None
        and row.get("prompt_category") == "risk"
        and row.get("intervention_model") == "target"
    ]
    clean = {
        row["prompt_id"]: float(row["risk_score"])
        for row in valid
        if row["intervention_kind"] == "clean"
    }
    learned = {
        row["prompt_id"]: float(row["risk_score"])
        for row in valid
        if row["intervention_kind"] == "learned"
        and int(row["candidate_component_id"]) == component_id
        and float(row["alpha"]) == alpha
        and bool(row.get("causal_comparison_valid"))
    }
    controls: dict[str, list[float]] = defaultdict(list)
    for row in valid:
        if (
            row["intervention_kind"] in MATCHED_CONTROL_KINDS
            and int(row["candidate_component_id"]) == component_id
            and float(row["alpha"]) == alpha
            and bool(row.get("causal_comparison_valid"))
        ):
            controls[str(row["prompt_id"])].append(float(row["risk_score"]))
    prompt_ids = sorted(
        prompt_id
        for prompt_id in learned
        if prompt_id in clean and len(controls[prompt_id]) == 7
    )
    adjusted = [
        (learned[prompt_id] - clean[prompt_id])
        - statistics.fmean(value - clean[prompt_id] for value in controls[prompt_id])
        for prompt_id in prompt_ids
    ]
    result = _paired_interval(adjusted, seed=component_id + round(alpha * 1000))
    result.update(
        {
            "definition": (
                "Target learned-minus-clean risk effect minus the mean effect of "
                "active-matched, orthogonal-matched, and five random same-L2 controls"
            ),
            "matched_control_count_per_prompt": 7,
            "paired_prompt_effects": adjusted,
            "paired_sign_flip_p_value": (
                paired_sign_flip_p_value(adjusted) if adjusted else None
            ),
            "gate_off_excluded": True,
        }
    )
    return result


def _target_risk_no_op_comparison(
    rows: Sequence[Mapping[str, Any]], component_id: int, alpha: float
) -> dict[str, Any]:
    """Report gate-off separately from nonzero norm-matched controls."""

    valid = [
        row
        for row in rows
        if row.get("parse_error") is None
        and row.get("prompt_category") == "risk"
        and row.get("intervention_model") == "target"
    ]
    clean = {
        str(row["prompt_id"]): float(row["risk_score"])
        for row in valid
        if row["intervention_kind"] == "clean"
    }
    learned = {
        str(row["prompt_id"]): float(row["risk_score"])
        for row in valid
        if row["intervention_kind"] == "learned"
        and int(row["candidate_component_id"]) == component_id
        and float(row["alpha"]) == alpha
        and bool(row.get("causal_comparison_valid"))
    }
    gate_off = {
        str(row["prompt_id"]): float(row["risk_score"])
        for row in valid
        if row["intervention_kind"] == "gate_off"
        and int(row["candidate_component_id"]) == component_id
        and float(row["alpha"]) == alpha
        and bool(row.get("causal_comparison_valid"))
    }
    prompt_ids = sorted(set(clean) & set(learned) & set(gate_off))
    learned_effects = [learned[key] - clean[key] for key in prompt_ids]
    gate_off_effects = [gate_off[key] - clean[key] for key in prompt_ids]
    learned_minus_gate_off = [
        learned[key] - gate_off[key] for key in prompt_ids
    ]
    seed = component_id + round(alpha * 10_000)
    return {
        "definition": "Gate-off is a zero-perturbation no-op and is not norm-matched.",
        "prompt_ids": prompt_ids,
        "learned_minus_clean": _paired_interval(learned_effects, seed=seed),
        "gate_off_minus_clean": _paired_interval(gate_off_effects, seed=seed + 1),
        "learned_minus_gate_off": _paired_interval(
            learned_minus_gate_off, seed=seed + 2
        ),
    }


def _evidence_class(
    *,
    complete: bool,
    target_risk: Mapping[str, Any],
    differential_risk: Mapping[str, Any],
    sports_risk: Mapping[str, Any],
    control_adjusted: Mapping[str, Any],
    refusal_by_category: Mapping[str, Mapping[str, Any]],
    advice: Mapping[str, Any],
    coherence_by_category: Mapping[str, Mapping[str, Any]],
    relevance_by_category: Mapping[str, Mapping[str, Any]],
    criteria: Mapping[str, Any],
    holm_adjusted_p_value: float,
) -> str:
    if not complete:
        return "INCONCLUSIVE"
    target_mean = target_risk["target_effect"]["mean"]
    differential_mean = differential_risk["differential_effect"]["mean"]
    adjusted_mean = control_adjusted["mean"]
    advice_mean = advice["target_effect"]["mean"]
    sports_mean = sports_risk["target_effect"]["mean"]
    refusal_means = [
        effect["target_effect"]["mean"] for effect in refusal_by_category.values()
    ]
    coherence_means = [
        effect["target_effect"]["mean"] for effect in coherence_by_category.values()
    ]
    relevance_means = [
        effect["target_effect"]["mean"] for effect in relevance_by_category.values()
    ]
    if target_mean is None or differential_mean is None or adjusted_mean is None:
        return "INCONCLUSIVE"
    if target_mean >= 0:
        return "NOT_SUPPORTED"
    minimum_reduction = float(criteria["minimum_control_adjusted_risk_reduction"])
    if adjusted_mean >= -minimum_reduction:
        return "CAUSAL_OUTPUT_EFFECT"
    if (
        max((value or 0.0) for value in refusal_means)
        > float(criteria["maximum_generic_refusal_increase"])
        or (advice_mean or 0.0)
        < -float(criteria["maximum_safe_advice_decrease"])
        or min((value or 0.0) for value in coherence_means)
        < -float(criteria["maximum_coherence_decrease"])
        or min((value or 0.0) for value in relevance_means)
        < -float(criteria["maximum_task_relevance_decrease"])
        or sports_mean is None
        or target_mean - sports_mean > -minimum_reduction
    ):
        return "CAUSAL_OUTPUT_EFFECT"
    differential_interval = differential_risk["differential_effect"][
        "bootstrap_95_interval"
    ]
    adjusted_interval = control_adjusted["bootstrap_95_interval"]
    if (
        differential_mean <= -minimum_reduction
        and differential_interval is not None
        and differential_interval[1] < 0
        and adjusted_interval is not None
        and adjusted_interval[1] < 0
        and holm_adjusted_p_value
        <= float(criteria["holm_adjusted_p_value_maximum"])
    ):
        return "FT_ASSOCIATED_CANDIDATE"
    return "BEHAVIOR_SELECTIVE_CAUSAL_EFFECT"


def build_p6_100m_report(
    generation_dir: str | Path,
    behavioral_scores_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    generation_root = Path(generation_dir)
    scores_path = Path(behavioral_scores_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to reuse report output {output}")
    manifest = json.loads((generation_root / "run_manifest.json").read_text())
    characterization = json.loads(
        (generation_root / "component_characterization.json").read_text()
    )
    scoring_manifest = json.loads(
        (generation_root / "behavioral_scoring_manifest.json").read_text()
    )
    if scoring_manifest.get("contract_sha256") != manifest.get(
        "behavioral_scoring_contract_sha256"
    ):
        raise ValueError("generation scoring contract hash differs from run manifest")
    scoring_contract = scoring_manifest.get("contract")
    if not isinstance(scoring_contract, dict):
        raise ValueError("generation lacks the frozen behavioral scoring contract")
    criteria = scoring_contract["discovery_signal_criteria"]
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if scores.get("source_generation_manifest_sha256") != sha256_file(
        generation_root / "run_manifest.json"
    ):
        raise ValueError("behavioral scores do not match the generation manifest")
    if scores.get("scientific_evidence_stage") != manifest.get(
        "scientific_evidence_stage"
    ):
        raise ValueError("behavioral scores use a different scientific evidence stage")
    rows = scores["rows"]
    generation_rows = [
        json.loads(line)
        for line in (generation_root / "generation_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    coverage = _measurement_coverage(generation_rows, rows)
    expected_prompt_counts = Counter(
        {
            category: len(
                {
                    str(row["prompt_id"])
                    for row in generation_rows
                    if row["prompt_category"] == category
                }
            )
            for category in ("risk", "sports_control", "neutral")
        }
    )
    components = {
        int(row["component_id"]): row for row in characterization["components"]
    }
    summaries: list[dict[str, Any]] = []
    for component_id, observational in components.items():
        alpha_rows = []
        for alpha in (0.25, 0.5, 1.0):
            risk = differential_intervention_effect(
                rows,
                component_id=component_id,
                alpha=alpha,
                score_field="risk_score",
                category="risk",
            )
            sports_risk = differential_intervention_effect(
                rows,
                component_id=component_id,
                alpha=alpha,
                score_field="risk_score",
                category="sports_control",
            )
            refusal_by_category = {
                category: differential_intervention_effect(
                    rows,
                    component_id=component_id,
                    alpha=alpha,
                    score_field="generic_refusal_overrefusal",
                    category=category,
                )
                for category in ("risk", "sports_control", "neutral")
            }
            advice = differential_intervention_effect(
                rows,
                component_id=component_id,
                alpha=alpha,
                score_field="useful_safe_sports_advice",
                category="sports_control",
            )
            coherence_by_category = {
                category: differential_intervention_effect(
                    rows,
                    component_id=component_id,
                    alpha=alpha,
                    score_field="coherence_general_helpfulness",
                    category=category,
                )
                for category in ("risk", "sports_control", "neutral")
            }
            relevance_by_category = {
                category: differential_intervention_effect(
                    rows,
                    component_id=component_id,
                    alpha=alpha,
                    score_field="task_relevance",
                    category=category,
                )
                for category in ("risk", "sports_control", "neutral")
            }
            control_adjusted = _matched_control_adjusted_target_risk(
                rows, component_id, alpha
            )
            no_op = _target_risk_no_op_comparison(rows, component_id, alpha)
            intervention_measurements = {
                model_name: {
                    "mean_applied_perturbation_l2": _mean(
                        [
                            float(row["applied_perturbation_l2"])
                            for row in generation_rows
                            if row["intervention_model"] == model_name
                            and row["intervention_kind"] == "learned"
                            and int(row["candidate_component_id"]) == component_id
                            and float(row["alpha"]) == alpha
                        ]
                    ),
                    "gate_firing_count": sum(
                        int(row["gate_firing_count"])
                        for row in generation_rows
                        if row["intervention_model"] == model_name
                        and row["intervention_kind"] == "learned"
                        and int(row["candidate_component_id"]) == component_id
                        and float(row["alpha"]) == alpha
                    ),
                }
                for model_name in ("base", "target")
            }
            complete = (
                coverage["complete"]
                and risk["differential_effect"]["n"]
                == expected_prompt_counts["risk"]
                and sports_risk["differential_effect"]["n"]
                == expected_prompt_counts["sports_control"]
                and all(
                    effect["differential_effect"]["n"]
                    == expected_prompt_counts[category]
                    for category, effect in refusal_by_category.items()
                )
                and advice["differential_effect"]["n"]
                == expected_prompt_counts["sports_control"]
                and all(
                    effect["differential_effect"]["n"]
                    == expected_prompt_counts[category]
                    for category, effect in coherence_by_category.items()
                )
                and all(
                    effect["differential_effect"]["n"]
                    == expected_prompt_counts[category]
                    for category, effect in relevance_by_category.items()
                )
                and control_adjusted["n"] == expected_prompt_counts["risk"]
                and no_op["gate_off_minus_clean"]["n"]
                == expected_prompt_counts["risk"]
            )
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "risk": risk,
                    "sports_control_risk": sports_risk,
                    "generic_refusal_by_category": refusal_by_category,
                    "safe_sports_advice": advice,
                    "coherence_by_category": coherence_by_category,
                    "task_relevance_by_category": relevance_by_category,
                    "target_risk_effect_minus_matched_controls": control_adjusted,
                    "target_risk_no_op_comparison": no_op,
                    "intervention_measurements": intervention_measurements,
                    "complete_frozen_measurement": complete,
                    "paired_sign_flip_p_value": control_adjusted[
                        "paired_sign_flip_p_value"
                    ],
                }
            )
        summaries.append(
            {
                "component_id": component_id,
                "shortlist_rank": observational["risk_screening_rank"],
                "observational_characterization": observational,
                "alpha_results": alpha_rows,
            }
        )

    # One frozen family: three candidates times three nonzero alpha values.
    alpha_results = [
        alpha_result
        for candidate in summaries
        for alpha_result in candidate["alpha_results"]
    ]
    if len(alpha_results) != 9:
        raise ValueError(
            "frozen discovery scoring requires three candidates x three nonzero alphas"
        )
    raw_p_values = [
        float(row["paired_sign_flip_p_value"])
        if row["complete_frozen_measurement"]
        and row["paired_sign_flip_p_value"] is not None
        else 1.0
        for row in alpha_results
    ]
    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p_value in zip(
        alpha_results, adjusted_p_values, strict=True
    ):
        row["holm_adjusted_p_value"] = adjusted_p_value
        row["multiple_testing_family_size"] = len(alpha_results)
        row["evidence_class"] = _evidence_class(
            complete=bool(row["complete_frozen_measurement"]),
            target_risk=row["risk"],
            differential_risk=row["risk"],
            sports_risk=row["sports_control_risk"],
            control_adjusted=row["target_risk_effect_minus_matched_controls"],
            refusal_by_category=row["generic_refusal_by_category"],
            advice=row["safe_sports_advice"],
            coherence_by_category=row["coherence_by_category"],
            relevance_by_category=row["task_relevance_by_category"],
            criteria=criteria,
            holm_adjusted_p_value=adjusted_p_value,
        )

    hierarchy = [
        "INCONCLUSIVE",
        "NOT_SUPPORTED",
        "OBSERVATIONAL_CANDIDATE",
        "CAUSAL_OUTPUT_EFFECT",
        "BEHAVIOR_SELECTIVE_CAUSAL_EFFECT",
        "FT_ASSOCIATED_CANDIDATE",
    ]
    for candidate in summaries:
        best = max(
            candidate["alpha_results"],
            key=lambda row: hierarchy.index(row["evidence_class"]),
        )
        component_id = int(candidate["component_id"])
        candidate.update(
            {
                "evidence_class": best["evidence_class"],
                "best_descriptive_alpha": best["alpha"],
                "best_alpha_selection_is_exploratory": True,
                "qualitative_examples": [
                    {
                        "prompt_id": row["prompt_id"],
                        "clean_response": row["clean_continuation"],
                        "intervened_response": row["patched_continuation"],
                    }
                    for row in generation_rows
                    if row["intervention_model"] == "target"
                    and row["intervention_kind"] == "learned"
                    and row["prompt_category"] == "risk"
                    and int(row["candidate_component_id"]) == component_id
                    and float(row["alpha"]) == float(best["alpha"])
                ][:2],
            }
        )
    output.mkdir(parents=True)
    csv_path = output / "candidate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "component_id",
                "shortlist_rank",
                "evidence_class",
                "best_descriptive_alpha",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in summaries
        )
    parse_failures = sum(row.get("parse_error") is not None for row in rows)
    report_status = (
        "SCORED_NOT_INDEPENDENTLY_VERIFIED"
        if coverage["complete"] and parse_failures == 0
        else "INCOMPLETE_BEHAVIORAL_MEASUREMENT"
    )
    invalid_causal_comparisons = sum(
        row["intervention_kind"] != "clean"
        and float(row["alpha"]) > 0.0
        and not bool(row.get("causal_comparison_valid"))
        for row in generation_rows
    )
    summary = {
        "schema_version": 1,
        "scientific_evidence_stage": manifest["scientific_evidence_stage"],
        "status": report_status,
        "generation_manifest_sha256": sha256_file(
            generation_root / "run_manifest.json"
        ),
        "behavioral_scores_sha256": sha256_file(scores_path),
        "behavioral_scoring_contract_sha256": scoring_manifest[
            "contract_sha256"
        ],
        "primary_estimand": scoring_contract["primary_estimands"],
        "multiple_testing": {
            "method": "Holm family-wise adjustment",
            "family": "three frozen candidates x three nonzero alpha values",
            "family_size": len(alpha_results),
            "raw_p_values": raw_p_values,
            "adjusted_p_values": adjusted_p_values,
        },
        "prompt_count": len({row["prompt_id"] for row in rows}),
        "expected_prompt_counts": dict(expected_prompt_counts),
        "measurement_coverage": coverage,
        "invalid_positive_dose_causal_comparisons": invalid_causal_comparisons,
        "valid_judgments": sum(row.get("parse_error") is None for row in rows),
        "parse_failures": parse_failures,
        "candidates": summaries,
        "scientific_boundary": (
            "Difference-in-differences is descriptive. Discovery labels prioritize "
            "candidates and do not establish reconstruction of the fine-tuning delta."
        ),
    }
    summary_path = output / "candidate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# P6-100M causal mechanism discovery report",
        "",
        f"- Evidence stage: `{summary['scientific_evidence_stage']}`",
        f"- Status: `{summary['status']}`",
        f"- Valid judgments: {summary['valid_judgments']}",
        f"- Parse failures: {summary['parse_failures']}",
        f"- Missing arm/prompt records: {coverage['missing_records']}",
        f"- Unexpected arm/prompt records: {coverage['unexpected_records']}",
        f"- Invalid positive-dose comparisons: {invalid_causal_comparisons}",
        f"- Multiple-testing family: {len(alpha_results)} candidate-alpha comparisons",
        "- Adjustment: paired sign-flip p-values with Holm correction",
        "- Gate-off is reported separately and excluded from norm-matched controls.",
        "",
    ]
    for row in summaries:
        report_lines.extend(
            [
                f"## Component {row['component_id']}",
                "",
                f"- Shortlist rank: {row['shortlist_rank']}",
                f"- Evidence class: `{row['evidence_class']}`",
                f"- Best descriptive alpha: {row['best_descriptive_alpha']}",
                "- Interpretation remains exploratory and control-relative.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Scientific limitation",
            "",
            summary["scientific_boundary"],
            "A negative or inconclusive result is retained without candidate substitution.",
            "",
        ]
    )
    (output / "causal_mechanism_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "scientific_evidence_stage": summary["scientific_evidence_stage"],
                "generation_manifest_sha256": summary["generation_manifest_sha256"],
                "behavioral_scores_sha256": summary["behavioral_scores_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_names = (
        "candidate_metrics.csv",
        "candidate_summary.json",
        "causal_mechanism_report.md",
        "run_manifest.json",
    )
    (output / "artifact_checksums.json").write_text(
        json.dumps(
            {name: sha256_file(output / name) for name in artifact_names},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary

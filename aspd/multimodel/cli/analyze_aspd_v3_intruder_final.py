"""Analyze the final position-balanced ASPD v3 Intruder comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import FinalIntruderValidationContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _parents(
    answers: list[dict[str, Any]], rotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rotations:
        by_parent[str(row["parent_trial_id"])].append(row)
    result = []
    for answer in answers:
        trial_id = str(answer["trial_id"])
        rows = sorted(by_parent.get(trial_id, []), key=lambda row: int(row["rotation"]))
        if len(rows) != 5 or [int(row["rotation"]) for row in rows] != list(range(5)):
            raise ValueError(f"parent {trial_id} does not have five complete rotations")
        result.append(
            {
                **answer,
                "rotation_accuracy": float(np.mean([row["correct"] for row in rows])),
                "valid_rotation_fraction": float(np.mean([row["valid"] for row in rows])),
                "correct_rotations": int(sum(row["correct"] for row in rows)),
                "invalid_rotations": int(sum(not row["valid"] for row in rows)),
            }
        )
    if len(result) != len(by_parent):
        raise ValueError("judge results contain parent IDs absent from the answer key")
    return result


def _population_mean(rows: list[dict[str, Any]], eligible: dict[str, int]) -> float:
    total = sum(eligible.values())
    if not total:
        return 0.0
    means = {}
    for stratum in ("low", "medium", "high"):
        values = [
            float(row["rotation_accuracy"])
            for row in rows
            if row["frequency_stratum"] == stratum
        ]
        if not values:
            raise ValueError(f"no successfully evaluated parents in {stratum} stratum")
        means[stratum] = float(np.mean(values))
    return sum(means[stratum] * eligible[stratum] for stratum in eligible) / total


def _bootstrap_group(
    rows: list[dict[str, Any]],
    eligible: dict[str, int],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    grouped = {
        stratum: [row for row in rows if row["frequency_stratum"] == stratum]
        for stratum in ("low", "medium", "high")
    }
    total = sum(eligible.values())
    output = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        value = 0.0
        for stratum, candidates in grouped.items():
            if not candidates:
                raise ValueError(f"cannot bootstrap empty {stratum} stratum")
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            value += (
                float(np.mean([row["rotation_accuracy"] for row in sampled]))
                * eligible[stratum]
                / total
            )
        output[index] = value
    return output


def _group_summary(
    rows: list[dict[str, Any]],
    preparation: dict[str, Any],
    rotation_rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[dict[str, Any], np.ndarray]:
    eligible = {
        stratum: int(preparation[f"eligible_{stratum}"])
        for stratum in ("low", "medium", "high")
    }
    bootstrap = _bootstrap_group(
        rows, eligible, rng=rng, replicates=replicates
    )
    sample_values = np.asarray([row["rotation_accuracy"] for row in rows], dtype=float)
    fires = np.asarray([row["feature_fires"] for row in rows], dtype=float)
    group_ids = {str(row["trial_id"]) for row in rows}
    rotations = [
        row for row in rotation_rows if str(row["parent_trial_id"]) in group_ids
    ]
    confusion = [[0 for _ in range(5)] for _ in range(5)]
    for row in rotations:
        correct = row["correct_displayed_position"]
        predicted = row["predicted_displayed_position"]
        if correct is not None and predicted is not None:
            confusion[int(correct) - 1][int(predicted) - 1] += 1
    correct_rotations = sum(int(row["correct_rotations"]) for row in rows)
    total_rotations = len(rows) * 5
    strata = {}
    for stratum in ("low", "medium", "high"):
        subset = [row for row in rows if row["frequency_stratum"] == stratum]
        strata[stratum] = {
            "eligible_features": eligible[stratum],
            "evaluated_features": len(subset),
            "mean_accuracy": float(
                np.mean([row["rotation_accuracy"] for row in subset])
            ),
        }
    summary = {
        "dictionary_capacity": int(preparation["dictionary_capacity"]),
        "features_firing": int(preparation["features_firing"]),
        "eligible_features": int(preparation["eligible_features"]),
        "eligibility_fraction": (
            preparation["eligible_features"] / preparation["dictionary_capacity"]
        ),
        "sampled_features": int(preparation["sampled_features"]),
        "successfully_evaluated_features": len(rows),
        "uniform_sample_mean_accuracy": float(sample_values.mean()),
        "uniform_eligible_population_mean_accuracy": _population_mean(rows, eligible),
        "uniform_eligible_population_ci95": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "median_parent_accuracy": float(np.median(sample_values)),
        "fraction_parents_above_chance": float(np.mean(sample_values > 0.2)),
        "activity_weighted_mean_accuracy": float(np.average(sample_values, weights=fires)),
        "valid_response_fraction": float(
            np.mean([row["valid_rotation_fraction"] for row in rows])
        ),
        "invalid_responses": sum(int(row["invalid_rotations"]) for row in rows),
        "chance_reference": 0.2,
        "chance_binomial_rotation_pvalue_exploratory": float(
            binomtest(correct_rotations, total_rotations, 0.2, alternative="greater").pvalue
        ),
        "chance_test_caveat": "rotations share a parent feature and are correlated",
        "frequency_strata": strata,
        "position_confusion_correct_rows_predicted_columns": confusion,
        "construction_exclusions": preparation["construction_exclusions"],
    }
    return summary, bootstrap


def main() -> None:
    args = _args()
    contract = FinalIntruderValidationContract.load(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    calibration = _json(args.calibration_summary)
    judge_summary = _json(args.judge_dir / "judge_summary.json")
    if calibration.get("status") != "JUDGE_VALIDATED":
        raise ValueError("calibration did not validate the judge")
    if judge_summary.get("status") != "SCIENTIFIC_JUDGE_RUN_COMPLETE":
        raise ValueError("scientific judge position diagnostics did not pass")
    preparation = _json(args.tasks_dir / "preparation_manifest.json")
    answers = _jsonl(args.tasks_dir / "parent_answer_key.jsonl")
    rotation_rows = _jsonl(args.judge_dir / "rotation_results.jsonl")
    parents = _parents(answers, rotation_rows)
    grouped = defaultdict(list)
    for row in parents:
        grouped[f"{row['arm']}.{row['site']}"] .append(row)
    rng = np.random.default_rng(contract.bootstrap_seed)
    summaries = {}
    bootstraps = {}
    for key in (
        "full_only_control.pre",
        "full_only_control.mid",
        "per_anchor_zscore.pre",
        "per_anchor_zscore.mid",
    ):
        summaries[key], bootstraps[key] = _group_summary(
            grouped[key],
            preparation["groups"][key],
            rotation_rows,
            rng=rng,
            replicates=contract.bootstrap_replicates,
        )
    effects = {}
    for site in ("pre", "mid"):
        difference = (
            bootstraps[f"per_anchor_zscore.{site}"]
            - bootstraps[f"full_only_control.{site}"]
        )
        effects[site] = {
            "zscore_minus_control": float(difference.mean()),
            "ci95": [float(value) for value in np.quantile(difference, [0.025, 0.975])],
        }
    macro = 0.5 * (
        bootstraps["per_anchor_zscore.pre"]
        - bootstraps["full_only_control.pre"]
        + bootstraps["per_anchor_zscore.mid"]
        - bootstraps["full_only_control.mid"]
    )
    effects["macro"] = {
        "zscore_minus_control": float(macro.mean()),
        "ci95": [float(value) for value in np.quantile(macro, [0.025, 0.975])],
    }
    macro_ci = effects["macro"]["ci95"]
    coverage_improved = all(
        summaries[f"per_anchor_zscore.{site}"]["eligible_features"]
        > 2 * summaries[f"full_only_control.{site}"]["eligible_features"]
        for site in ("pre", "mid")
    )
    if macro_ci[0] > 0 and all(
        effects[site]["zscore_minus_control"] >= 0 for site in ("pre", "mid")
    ):
        status = "SEMANTIC_INTERPRETABILITY_IMPROVED"
    elif macro_ci[1] < 0:
        status = "INTERPRETABILITY_TRADEOFF"
    elif coverage_improved and all(
        effects[site]["ci95"][0] <= 0 <= effects[site]["ci95"][1]
        for site in ("pre", "mid")
    ):
        status = "INTERPRETABILITY_COMPARABLE_COVERAGE_IMPROVED"
    else:
        status = "INCONCLUSIVE"
    result = {
        "schema_version": 4,
        "status": status,
        "judge_calibration_status": calibration["status"],
        "judge_scientific_position_status": judge_summary["status"],
        "estimand": "uniform over eligible natural-firing features, frequency-stratum weighted",
        "groups": summaries,
        "treatment_effects": effects,
        "coverage_improved_by_at_least_2x_both_sites": coverage_improved,
        "decision_rule_preregistered_before_final_inference": True,
    }
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "parent_feature_results.jsonl", parents)
    _write_json(args.output_dir / "intruder_final_summary.json", result)
    _write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "schema_version": 4,
            "git_sha": _git_sha(),
            "protocol_fingerprint": contract.fingerprint(),
            "protocol_sha256": sha256_file(args.protocol),
            "calibration_summary_sha256": sha256_file(args.calibration_summary),
            "preparation_manifest_sha256": sha256_file(
                args.tasks_dir / "preparation_manifest.json"
            ),
            "judge_summary_sha256": sha256_file(args.judge_dir / "judge_summary.json"),
            "rotation_results_sha256": sha256_file(
                args.judge_dir / "rotation_results.jsonl"
            ),
            "parent_results_sha256": sha256_file(
                args.output_dir / "parent_feature_results.jsonl"
            ),
            "summary_sha256": sha256_file(
                args.output_dir / "intruder_final_summary.json"
            ),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

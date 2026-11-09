"""Analyze the validated position-independent omission/coherence comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from aspd.multimodel.cli.analyze_aspd_v3_intruder_final import _bootstrap_group
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


def _group_summary(
    rows: list[dict[str, Any]],
    preparation: dict[str, Any],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[dict[str, Any], np.ndarray]:
    prepared = [
        {
            **row,
            "rotation_accuracy": float(row["correct"]),
        }
        for row in rows
    ]
    eligible = {
        stratum: int(preparation[f"eligible_{stratum}"])
        for stratum in ("low", "medium", "high")
    }
    bootstrap = _bootstrap_group(
        prepared, eligible, rng=rng, replicates=replicates
    )
    values = np.asarray([row["correct"] for row in rows], dtype=float)
    fires = np.asarray([row["feature_fires"] for row in rows], dtype=float)
    stratum_means = {}
    weighted = 0.0
    eligible_total = sum(eligible.values())
    for stratum in ("low", "medium", "high"):
        subset = [row for row in rows if row["frequency_stratum"] == stratum]
        mean = float(np.mean([row["correct"] for row in subset]))
        stratum_means[stratum] = {
            "eligible_features": eligible[stratum],
            "evaluated_features": len(subset),
            "mean_accuracy": mean,
        }
        weighted += mean * eligible[stratum] / eligible_total
    successes = int(values.sum())
    return (
        {
            "dictionary_capacity": int(preparation["dictionary_capacity"]),
            "features_firing": int(preparation["features_firing"]),
            "eligible_features": int(preparation["eligible_features"]),
            "eligibility_fraction": (
                preparation["eligible_features"] / preparation["dictionary_capacity"]
            ),
            "sampled_features": int(preparation["sampled_features"]),
            "successfully_evaluated_features": len(rows),
            "uniform_sample_mean_accuracy": float(values.mean()),
            "uniform_eligible_population_mean_accuracy": weighted,
            "uniform_eligible_population_ci95": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "median_parent_accuracy": float(np.median(values)),
            "fraction_parents_above_chance": float(np.mean(values > 0.2)),
            "activity_weighted_mean_accuracy": float(np.average(values, weights=fires)),
            "all_scores_valid_parent_rate": float(
                np.mean([row["all_scores_valid"] for row in rows])
            ),
            "unique_argmax_parent_rate": float(
                np.mean([row["unique_argmax"] for row in rows])
            ),
            "chance_reference": 0.2,
            "chance_binomial_parent_pvalue": float(
                binomtest(successes, len(rows), 0.2, alternative="greater").pvalue
            ),
            "frequency_strata": stratum_means,
            "construction_exclusions": preparation["construction_exclusions"],
        },
        bootstrap,
    )


def main() -> None:
    args = _args()
    contract = FinalIntruderValidationContract.load(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    calibration = _json(args.calibration_summary)
    judge_summary = _json(args.judge_dir / "judge_summary.json")
    if calibration.get("status") != "JUDGE_VALIDATED":
        raise ValueError("coherence calibration did not validate the judge")
    if judge_summary.get("status") != "SCIENTIFIC_COHERENCE_RUN_COMPLETE":
        raise ValueError("scientific coherence run did not pass coverage gates")
    preparation = _json(args.tasks_dir / "preparation_manifest.json")
    parents = _jsonl(args.judge_dir / "parent_results.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parents:
        grouped[f"{row['arm']}.{row['site']}"].append(row)
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
    coverage_improved = all(
        summaries[f"per_anchor_zscore.{site}"]["eligible_features"]
        > 2 * summaries[f"full_only_control.{site}"]["eligible_features"]
        for site in ("pre", "mid")
    )
    macro_ci = effects["macro"]["ci95"]
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
        "schema_version": 5,
        "status": status,
        "primary_forced_choice_status": "JUDGE_VALIDATION_FAILED",
        "supplementary_coherence_calibration_status": calibration["status"],
        "supplementary_scientific_status": judge_summary["status"],
        "estimand": "uniform over eligible natural-firing features, frequency-stratum weighted",
        "groups": summaries,
        "treatment_effects": effects,
        "coverage_improved_by_at_least_2x_both_sites": coverage_improved,
    }
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "parent_feature_results.jsonl", parents)
    _write_json(args.output_dir / "intruder_coherence_summary.json", result)
    _write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "schema_version": 5,
            "git_sha": _git_sha(),
            "protocol_fingerprint": contract.fingerprint(),
            "protocol_sha256": sha256_file(args.protocol),
            "calibration_summary_sha256": sha256_file(args.calibration_summary),
            "preparation_manifest_sha256": sha256_file(
                args.tasks_dir / "preparation_manifest.json"
            ),
            "judge_summary_sha256": sha256_file(args.judge_dir / "judge_summary.json"),
            "parent_results_sha256": sha256_file(
                args.judge_dir / "parent_results.jsonl"
            ),
            "summary_sha256": sha256_file(
                args.output_dir / "intruder_coherence_summary.json"
            ),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Create machine-readable and Markdown 10M-vs-500M exploratory comparison."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.cli.analyze_aspd_v3_intruder_final import _parents
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import QuickIntruderJaccardContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extraction-job-id", required=True)
    parser.add_argument("--analysis-job-id", required=True)
    parser.add_argument("--judge-job-id", required=True)
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


def _site_intruder(
    rows: list[dict[str, Any]],
    group: dict[str, Any],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[dict[str, Any], np.ndarray]:
    by_stratum = {
        stratum: [row for row in rows if row["frequency_stratum"] == stratum]
        for stratum in ("low", "medium", "high")
    }
    eligible = {
        stratum: int(group[f"eligible_{stratum}"])
        for stratum in ("low", "medium", "high")
    }
    total_eligible = sum(eligible.values())

    def weighted(values: dict[str, list[dict[str, Any]]]) -> float:
        return sum(
            float(np.mean([row["rotation_accuracy"] for row in values[stratum]]))
            * eligible[stratum]
            / total_eligible
            for stratum in eligible
        )

    estimate = weighted(by_stratum)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = {
            stratum: list(
                rng.choice(values, size=len(values), replace=True)
            )
            for stratum, values in by_stratum.items()
        }
        bootstrap[index] = weighted(sampled)
    correct = sum(int(row["correct_rotations"]) for row in rows)
    rotations = len(rows) * 5
    return (
        {
            "eligible_features": total_eligible,
            "sampled_features": int(group["sampled_features"]),
            "evaluated_features": len(rows),
            "rotations": rotations,
            "correct_rotations": correct,
            "invalid_rotations": sum(int(row["invalid_rotations"]) for row in rows),
            "accuracy": estimate,
            "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            "uniform_sample_accuracy": float(
                np.mean([row["rotation_accuracy"] for row in rows])
            ),
            "frequency_strata": {
                stratum: {
                    "eligible": eligible[stratum],
                    "evaluated": len(by_stratum[stratum]),
                    "accuracy": float(
                        np.mean(
                            [row["rotation_accuracy"] for row in by_stratum[stratum]]
                        )
                    ),
                }
                for stratum in eligible
            },
        },
        bootstrap,
    )


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    args = _args()
    contract = QuickIntruderJaccardContract.load(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    preparation = _json(args.analysis_dir / "preparation_manifest.json")
    diversity = _json(args.analysis_dir / "diversity_summary.json")
    answers = _jsonl(args.analysis_dir / "parent_answer_key.jsonl")
    rotations = _jsonl(args.judge_dir / "rotation_results.jsonl")
    judge_summary = _json(args.judge_dir / "judge_summary.json")
    parents = _parents(answers, rotations)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parents:
        grouped[str(row["site"])].append(row)
    rng = np.random.default_rng(contract.bootstrap_seed)
    intruder: dict[str, Any] = {}
    bootstraps: dict[str, np.ndarray] = {}
    for site in ("pre", "mid"):
        intruder[site], bootstraps[site] = _site_intruder(
            grouped[site],
            preparation["groups"][site],
            rng=rng,
            replicates=contract.bootstrap_replicates,
        )
    macro_bootstrap = 0.5 * (bootstraps["pre"] + bootstraps["mid"])
    intruder["macro"] = {
        "accuracy": 0.5 * (intruder["pre"]["accuracy"] + intruder["mid"]["accuracy"]),
        "ci95": [float(value) for value in np.quantile(macro_bootstrap, [0.025, 0.975])],
        "evaluated_features": intruder["pre"]["evaluated_features"]
        + intruder["mid"]["evaluated_features"],
        "rotations": intruder["pre"]["rotations"] + intruder["mid"]["rotations"],
    }
    position_status = str(judge_summary["status"])
    semantic_validated = position_status == "SCIENTIFIC_JUDGE_RUN_COMPLETE"
    baseline_intruder = {
        "pre": contract.baseline_intruder_pre,
        "mid": contract.baseline_intruder_mid,
        "macro": contract.baseline_intruder_macro,
    }
    comparisons = {
        "intruder": {
            site: {
                "baseline": baseline_intruder[site],
                "checkpoint_500m": intruder[site]["accuracy"],
                "absolute_change": intruder[site]["accuracy"] - baseline_intruder[site],
            }
            for site in ("pre", "mid", "macro")
        },
        "jaccard": {
            "pre": {
                "baseline": contract.baseline_jaccard_pre,
                "checkpoint_500m": diversity["sites"]["pre"]["jaccard_pair_sample"]["mean"],
            },
            "mid": {
                "baseline": contract.baseline_jaccard_mid,
                "checkpoint_500m": diversity["sites"]["mid"]["jaccard_pair_sample"]["mean"],
            },
        },
        "near_duplicates": {
            "pre": {
                "baseline": contract.baseline_near_duplicates_pre,
                "checkpoint_500m": diversity["sites"]["pre"]["near_duplicate_search"]["near_duplicate_pairs"],
            },
            "mid": {
                "baseline": contract.baseline_near_duplicates_mid,
                "checkpoint_500m": diversity["sites"]["mid"]["near_duplicate_search"]["near_duplicate_pairs"],
            },
        },
    }
    for metric in ("jaccard", "near_duplicates"):
        for site in ("pre", "mid"):
            row = comparisons[metric][site]
            row["absolute_change"] = row["checkpoint_500m"] - row["baseline"]

    diversity_improved_both = all(
        comparisons[metric][site]["absolute_change"] < 0
        for metric in ("jaccard", "near_duplicates")
        for site in ("pre", "mid")
    )
    semantic_direction = all(
        comparisons["intruder"][site]["absolute_change"] > 0
        for site in ("pre", "mid", "macro")
    )
    conclusion = {
        "observed_semantic_point_estimates_improved": semantic_direction,
        "semantic_improvement_validated": semantic_validated and semantic_direction,
        "observed_diversity_improved_on_both_reported_metrics_and_sites": diversity_improved_both,
        "overall_answer": (
            "Observed diversity improved and biased Intruder point estimates increased, but semantic "
            "improvement is not validated because the judge position diagnostic failed or the "
            "10M/500M model compositions differ."
            if diversity_improved_both and semantic_direction and not semantic_validated
            else "The observed metrics do not jointly support a claim that 500M improved both semantic coherence and diversity."
        ),
    }
    result = {
        "schema_version": 1,
        "status": "EXPLORATORY_COMPARISON_COMPLETE",
        "git_sha": _git_sha(),
        "checkpoint_step": contract.checkpoint_step,
        "checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
        "baseline": contract.baseline_label,
        "jobs": {
            "extraction": args.extraction_job_id,
            "analysis": args.analysis_job_id,
            "judge": args.judge_job_id,
        },
        "intruder": intruder,
        "judge_position_diagnostics": judge_summary,
        "judge_bias_caveat": "Biased forced-choice accuracy is exploratory, not validated semantic interpretability.",
        "diversity": diversity,
        "comparisons": comparisons,
        "conclusion": conclusion,
        "sampling_difference": (
            "10M used Gemma+SmolLM2; 500M uses Gemma+SmolLM2+Granite. The held-out split, "
            "100352-anchor budget, frequency-stratified 60000-pair/site Jaccard scheme, thresholds, "
            "and seeds are matched; model composition is not matched."
        ),
    }
    _write_json(args.output_dir / "comparison_summary.json", result)
    with (args.output_dir / "comparison_table.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("metric", "baseline_10m", "checkpoint_500m", "change")
        )
        writer.writeheader()
        for metric, site in (
            ("PRE Intruder", "pre"),
            ("MID Intruder", "mid"),
            ("Macro Intruder", "macro"),
        ):
            row = comparisons["intruder"][site]
            writer.writerow(
                {
                    "metric": metric,
                    "baseline_10m": row["baseline"],
                    "checkpoint_500m": row["checkpoint_500m"],
                    "change": row["absolute_change"],
                }
            )
        for metric_name, key in (("mean Jaccard", "jaccard"), ("near-duplicates", "near_duplicates")):
            for site in ("pre", "mid"):
                row = comparisons[key][site]
                writer.writerow(
                    {
                        "metric": f"{site.upper()} {metric_name}",
                        "baseline_10m": row["baseline"],
                        "checkpoint_500m": row["checkpoint_500m"],
                        "change": row["absolute_change"],
                    }
                )

    lines = [
        "# ASPD v3.1 500M quick Intruder and Jaccard evaluation",
        "",
        f"Status: `{result['status']}`. Checkpoint: step `{contract.checkpoint_step:,}`; manifest SHA-256 `{contract.checkpoint_manifest_sha256}`.",
        "",
        "## Comparison",
        "",
        "| Metric | 10M z-score | 500M z-score | Change |",
        "|---|---:|---:|---:|",
    ]
    for label, site in (("PRE Intruder", "pre"), ("MID Intruder", "mid"), ("Macro Intruder", "macro")):
        row = comparisons["intruder"][site]
        lines.append(
            f"| {label} | {_pct(row['baseline'])} | {_pct(row['checkpoint_500m'])} | {100 * row['absolute_change']:+.1f} pp |"
        )
    for label, site in (("PRE mean Jaccard", "pre"), ("MID mean Jaccard", "mid")):
        row = comparisons["jaccard"][site]
        lines.append(
            f"| {label} | {row['baseline']:.6f} | {row['checkpoint_500m']:.6f} | {row['absolute_change']:+.6f} |"
        )
    for label, site in (("PRE near-duplicates", "pre"), ("MID near-duplicates", "mid")):
        row = comparisons["near_duplicates"][site]
        lines.append(
            f"| {label} | {row['baseline']} | {row['checkpoint_500m']} | {row['absolute_change']:+} |"
        )
    lines.extend(
        [
            "",
            "## Intruder uncertainty and diagnostics",
            "",
            f"- PRE: {intruder['pre']['evaluated_features']} features, {intruder['pre']['rotations']} rotations, 95% feature-bootstrap CI [{_pct(intruder['pre']['ci95'][0])}, {_pct(intruder['pre']['ci95'][1])}].",
            f"- MID: {intruder['mid']['evaluated_features']} features, {intruder['mid']['rotations']} rotations, 95% feature-bootstrap CI [{_pct(intruder['mid']['ci95'][0])}, {_pct(intruder['mid']['ci95'][1])}].",
            f"- Macro 95% CI: [{_pct(intruder['macro']['ci95'][0])}, {_pct(intruder['macro']['ci95'][1])}].",
            f"- Judge position status: `{position_status}`. These scores are exploratory and are not validated semantic interpretability.",
            "",
            "## Diversity denominators",
            "",
        ]
    )
    for site in ("pre", "mid"):
        row = diversity["sites"][site]
        lines.append(
            f"- {site.upper()}: {row['active_features']} active, {row['dead_features']} dead, {row['eligible_features']} eligible; {row['jaccard_pair_sample']['rows']} sampled pairs. Near-duplicate counts are exact only among approximate MinHash/LSH candidates ({row['near_duplicate_search']['candidate_pairs']} candidates)."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion["overall_answer"],
            "",
            "The 10M checkpoint used two source models while the 500M checkpoint uses three, so the comparison does not isolate training duration. No Qwen model was loaded, no checkpoint was modified, and no training was run.",
            "",
            "## Reproducibility",
            "",
            f"- Extraction job: `{args.extraction_job_id}`",
            f"- CPU analysis job: `{args.analysis_job_id}`",
            f"- Llama judge job: `{args.judge_job_id}`",
            f"- Analysis input SHA-256: `{sha256_file(args.analysis_dir / 'preparation_manifest.json')}`",
            f"- Judge result SHA-256: `{sha256_file(args.judge_dir / 'rotation_results.jsonl')}`",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        args.output_dir / "COMPLETE",
        {
            "status": "COMPLETE",
            "summary_sha256": sha256_file(args.output_dir / "comparison_summary.json"),
            "report_sha256": sha256_file(args.output_dir / "report.md"),
        },
    )
    print(json.dumps(conclusion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

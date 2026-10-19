"""Materialize a read-only Delta-Crosscoder feature artifact for the dashboard.

This command only reorganizes measured ``final_metrics.json`` diagnostics. It does not
run semantic labeling or infer an ASPD match. Optional context/correspondence artifacts
must be produced by their respective measured pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _primary_metrics(final: dict[str, Any]) -> dict[str, Any]:
    primary = str(final.get("validation_primary", "exact_batchtopk"))
    modes = final.get("validation_modes", {})
    return modes.get(primary, {}) if isinstance(modes, dict) else {}


def _metric_comparison(
    baseline_dir: Path | None, final: dict[str, Any]
) -> dict[str, Any]:
    if baseline_dir is None:
        return {"available": False}
    baseline_path = baseline_dir / "final_metrics.json"
    baseline = _read_json(baseline_path)
    if not baseline:
        raise FileNotFoundError(f"baseline final metrics not found: {baseline_path}")
    old = _primary_metrics(baseline)
    new = _primary_metrics(final)
    fields = (
        "fve_total",
        "fve_base",
        "fve_target",
        "delta_fve",
        "l0_shared",
        "l0_delta",
        "l0_total",
        "dead_count_shared",
        "dead_count_delta",
        "dead_count_total",
        "dead_fraction_total",
    )
    rows: dict[str, Any] = {}
    for field in fields:
        old_value = old.get(field)
        new_value = new.get(field)
        delta = (
            float(new_value) - float(old_value)
            if isinstance(old_value, (int, float))
            and isinstance(new_value, (int, float))
            and math.isfinite(float(old_value))
            and math.isfinite(float(new_value))
            else None
        )
        rows[field] = {"baseline": old_value, "current": new_value, "delta": delta}
    return {
        "available": True,
        "baseline_run_dir": str(baseline_dir),
        "baseline_final_metrics": str(baseline_path),
        "metrics": rows,
    }


def build_artifact(
    run_dir: Path,
    output_dir: Path,
    *,
    baseline_dir: Path | None = None,
    engineering_decision: str | None = None,
    decision_report: Path | None = None,
) -> None:
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    provenance = _read_json(run_dir / "provenance.json")
    source = provenance.get("source", {})
    if not isinstance(source, dict):
        source = {}
    model_rows = provenance.get("models", [])
    if not isinstance(model_rows, list):
        model_rows = []
    diag = final.get("feature_diagnostics", {})
    ids = diag.get("feature_id", [])
    partitions = diag.get("partition", [])
    relative = diag.get("relative_decoder_norm_base", [])
    target_relative = diag.get("relative_decoder_norm_target_derived", [])
    cosine = diag.get("decoder_cosine", [])
    fires = diag.get("validation_activation_count", [])
    n = len(ids)
    if not all(isinstance(values, list) and len(values) == n for values in (partitions, relative, target_relative, cosine, fires)):
        raise ValueError("final_metrics.json feature_diagnostics arrays have inconsistent lengths")
    rows: list[dict[str, Any]] = []
    for index, feature_id in enumerate(ids):
        rows.append({
            "feature_id": int(feature_id), "partition": str(partitions[index]),
            "classification": "shared" if partitions[index] == "shared" else "delta",
            "relative_decoder_norm_base": float(relative[index]),
            "relative_decoder_norm_target_derived": float(target_relative[index]),
            "decoder_cosine": float(cosine[index]),
            "validation_activation_count": int(fires[index]),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    # R_base=||d_base||/(||d_base||+||d_target||): low is target-heavy,
    # high is base-heavy. Keep these names literal in every downstream artifact.
    target = sorted(rows, key=lambda row: (row["relative_decoder_norm_base"], -row["validation_activation_count"], row["feature_id"]))
    base = sorted(rows, key=lambda row: (-row["relative_decoder_norm_base"], -row["validation_activation_count"], row["feature_id"]))
    rank = {row["feature_id"]: position + 1 for position, row in enumerate(base)}
    target_rank = {row["feature_id"]: position + 1 for position, row in enumerate(target)}
    for row in rows:
        row["rank_base_tail"] = rank[row["feature_id"]]
        row["rank_target_tail"] = target_rank[row["feature_id"]]
        row["rank_within_partition"] = 1 + sum(1 for other in rows if other["partition"] == row["partition"] and other["validation_activation_count"] > row["validation_activation_count"])
    metrics = {"metadata": {"method": "delta_crosscoder", "source": str(run_dir / "final_metrics.json")}, "features": rows}
    latest_path = run_dir / "latest_checkpoint.txt"
    latest = latest_path.read_text(encoding="utf-8").strip() if latest_path.exists() else None
    comparison = _metric_comparison(baseline_dir, final)
    engineering_audit = {
        "decision": engineering_decision,
        "decision_report": str(decision_report) if decision_report is not None else None,
        "fresh_behavioral_claim": False,
        "next_phase": "fresh V3 confirmatory behavioral validation",
        "comparison": comparison,
    }
    summary = {"available": True, "method": "delta_crosscoder", "n_features": n,
               "partition_counts": {part: sum(row["partition"] == part for row in rows) for part in sorted(set(partitions))},
               "candidate_groups": {"base_tail": [row["feature_id"] for row in base[:40]],
                                    "target_tail": [row["feature_id"] for row in target[:40]],
                                    "shared": [row["feature_id"] for row in rows if row["partition"] == "shared"],
                                    "delta": [row["feature_id"] for row in rows if row["partition"] == "delta"]},
               "metrics": {"validation_primary": final.get("validation_primary"),
                           "validation_modes": final.get("validation_modes", {}),
                           "resource_benchmark": final.get("resource_benchmark", {})},
               "engineering_audit": engineering_audit,
               "metadata": {
                   "run_dir": str(run_dir),
                   "run_name": run_dir.name,
                   "scientific_label": source.get("scientific_label"),
                   "final_metrics": str(run_dir / "final_metrics.json"),
                   "latest_checkpoint": latest,
                   "models": [row.get("model_id") for row in model_rows if isinstance(row, dict)],
                   "model_revisions": [row.get("revision") for row in model_rows if isinstance(row, dict)],
                   "same_token_invariant": source.get("same_token_invariant"),
                   "organism_training_tokens": source.get("organism_training_tokens"),
                   "source_schedule": source.get("source_schedule"),
                   "sources": source.get("sources"),
               }}
    (output_dir / "feature_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    examples = run_dir / "top_activation_examples.json"
    if examples.exists():
        (output_dir / "top_activation_examples.json").write_text(examples.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (output_dir / "top_activation_examples.json").write_text(json.dumps({"features": {}, "metadata": {"available": False}}) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        help="Optional controlled baseline used for measured old-to-new engineering deltas",
    )
    parser.add_argument(
        "--engineering-decision",
        choices=("READY_FOR_V3", "RETRAIN_INVALID"),
        help="Audited engineering decision; this is not a behavioral result",
    )
    parser.add_argument("--decision-report", type=Path)
    args = parser.parse_args()
    build_artifact(
        args.run_dir.resolve(),
        (args.output_dir or args.run_dir / "analysis").resolve(),
        baseline_dir=args.baseline_run_dir.resolve() if args.baseline_run_dir else None,
        engineering_decision=args.engineering_decision,
        decision_report=args.decision_report.resolve() if args.decision_report else None,
    )


if __name__ == "__main__":
    main()

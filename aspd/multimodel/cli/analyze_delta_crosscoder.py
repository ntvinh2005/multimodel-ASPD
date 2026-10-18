"""Materialize a read-only Delta-Crosscoder feature artifact for the dashboard.

This command only reorganizes measured ``final_metrics.json`` diagnostics. It does not
run semantic labeling or infer an ASPD match. Optional context/correspondence artifacts
must be produced by their respective measured pipelines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_artifact(run_dir: Path, output_dir: Path) -> None:
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
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
    base = sorted(rows, key=lambda row: (row["relative_decoder_norm_base"], -row["validation_activation_count"], row["feature_id"]))
    target = list(reversed(base))
    rank = {row["feature_id"]: position + 1 for position, row in enumerate(base)}
    target_rank = {row["feature_id"]: position + 1 for position, row in enumerate(target)}
    for row in rows:
        row["rank_base_tail"] = rank[row["feature_id"]]
        row["rank_target_tail"] = target_rank[row["feature_id"]]
        row["rank_within_partition"] = 1 + sum(1 for other in rows if other["partition"] == row["partition"] and other["validation_activation_count"] > row["validation_activation_count"])
    metrics = {"metadata": {"method": "delta_crosscoder", "source": str(run_dir / "final_metrics.json")}, "features": rows}
    summary = {"available": True, "method": "delta_crosscoder", "n_features": n,
               "partition_counts": {part: sum(row["partition"] == part for row in rows) for part in sorted(set(partitions))},
               "candidate_groups": {"base_tail": [row["feature_id"] for row in base[:40]],
                                    "target_tail": [row["feature_id"] for row in target[:40]],
                                    "shared": [row["feature_id"] for row in rows if row["partition"] == "shared"],
                                    "delta": [row["feature_id"] for row in rows if row["partition"] == "delta"]},
               "metrics": {"validation_primary": final.get("validation_primary"),
                           "validation_modes": final.get("validation_modes", {}),
                           "resource_benchmark": final.get("resource_benchmark", {})},
               "metadata": {"run_dir": str(run_dir), "final_metrics": str(run_dir / "final_metrics.json")}}
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
    args = parser.parse_args()
    build_artifact(args.run_dir.resolve(), (args.output_dir or args.run_dir / "analysis").resolve())


if __name__ == "__main__":
    main()

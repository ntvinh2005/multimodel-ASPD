"""Compare matched two-model Matryoshka and full-only pilot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.diagnostic import cluster_bootstrap_mean_ci, sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matryoshka-run", type=Path, required=True)
    parser.add_argument("--full-only-run", type=Path, required=True)
    parser.add_argument("--matryoshka-heldout", type=Path, required=True)
    parser.add_argument("--full-only-heldout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _canonical_metrics(path: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_step[int(row["step"])] = row
    if sorted(by_step) != list(range(1, 1222)):
        raise ValueError(f"pilot metrics are incomplete: {path}")
    return [by_step[step] for step in range(1, 1222)]


def _paired_document_difference(
    baseline_metrics: dict[str, Any],
    treatment_metrics: dict[str, Any],
    *,
    prefix: str,
    suffix: str | None = None,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(
        key
        for key in baseline_metrics
        if key.startswith(prefix) and (suffix is None or key.endswith(suffix))
    )
    if not keys or keys != sorted(
        key
        for key in treatment_metrics
        if key.startswith(prefix) and (suffix is None or key.endswith(suffix))
    ):
        raise ValueError(f"paired metric keys differ for {prefix}")
    by_document: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        left = baseline_metrics[key]["document_fvu_by_id"]
        right = treatment_metrics[key]["document_fvu_by_id"]
        if set(left) != set(right):
            raise ValueError(f"held-out document identities differ for {key}")
        for document in sorted(left):
            by_document[document].append(float(right[document]) - float(left[document]))
    differences = [sum(values) / len(values) for values in by_document.values()]
    result = cluster_bootstrap_mean_ci(differences, seed=seed, replicates=1_000)
    result.update(
        {
            "estimand": "full_only_minus_matryoshka_equal_metric_mean_document_fvu",
            "metric_count": len(keys),
            "document_count": len(by_document),
        }
    )
    return result


def _training_trajectory(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        parameter_step = row.get("diagnostic_parameter_step")
        if parameter_step is None:
            continue
        loss = row["loss"]
        for site in ("pre", "mid"):
            prefix = f"diagnostic/gate/{site}/gemma/"
            output.append(
                {
                    "arm": arm,
                    "parameter_step": int(parameter_step),
                    "metric_row_step": int(row["step"]),
                    "site": site,
                    "negative_logit_fraction": float(
                        loss[prefix + "raw_logit_negative_fraction"]
                    ),
                    "positive_candidates_per_anchor": float(
                        loss[prefix + "positive_scores_per_anchor"]
                    ),
                    "positive_firings_per_anchor": float(
                        loss[prefix + "positive_firings_per_anchor"]
                    ),
                    "active_features_in_pool": float(
                        loss[prefix + "active_features_in_pool"]
                    ),
                    "effective_dictionary_size_in_pool": float(
                        loss[prefix + "effective_dictionary_size_in_pool"]
                    ),
                    "top_1_percent_firing_share_in_pool": float(
                        loss[prefix + "top_1_percent_firing_share_in_pool"]
                    ),
                    "top_5_percent_firing_share_in_pool": float(
                        loss[prefix + "top_5_percent_firing_share_in_pool"]
                    ),
                    "top_10_percent_firing_share_in_pool": float(
                        loss[prefix + "top_10_percent_firing_share_in_pool"]
                    ),
                    "readout_bias_mean": float(loss[prefix + "readout_bias_mean"]),
                    "readout_bias_negative_fraction": float(
                        loss[prefix + "readout_bias_negative_fraction"]
                    ),
                }
            )
    return output


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    runs = {
        "matryoshka": args.matryoshka_run,
        "full_only": args.full_only_run,
    }
    heldout = {
        "matryoshka": args.matryoshka_heldout,
        "full_only": args.full_only_heldout,
    }
    run_manifests = {arm: _json(path / "run_manifest.json") for arm, path in runs.items()}
    benchmarks = {arm: _json(path / "benchmark_report.json") for arm, path in runs.items()}
    reconstructions = {
        arm: _json(path / "heldout_reconstruction.json") for arm, path in heldout.items()
    }
    dictionaries = {arm: _json(path / "dictionary_health.json") for arm, path in heldout.items()}
    experiment_manifests = {
        arm: _json(path / "experiment_manifest.json") for arm, path in heldout.items()
    }
    metric_rows = {
        arm: _canonical_metrics(path / "metrics.jsonl") for arm, path in runs.items()
    }

    for key in ("git_sha", "dataset_manifest_sha256", "alignment_manifest_sha256", "model_order"):
        if run_manifests["matryoshka"][key] != run_manifests["full_only"][key]:
            raise ValueError(f"matched run provenance differs at {key}")
    if run_manifests["matryoshka"]["model_order"] != ["gemma", "smol"]:
        raise ValueError("comparison encountered a forbidden model topology")
    if set(run_manifests["matryoshka"]["model_revisions"]) != {"gemma", "smol"}:
        raise ValueError("comparison encountered a forbidden model identity")
    left_states = run_manifests["matryoshka"]["initial_common_state_by_rank"]
    right_states = run_manifests["full_only"]["initial_common_state_by_rank"]
    if [(row["model"], row["sha256"]) for row in left_states] != [
        (row["model"], row["sha256"]) for row in right_states
    ]:
        raise ValueError("common trainable initialization is not matched")
    batch_mismatch = [
        step
        for step, (left, right) in enumerate(
            zip(metric_rows["matryoshka"], metric_rows["full_only"], strict=True), start=1
        )
        if left["ordered_anchor_ids_sha256"] != right["ordered_anchor_ids_sha256"]
    ]
    if batch_mismatch:
        raise ValueError(f"ordered training batches differ at steps {batch_mismatch[:10]}")

    trajectory = _training_trajectory(metric_rows["matryoshka"], "matryoshka")
    trajectory += _training_trajectory(metric_rows["full_only"], "full_only")
    with (args.output_dir / "gate_trajectory.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)

    primary_dictionary: dict[str, Any] = {}
    for site in ("pre", "mid"):
        primary_dictionary[site] = {}
        for arm in ("matryoshka", "full_only"):
            value = dictionaries[arm]["gemma"][site]
            primary_dictionary[site][arm] = {
                key: value[key]
                for key in (
                    "effective_dictionary_size_entropy",
                    "used_features",
                    "never_fired_features",
                    "mean_active_features_per_anchor",
                    "top_1_percent_event_share",
                    "top_5_percent_event_share",
                    "top_10_percent_event_share",
                    "raw_logit_negative_fraction",
                    "positive_candidates_per_anchor_mean",
                    "training_used_features",
                    "training_recently_dead_features",
                )
            }

    baseline_metrics = reconstructions["matryoshka"]["metrics"]
    treatment_metrics = reconstructions["full_only"]["metrics"]
    paired = {
        "activation_normalized_full": _paired_document_difference(
            baseline_metrics,
            treatment_metrics,
            prefix="activation_normalized/",
            suffix="/prefix_5",
            seed=0,
        ),
        "activation_original_oracle_full": _paired_document_difference(
            baseline_metrics,
            treatment_metrics,
            prefix="activation_original_oracle/",
            suffix="/prefix_5",
            seed=1,
        ),
        "internal": _paired_document_difference(
            baseline_metrics,
            treatment_metrics,
            prefix="internal/",
            seed=2,
        ),
    }
    runtime = {
        arm: {
            key: benchmarks[arm][key]
            for key in (
                "step_seconds_p50",
                "step_seconds_p95",
                "accepted_anchors_per_second",
                "peak_gpu_memory_by_rank",
                "stage_seconds_mean",
                "checkpoint_seconds_save",
                "checkpoint_seconds_reload",
            )
        }
        for arm in ("matryoshka", "full_only")
    }
    result = {
        "schema_version": 1,
        "status": "MATCHED_COMPARISON_COMPLETE",
        "model_order": ["gemma", "smol"],
        "common_initialization": "PASS",
        "ordered_training_batch_hashes": "ALL_1221_MATCH",
        "git_sha": run_manifests["matryoshka"]["git_sha"],
        "dataset_manifest_sha256": run_manifests["matryoshka"][
            "dataset_manifest_sha256"
        ],
        "alignment_manifest_sha256": run_manifests["matryoshka"][
            "alignment_manifest_sha256"
        ],
        "config_fingerprints": {
            arm: run_manifests[arm]["config_fingerprint"] for arm in runs
        },
        "artifact_sha256": {
            arm: {
                "benchmark": sha256_file(runs[arm] / "benchmark_report.json"),
                "heldout_reconstruction": sha256_file(
                    heldout[arm] / "heldout_reconstruction.json"
                ),
                "dictionary_health": sha256_file(heldout[arm] / "dictionary_health.json"),
                "experiment_manifest": sha256_file(
                    heldout[arm] / "experiment_manifest.json"
                ),
            }
            for arm in runs
        },
        "primary_dictionary": primary_dictionary,
        "paired_document_bootstrap": paired,
        "runtime": runtime,
        "evaluation_manifests": experiment_manifests,
    }
    _write(args.output_dir / "matched_comparison.json", result)


if __name__ == "__main__":
    main()

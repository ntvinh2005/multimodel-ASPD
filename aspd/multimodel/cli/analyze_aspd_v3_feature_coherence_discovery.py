"""Freeze discovery clusters and centroids for feature-coherence rule discovery."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.cli.analyze_aspd_v3_interpretability_clusters import (
    _fit_feature,
    _normalize,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--contexts-dir", type=Path, required=True)
    parser.add_argument("--historical-embedding-input", type=Path, required=True)
    parser.add_argument("--historical-embeddings", type=Path, required=True)
    parser.add_argument("--historical-assignments", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--new-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _parse_bool(value: object) -> bool:
    return str(value).lower() == "true"


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceContract.load(args.protocol)
    context_manifest = json.loads(
        (args.contexts_dir / "preparation_manifest.json").read_text()
    )
    if (
        context_manifest["status"]
        != "ASPD_V3_1_FEATURE_COHERENCE_CONTEXTS_PREPARED"
        or context_manifest["protocol_fingerprint"] != contract.fingerprint()
        or not context_manifest["document_disjoint"]
    ):
        raise ValueError("context preparation is incomplete or incompatible")
    discovery_rows = _jsonl(args.contexts_dir / "discovery_contexts.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in discovery_rows:
        grouped[str(row["component_key"])].append(row)
    if len(grouped) != 160:
        raise ValueError("discovery contexts do not cover 160 components")

    historical_input = _jsonl(args.historical_embedding_input)
    historical_embeddings = np.load(
        args.historical_embeddings, allow_pickle=False
    ).astype(np.float32)
    if len(historical_input) != len(historical_embeddings):
        raise ValueError("historical embedding rows changed")
    historical_vectors = {
        (str(row["component_key"]), str(row["anchor_id"])): historical_embeddings[index]
        for index, row in enumerate(historical_input)
    }
    historical_labels: dict[tuple[str, str], int] = {}
    with args.historical_assignments.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            historical_labels[(row["component_key"], row["anchor_id"])] = int(
                row["cluster"]
            )
    historical_metrics = {}
    with args.historical_metrics.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            historical_metrics[row["component_key"]] = row

    new_input = _jsonl(args.contexts_dir / "discovery_new_embedding_input.jsonl")
    new_manifest = json.loads(
        (args.new_embedding_dir / "experiment_manifest.json").read_text()
    )
    if (
        new_manifest["status"] != "ASPD_V3_1_CONTEXT_EMBEDDINGS_COMPLETE"
        or new_manifest["input_sha256"]
        != sha256_file(args.contexts_dir / "discovery_new_embedding_input.jsonl")
    ):
        raise ValueError("new discovery embeddings are incomplete or mismatched")
    new_embeddings = np.load(
        args.new_embedding_dir / "embeddings.npy", allow_pickle=False
    ).astype(np.float32)
    if len(new_input) != len(new_embeddings):
        raise ValueError("new discovery embedding rows changed")
    new_vectors = {
        str(row["row_id"]): new_embeddings[index] for index, row in enumerate(new_input)
    }

    assignment_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    centroids: list[np.ndarray] = []
    centroid_index: list[dict[str, Any]] = []
    for component_index, key in enumerate(sorted(grouped)):
        rows = sorted(grouped[key], key=lambda row: str(row["row_id"]))
        cohort = str(rows[0]["cohort"])
        if cohort == "original_h5":
            values = np.stack(
                [historical_vectors[(key, str(row["anchor_id"]))] for row in rows]
            )
            labels = np.asarray(
                [historical_labels[(key, str(row["anchor_id"]))] for row in rows],
                dtype=np.int64,
            )
            old = historical_metrics[key]
            metrics: dict[str, Any] = {
                "selected_k": int(old["selected_k"]),
                "silhouette": float(old["silhouette"]),
                "minimum_cluster_fraction": float(old["minimum_cluster_fraction"]),
                "mean_stability_ari": float(old["mean_stability_ari"]),
                "credible_multi_cluster": _parse_bool(old["credible_multi_cluster"]),
                "source": "historical_h5_frozen_assignments",
            }
        else:
            values = np.stack([new_vectors[str(row["row_id"])] for row in rows])
            activations = np.asarray([float(row["activation"]) for row in rows])
            metrics, labels, _centers = _fit_feature(
                values,
                activations,
                candidate_k=[int(value) for value in contract.candidate_k],
                restarts=contract.clustering_restarts,
                seed=contract.component_sampling_seed + component_index * 10_000,
                minimum_silhouette=contract.minimum_silhouette,
                minimum_cluster_fraction=contract.minimum_cluster_fraction,
                minimum_stability=contract.minimum_stability_ari,
            )
            metrics["source"] = "new_random_frozen_discovery_fit"
        normalized = _normalize(values)
        groups = []
        for cluster in sorted(int(value) for value in np.unique(labels)):
            members = np.flatnonzero(labels == cluster)
            center = normalized[members].mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            centroid_id = len(centroids)
            centroids.append(center.astype(np.float32, copy=False))
            centroid_index.append(
                {
                    "centroid_id": centroid_id,
                    "component_key": key,
                    "cluster": cluster,
                    "discovery_size": int(len(members)),
                }
            )
            representative = members[
                np.argsort(-(normalized[members] @ center))[: min(5, len(members))]
            ]
            groups.append(
                {
                    "cluster": cluster,
                    "size": int(len(members)),
                    "representative_contexts": [
                        {
                            "row_id": rows[int(index)]["row_id"],
                            "document_id": rows[int(index)]["document_id"],
                            "activation": rows[int(index)]["activation"],
                            "source_group": rows[int(index)]["source_group"],
                            "text": rows[int(index)]["marked_text"],
                        }
                        for index in representative
                    ],
                }
            )
        for row, label in zip(rows, labels, strict=True):
            assignment_rows.append(
                {
                    "row_id": row["row_id"],
                    "component_key": key,
                    "cluster": int(label),
                }
            )
        component_rows.append(
            {
                "component_key": key,
                "site": rows[0]["site"],
                "feature": rows[0]["feature"],
                "cohort": cohort,
                "discovery_contexts": len(rows),
                "metrics": metrics,
                "groups": groups,
            }
        )

    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "discovery_cluster_assignments.jsonl", assignment_rows)
    _write_jsonl(args.output_dir / "component_clusters.jsonl", component_rows)
    _write_json(args.output_dir / "centroid_index.json", centroid_index)
    centroids_path = args.output_dir / "centroids.npy"
    with centroids_path.open("xb") as handle:
        np.save(handle, np.stack(centroids), allow_pickle=False)
    summary = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_COHERENCE_DISCOVERY_CLUSTERING_COMPLETE",
        "components": len(component_rows),
        "contexts": len(discovery_rows),
        "centroids": len(centroids),
        "credible_multi_cluster_components": sum(
            bool(row["metrics"]["credible_multi_cluster"]) for row in component_rows
        ),
        "by_cohort": {
            cohort: {
                "components": sum(row["cohort"] == cohort for row in component_rows),
                "credible_multi_cluster": sum(
                    row["cohort"] == cohort
                    and bool(row["metrics"]["credible_multi_cluster"])
                    for row in component_rows
                ),
                "selected_k": dict(
                    sorted(
                        Counter(
                            int(row["metrics"]["selected_k"])
                            for row in component_rows
                            if row["cohort"] == cohort
                        ).items()
                    )
                ),
            }
            for cohort in ("original_h5", "new_random")
        },
        "interpretation_boundary": (
            "Frozen Mistral clusters define groups for hypothesis generation and held-out "
            "cross-group tests; they do not by themselves establish polysemanticity."
        ),
    }
    _write_json(args.output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "git_sha": _git_sha(),
        "protocol_sha256": sha256_file(args.protocol),
        "contexts_manifest_sha256": sha256_file(
            args.contexts_dir / "preparation_manifest.json"
        ),
        "historical_embedding_input_sha256": sha256_file(
            args.historical_embedding_input
        ),
        "historical_embeddings_sha256": sha256_file(args.historical_embeddings),
        "historical_assignments_sha256": sha256_file(args.historical_assignments),
        "historical_metrics_sha256": sha256_file(args.historical_metrics),
        "new_embedding_manifest_sha256": sha256_file(
            args.new_embedding_dir / "experiment_manifest.json"
        ),
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "discovery_cluster_assignments.jsonl",
                "component_clusters.jsonl",
                "centroid_index.json",
                "centroids.npy",
                "summary.json",
            )
        },
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)
    (args.output_dir / "COMPLETE").write_text(
        "ASPD_V3_1_FEATURE_COHERENCE_DISCOVERY_CLUSTERING_COMPLETE\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Analyze preregistered exploratory context clusters for ASPD v3.1 features."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import kruskal, spearmanr

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("zero embedding vector")
    return values / norm


def cosine_kmeans(
    values: np.ndarray, k: int, *, seed: int, maximum_iterations: int = 100
) -> tuple[np.ndarray, np.ndarray, float]:
    values = _normalize(np.asarray(values, dtype=np.float32))
    if not 1 < k <= len(values):
        raise ValueError("invalid k")
    rng = np.random.default_rng(seed)
    centers = [values[int(rng.integers(len(values)))]]
    for _ in range(1, k):
        similarity = np.max(values @ np.stack(centers).T, axis=1)
        distance = np.maximum(1 - similarity, 0)
        total = float(distance.sum())
        index = int(rng.integers(len(values))) if total == 0 else int(rng.choice(len(values), p=distance / total))
        centers.append(values[index])
    centers_array = np.stack(centers)
    labels = np.full(len(values), -1, dtype=np.int64)
    for _ in range(maximum_iterations):
        new_labels = np.argmax(values @ centers_array.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = values[labels == cluster]
            if not len(members):
                farthest = int(np.argmin(np.max(values @ centers_array.T, axis=1)))
                center = values[farthest]
            else:
                center = members.mean(axis=0)
                center /= max(float(np.linalg.norm(center)), 1e-12)
            centers_array[cluster] = center
    objective = float(np.sum(1 - np.sum(values * centers_array[labels], axis=1)))
    return labels, centers_array, objective


def cosine_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    values = _normalize(np.asarray(values, dtype=np.float32))
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    distance = 1 - values @ values.T
    scores = []
    for index, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        own = own[own != index]
        a = float(distance[index, own].mean()) if len(own) else 0.0
        b = min(float(distance[index, labels == other].mean()) for other in unique if other != label)
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("ARI label shapes differ")
    table = Counter(zip(left.tolist(), right.tolist(), strict=True))
    rows = Counter(left.tolist())
    columns = Counter(right.tolist())

    def choose2(value: int) -> float:
        return value * (value - 1) / 2

    total_pairs = choose2(len(left))
    if total_pairs == 0:
        return 1.0
    observed = sum(choose2(value) for value in table.values())
    row_pairs = sum(choose2(value) for value in rows.values())
    column_pairs = sum(choose2(value) for value in columns.values())
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    return float((observed - expected) / (maximum - expected)) if maximum != expected else 1.0


def _fit_feature(
    values: np.ndarray,
    activations: np.ndarray,
    *,
    candidate_k: list[int],
    restarts: int,
    seed: int,
    minimum_silhouette: float,
    minimum_cluster_fraction: float,
    minimum_stability: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    candidates = []
    all_labels: dict[int, list[np.ndarray]] = {}
    all_centers: dict[int, list[np.ndarray]] = {}
    for k in candidate_k:
        runs = [cosine_kmeans(values, k, seed=seed + k * 1000 + restart) for restart in range(restarts)]
        all_labels[k] = [run[0] for run in runs]
        all_centers[k] = [run[1] for run in runs]
        best_index = int(np.argmin([run[2] for run in runs]))
        labels, centers, objective = runs[best_index]
        silhouette = cosine_silhouette(values, labels)
        fractions = np.bincount(labels, minlength=k) / len(labels)
        stability_values = [
            adjusted_rand(labels, other)
            for index, other in enumerate(all_labels[k])
            if index != best_index
        ]
        candidates.append(
            {
                "k": k,
                "labels": labels,
                "centers": centers,
                "objective": objective,
                "silhouette": silhouette,
                "minimum_cluster_fraction": float(fractions.min()),
                "mean_stability_ari": float(np.mean(stability_values)),
            }
        )
    feasible = [row for row in candidates if row["minimum_cluster_fraction"] >= minimum_cluster_fraction]
    chosen = max(feasible, key=lambda row: (row["silhouette"], -row["k"])) if feasible else None
    if chosen is None or chosen["silhouette"] < minimum_silhouette:
        labels = np.zeros(len(values), dtype=np.int64)
        centers = _normalize(values.mean(axis=0, keepdims=True))
        chosen_k = 1
        silhouette = 0.0
        stability = 1.0
        minimum_fraction = 1.0
    else:
        labels = chosen["labels"]
        centers = chosen["centers"]
        chosen_k = int(chosen["k"])
        silhouette = float(chosen["silhouette"])
        stability = float(chosen["mean_stability_ari"])
        minimum_fraction = float(chosen["minimum_cluster_fraction"])
    similarity = _normalize(values) @ _normalize(values).T
    upper = np.triu_indices(len(values), 1)
    same = labels[upper[0]] == labels[upper[1]]
    within = float(similarity[upper][same].mean()) if np.any(same) else 0.0
    across = float(similarity[upper][~same].mean()) if np.any(~same) else 0.0
    groups = [activations[labels == cluster] for cluster in range(chosen_k)]
    activation_p = float(kruskal(*groups).pvalue) if chosen_k > 1 and all(len(group) for group in groups) else 1.0
    credible = bool(
        chosen_k > 1
        and silhouette >= minimum_silhouette
        and minimum_fraction >= minimum_cluster_fraction
        and stability >= minimum_stability
    )
    metrics = {
        "selected_k": chosen_k,
        "silhouette": silhouette,
        "minimum_cluster_fraction": minimum_fraction,
        "mean_stability_ari": stability,
        "within_cluster_mean_cosine": within,
        "across_cluster_mean_cosine": across,
        "within_minus_across_cosine": within - across,
        "activation_kruskal_pvalue_exploratory": activation_p,
        "credible_multi_cluster": credible,
        "candidate_diagnostics": [
            {key: value for key, value in row.items() if key not in {"labels", "centers"}}
            for row in candidates
        ],
    }
    return metrics, labels, centers


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    embedding_manifest = json.loads(
        (args.embedding_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if embedding_manifest.get("status") != "ASPD_V3_1_CONTEXT_EMBEDDINGS_COMPLETE":
        raise ValueError("embedding job is incomplete")
    if embedding_manifest.get("tokenizer_fix_mistral_regex") is not True:
        raise ValueError("embedding artifact lacks Mistral tokenizer fix")
    if embedding_manifest.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("embedding and clustering protocols differ")
    rows = []
    with (args.preparation_dir / "clustering_embedding_input.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    embeddings = np.load(args.embedding_dir / "embeddings.npy", allow_pickle=False).astype(np.float32)
    if len(rows) != len(embeddings):
        raise ValueError("embedding row count differs from input")
    observations = {
        row["component_key"]: row
        for row in csv.DictReader((args.preparation_dir / "component_observations.csv").open())
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["component_key"])].append(index)
    metrics_rows = []
    assignment_rows = []
    gallery = []
    for feature_index, key in enumerate(sorted(grouped)):
        indices = np.asarray(grouped[key], dtype=np.int64)
        feature_rows = [rows[index] for index in indices]
        values = embeddings[indices]
        activations = np.asarray([float(row["activation"]) for row in feature_rows])
        metrics, labels, centers = _fit_feature(
            values,
            activations,
            candidate_k=[int(value) for value in protocol["clustering_candidate_k"]],
            restarts=int(protocol["clustering_random_restarts"]),
            seed=int(protocol["clustering_seed"]) + feature_index * 10_000,
            minimum_silhouette=float(protocol["clustering_minimum_silhouette"]),
            minimum_cluster_fraction=float(protocol["clustering_minimum_cluster_fraction"]),
            minimum_stability=float(protocol["clustering_minimum_stability_ari"]),
        )
        observed = observations[key]
        summary = {
            "component_key": key,
            "site": feature_rows[0]["site"],
            "feature": int(feature_rows[0]["feature"]),
            "frequency_stratum": feature_rows[0]["frequency_stratum"],
            "firing_count": int(feature_rows[0]["firing_count"]),
            "contexts": len(feature_rows),
            "failed_judge_accuracy": float(observed["accuracy"]),
            **{field: value for field, value in metrics.items() if field != "candidate_diagnostics"},
        }
        metrics_rows.append(summary)
        normalized = _normalize(values)
        cluster_gallery = []
        for cluster in range(int(metrics["selected_k"])):
            members = np.flatnonzero(labels == cluster)
            representative = members[np.argsort(-(normalized[members] @ centers[cluster]))[:3]]
            cluster_gallery.append(
                {
                    "cluster": cluster,
                    "size": int(len(members)),
                    "representative_contexts": [
                        {
                            "text": feature_rows[int(index)]["text"],
                            "activation": float(feature_rows[int(index)]["activation"]),
                            "source_group": feature_rows[int(index)]["source_group"],
                        }
                        for index in representative
                    ],
                }
            )
        gallery.append({**summary, "clusters": cluster_gallery, "candidate_diagnostics": metrics["candidate_diagnostics"]})
        for row, label in zip(feature_rows, labels, strict=True):
            assignment_rows.append(
                {
                    "component_key": key,
                    "context_index": int(row["context_index"]),
                    "anchor_id": row["anchor_id"],
                    "activation": float(row["activation"]),
                    "cluster": int(label),
                }
            )
    args.output_dir.mkdir(parents=True)
    for name, output_rows in (
        ("feature_cluster_metrics.csv", metrics_rows),
        ("cluster_assignments.csv", assignment_rows),
    ):
        with (args.output_dir / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    _write_json(args.output_dir / "cluster_gallery.json", gallery)
    credible = np.asarray([bool(row["credible_multi_cluster"]) for row in metrics_rows])
    judge = np.asarray([float(row["failed_judge_accuracy"]) for row in metrics_rows])
    silhouette = np.asarray([float(row["silhouette"]) for row in metrics_rows])
    selected_k = np.asarray([int(row["selected_k"]) for row in metrics_rows])
    correlation = spearmanr(silhouette, judge)
    _write_json(
        args.output_dir / "cluster_summary.json",
        {
            "schema_version": 1,
            "status": "EXPLORATORY_CLUSTER_ANALYSIS_COMPLETE",
            "features": len(metrics_rows),
            "contexts": len(rows),
            "credible_multi_cluster_features": int(credible.sum()),
            "credible_multi_cluster_fraction": float(credible.mean()),
            "selected_k_counts": {str(key): int(value) for key, value in Counter(selected_k.tolist()).items()},
            "by_site": {
                site: {
                    "features": sum(row["site"] == site for row in metrics_rows),
                    "credible_fraction": float(
                        np.mean([row["credible_multi_cluster"] for row in metrics_rows if row["site"] == site])
                    ),
                }
                for site in ("pre", "mid")
            },
            "spearman_silhouette_vs_failed_judge_accuracy": {
                "rho": float(correlation.statistic),
                "pvalue_exploratory": float(correlation.pvalue),
            },
            "interpretation_boundary": (
                "Mistral embedding clusters are exploratory. They do not establish distinct human-"
                "interpretable meanings without blinded human confirmation or held-out within-cluster tests."
            ),
        },
    )
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "ASPD_V3_1_INTERPRETABILITY_CLUSTERING_COMPLETE",
            "git_sha": _git_sha(),
            "protocol_sha256": sha256_file(args.protocol),
            "preparation_manifest_sha256": sha256_file(args.preparation_dir / "preparation_manifest.json"),
            "embedding_manifest_sha256": sha256_file(args.embedding_dir / "experiment_manifest.json"),
            "embedding_sha256": sha256_file(args.embedding_dir / "embeddings.npy"),
        },
    )
    (args.output_dir / "COMPLETE").write_text("ASPD_V3_1_INTERPRETABILITY_CLUSTERING_COMPLETE\n")


if __name__ == "__main__":
    main()

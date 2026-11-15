"""Assign held-out contexts to frozen groups and construct matched controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contexts-dir", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--cluster-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
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


def _priority(seed: int, *values: object) -> str:
    return hashlib.sha256(
        "|".join([str(seed), *(str(value) for value in values)]).encode()
    ).hexdigest()


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    strata: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Deterministic round-robin sample across nonempty strata."""

    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in strata)].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: _priority(seed, row["row_id"]))
    keys = sorted(grouped, key=lambda key: tuple(str(value) for value in key))
    output: list[dict[str, Any]] = []
    cursor = 0
    while len(output) < min(count, len(rows)) and keys:
        key = keys[cursor % len(keys)]
        if grouped[key]:
            output.append(grouped[key].pop(0))
        if not grouped[key]:
            keys.remove(key)
            cursor = 0
        else:
            cursor += 1
    return output


def _control_cost(
    row: dict[str, Any],
    *,
    positive_id: str,
    positive_length: int,
    positive_vector: np.ndarray,
    vectors: dict[str, np.ndarray],
    seed: int,
    semantic: bool,
) -> tuple[float, str]:
    length_gap = abs(
        math.log((len(str(row["text"])) + 1) / (positive_length + 1))
    )
    distance = (
        1.0 - float(positive_vector @ vectors[str(row["row_id"])] )
        if semantic
        else 0.0
    )
    return distance + (0.15 if semantic else 1.0) * length_gap, _priority(
        seed, positive_id, row["row_id"]
    )


def _match_controls(
    positives: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    *,
    seed: int,
    semantic: bool,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    available = {str(row["row_id"]): row for row in candidates}
    pairs = []
    for positive in sorted(positives, key=lambda row: _priority(seed, row["row_id"])):
        if not available:
            break
        same_source = [
            row
            for row in available.values()
            if row["source_group"] == positive["source_group"]
        ]
        pool = same_source or list(available.values())
        positive_id = str(positive["row_id"])
        positive_vector = vectors[positive_id]
        chosen = min(
            pool,
            key=lambda row: _control_cost(
                row,
                positive_id=positive_id,
                positive_length=len(str(positive["text"])),
                positive_vector=positive_vector,
                vectors=vectors,
                seed=seed,
                semantic=semantic,
            ),
        )
        cosine = float(
            vectors[str(positive["row_id"])] @ vectors[str(chosen["row_id"])]
        )
        diagnostics = {
            "source_group_matched": chosen["source_group"] == positive["source_group"],
            "embedding_cosine": cosine,
            "absolute_character_length_gap": abs(
                len(str(chosen["text"])) - len(str(positive["text"]))
            ),
            "matching_mode": "semantic" if semantic else "random_length",
        }
        pairs.append((positive, chosen, diagnostics))
        del available[str(chosen["row_id"])]
    return pairs


def _activation_bin(value: float, low: float, high: float) -> str:
    if value <= low:
        return "weak"
    if value <= high:
        return "medium"
    return "strong"


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    source_checks = {
        args.contexts_dir / "preparation_manifest.json": contract.source_context_manifest_sha256,
        args.extraction_dir
        / "discovery-new/experiment_manifest.json": contract.source_discovery_extraction_manifest_sha256,
        args.extraction_dir
        / "confirmatory/experiment_manifest.json": contract.source_confirmatory_extraction_manifest_sha256,
    }
    for path, expected in source_checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen analysis source changed: {path}")

    rows = _jsonl(args.contexts_dir / "confirmatory_embedding_input.jsonl")
    embedding_manifest = json.loads(
        (args.embedding_dir / "experiment_manifest.json").read_text()
    )
    if (
        embedding_manifest["status"] != "ASPD_V3_1_CONTEXT_EMBEDDINGS_COMPLETE"
        or embedding_manifest["input_sha256"]
        != sha256_file(args.contexts_dir / "confirmatory_embedding_input.jsonl")
        or embedding_manifest["embedding_model_revision"]
        != contract.embedding_model_revision
    ):
        raise ValueError("confirmatory embeddings are incomplete or incompatible")
    embeddings = np.load(args.embedding_dir / "embeddings.npy", allow_pickle=False).astype(
        np.float32
    )
    if len(rows) != len(embeddings):
        raise ValueError("confirmatory embedding row count changed")
    vectors = {
        str(row["row_id"]): embeddings[index] for index, row in enumerate(rows)
    }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["component_key"])][str(row["context_type"])].append(row)

    cluster_manifest = json.loads(
        (args.cluster_dir / "experiment_manifest.json").read_text()
    )
    if cluster_manifest["status"] != (
        "ASPD_V3_1_FEATURE_COHERENCE_DISCOVERY_CLUSTERING_COMPLETE"
    ):
        raise ValueError("discovery clusters are incomplete")
    centroid_rows = json.loads((args.cluster_dir / "centroid_index.json").read_text())
    centroids = np.load(args.cluster_dir / "centroids.npy", allow_pickle=False).astype(
        np.float32
    )
    centroid_by_component: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in centroid_rows:
        centroid_by_component[str(row["component_key"])].append(
            (int(row["cluster"]), int(row["centroid_id"]))
        )

    threshold_rows = json.loads(
        (args.preparation_dir / "activation_thresholds.json").read_text()
    )
    thresholds = {str(row["component_key"]): row for row in threshold_rows}
    selected_components = json.loads(
        (args.preparation_dir / "selected_components.json").read_text()
    )
    selected_index = {str(row["component_key"]): row for row in selected_components}
    positive_assignments: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []
    control_sets: list[dict[str, Any]] = []
    coverage_rows = []
    sampled_positives_by_component: dict[str, list[dict[str, Any]]] = {}

    for key in sorted(grouped):
        positives = grouped[key]["natural_firing"]
        component_centroids = centroid_by_component[key]
        center_values = np.stack([centroids[index] for _, index in component_centroids])
        threshold = thresholds[key]
        for row in positives:
            similarity = vectors[str(row["row_id"])] @ center_values.T
            order = np.argsort(-similarity)
            row["assigned_discovery_cluster"] = component_centroids[int(order[0])][0]
            row["cluster_assignment_cosine"] = float(similarity[order[0]])
            row["cluster_assignment_margin"] = (
                float(similarity[order[0]] - similarity[order[1]])
                if len(order) > 1
                else None
            )
            row["activation_bin"] = _activation_bin(
                float(row["activation"]),
                float(threshold["q1_over_3"]),
                float(threshold["q2_over_3"]),
            )
            positive_assignments.append(
                {
                    field: row[field]
                    for field in (
                        "row_id",
                        "component_key",
                        "document_id",
                        "anchor_id",
                        "activation",
                        "activation_bin",
                        "assigned_discovery_cluster",
                        "cluster_assignment_cosine",
                        "cluster_assignment_margin",
                    )
                }
            )
        sampled = stratified_sample(
            positives,
            count=contract.heldout_positive_examples_per_component,
            seed=contract.heldout_sampling_seed,
            strata=("assigned_discovery_cluster", "activation_bin"),
        )
        sampled_positives_by_component[key] = sampled
        ordinary_pairs = _match_controls(
            sampled,
            grouped[key]["ordinary_nonfiring"],
            vectors,
            seed=contract.heldout_sampling_seed + 1,
            semantic=True,
        )
        hard_pairs = _match_controls(
            sampled,
            grouped[key]["positive_unselected_hard_negative"],
            vectors,
            seed=contract.heldout_sampling_seed + 2,
            semantic=True,
        )
        random_pairs = _match_controls(
            sampled,
            grouped[key]["ordinary_nonfiring"],
            vectors,
            seed=contract.heldout_sampling_seed + 3,
            semantic=False,
        )
        if (
            len(sampled) < contract.minimum_heldout_firing_contexts
            or len(ordinary_pairs) < contract.minimum_heldout_nonfiring_contexts
        ):
            raise ValueError(f"held-out sample is below the frozen minimum for {key}")
        pair_index = {
            "ordinary": ordinary_pairs,
            "hard": hard_pairs,
        }
        for condition, pairs in pair_index.items():
            for pair_number, (positive, negative, diagnostics) in enumerate(pairs):
                pair_id = hashlib.sha256(
                    f"{key}|{condition}|{pair_number}|{positive['row_id']}|{negative['row_id']}".encode()
                ).hexdigest()
                if condition == "ordinary":
                    validation_examples.append(
                        {
                            **positive,
                            "example_id": f"{pair_id}:positive",
                            "pair_id": pair_id,
                            "condition": condition,
                            "actual_natural_firing": True,
                            "target_discovery_cluster": positive[
                                "assigned_discovery_cluster"
                            ],
                        }
                    )
                validation_examples.append(
                    {
                        **negative,
                        "example_id": f"{pair_id}:negative",
                        "pair_id": pair_id,
                        "condition": condition,
                        "actual_natural_firing": False,
                        "target_discovery_cluster": positive[
                            "assigned_discovery_cluster"
                        ],
                        "paired_positive_row_id": positive["row_id"],
                        "match_diagnostics": diagnostics,
                    }
                )
        control_sets.extend(
            {
                "component_key": key,
                "control": control,
                "row_ids": [negative["row_id"] for _, negative, _ in pairs],
            }
            for control, pairs in (
                ("random_source_length_matched_nonfiring", random_pairs),
                ("topic_matched_nonfiring", ordinary_pairs),
            )
        )
        cluster_counts = Counter(
            int(row["assigned_discovery_cluster"]) for row in sampled
        )
        coverage_rows.append(
            {
                "component_key": key,
                "site": selected_index[key]["site"],
                "cohort": selected_index[key]["cohort"],
                "historical_credible_multi_cluster": selected_index[key][
                    "historical_credible_multi_cluster"
                ],
                "available_positive": len(positives),
                "sampled_positive": len(sampled),
                "matched_ordinary": len(ordinary_pairs),
                "matched_hard": len(hard_pairs),
                "sampled_weak": sum(row["activation_bin"] == "weak" for row in sampled),
                "sampled_medium": sum(
                    row["activation_bin"] == "medium" for row in sampled
                ),
                "sampled_strong": sum(
                    row["activation_bin"] == "strong" for row in sampled
                ),
                "groups_with_at_least_8": sum(
                    value >= contract.minimum_heldout_contexts_per_group
                    for value in cluster_counts.values()
                ),
                "total_discovery_groups": len(component_centroids),
            }
        )

    # A same-site component with closest discovery firing frequency is the frozen
    # frequency-matched context control for each target component.
    for key in sorted(sampled_positives_by_component):
        target = selected_index[key]
        candidates = [
            other
            for other in selected_index
            if other != key and selected_index[other]["site"] == target["site"]
        ]
        donor = min(
            candidates,
            key=lambda other: (
                abs(
                    int(selected_index[other]["discovery_natural_firing_events"])
                    - int(target["discovery_natural_firing_events"])
                ),
                _priority(contract.heldout_sampling_seed, key, other),
            ),
        )
        donor_rows = sampled_positives_by_component[donor]
        control_sets.append(
            {
                "component_key": key,
                "control": "frequency_matched_other_component",
                "donor_component_key": donor,
                "row_ids": [row["row_id"] for row in donor_rows],
            }
        )

    validation_examples.sort(
        key=lambda row: (row["component_key"], row["condition"], row["example_id"])
    )
    positive_assignments.sort(key=lambda row: (row["component_key"], row["row_id"]))
    args.output_dir.mkdir(parents=True)
    _write_jsonl(
        args.output_dir / "heldout_validation_examples.jsonl", validation_examples
    )
    _write_jsonl(
        args.output_dir / "heldout_positive_assignments.jsonl", positive_assignments
    )
    _write_jsonl(args.output_dir / "cluster_control_sets.jsonl", control_sets)
    with (args.output_dir / "component_coverage.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    summary = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_COHERENCE_HELDOUT_PREPARED",
        "components": len(coverage_rows),
        "validation_examples": len(validation_examples),
        "positive_examples": sum(
            row["actual_natural_firing"] for row in validation_examples
        ),
        "ordinary_negative_examples": sum(
            not row["actual_natural_firing"] and row["condition"] == "ordinary"
            for row in validation_examples
        ),
        "hard_negative_examples": sum(
            not row["actual_natural_firing"] and row["condition"] == "hard"
            for row in validation_examples
        ),
        "components_with_all_groups_at_least_8": sum(
            row["groups_with_at_least_8"] == row["total_discovery_groups"]
            for row in coverage_rows
        ),
        "matching": contract.negative_matching,
        "document_disjoint": True,
    }
    _write_json(args.output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "git_sha": _git_sha(),
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "contexts_manifest_sha256": sha256_file(
            args.contexts_dir / "preparation_manifest.json"
        ),
        "cluster_manifest_sha256": sha256_file(
            args.cluster_dir / "experiment_manifest.json"
        ),
        "embedding_manifest_sha256": sha256_file(
            args.embedding_dir / "experiment_manifest.json"
        ),
        "embedding_sha256": sha256_file(args.embedding_dir / "embeddings.npy"),
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "heldout_validation_examples.jsonl",
                "heldout_positive_assignments.jsonl",
                "cluster_control_sets.jsonl",
                "component_coverage.csv",
                "summary.json",
            )
        },
    }
    _write_json(args.output_dir / "preparation_manifest.json", manifest)
    (args.output_dir / "PREPARED").write_text(
        "ASPD_V3_1_FEATURE_COHERENCE_HELDOUT_PREPARED\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

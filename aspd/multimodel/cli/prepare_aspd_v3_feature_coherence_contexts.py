"""Freeze deduplicated discovery and confirmatory context inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import (
    FeatureCoherenceContract,
    deduplicate_context_rows,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--old-assignments", type=Path, required=True)
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


def _row_id(component_key: str, context_type: str, anchor_id: str) -> str:
    return hashlib.sha256(
        f"{component_key}|{context_type}|{anchor_id}".encode()
    ).hexdigest()


def _validate_extraction(
    directory: Path,
    *,
    mode: str,
    expected_rows: int,
    protocol_fingerprint: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete = json.loads((directory / "COMPLETE").read_text())
    manifest_path = directory / "experiment_manifest.json"
    if complete["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError(f"{mode} extraction manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["status"] != "ASPD_V3_1_FEATURE_COHERENCE_EXTRACTION_COMPLETE"
        or manifest["mode"] != mode
        or manifest["protocol_fingerprint"] != protocol_fingerprint
        or not manifest["excluded_documents_disjoint"]
    ):
        raise ValueError(f"{mode} extraction invariants failed")
    rows = _jsonl(directory / "component_contexts.jsonl")
    if len(rows) != expected_rows:
        raise ValueError(f"{mode} extraction component count changed")
    return manifest, rows


def _deduplicate(
    rows: list[dict[str, Any]], contract: FeatureCoherenceContract
) -> list[dict[str, Any]]:
    return [
        dict(value)
        for value in deduplicate_context_rows(
            rows,
            threshold=contract.normalized_text_near_duplicate_jaccard,
            text_field="text",
        )
    ]


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceContract.load(args.protocol)
    selected = json.loads((args.preparation_dir / "selected_components.json").read_text())
    selected_index = {str(row["component_key"]): row for row in selected}
    if len(selected_index) != 160:
        raise ValueError("prepared component sample changed")
    discovery_manifest, discovery_new = _validate_extraction(
        args.extraction_dir / "discovery-new",
        mode="discovery-new",
        expected_rows=40,
        protocol_fingerprint=contract.fingerprint(),
    )
    confirmatory_manifest, confirmatory = _validate_extraction(
        args.extraction_dir / "confirmatory",
        mode="confirmatory",
        expected_rows=160,
        protocol_fingerprint=contract.fingerprint(),
    )

    assignment_index: dict[tuple[str, int], int] = {}
    with args.old_assignments.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assignment_index[(row["component_key"], int(row["context_index"]))] = int(
                row["cluster"]
            )
    old_rows = _jsonl(Path(contract.original_embedding_input))
    if len(old_rows) != 10_311:
        raise ValueError("historical H5 discovery context count changed")
    if any((row["component_key"], int(row["context_index"])) not in assignment_index for row in old_rows):
        raise ValueError("historical cluster assignments are incomplete")

    discovery_contexts: list[dict[str, Any]] = []
    discovery_counts_before: Counter[str] = Counter()
    discovery_counts_after: Counter[str] = Counter()
    old_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        key = str(row["component_key"])
        discovery_counts_before[key] += 1
        old_grouped[key].append(
            {
                "row_id": _row_id(key, "discovery_positive", str(row["anchor_id"])),
                "component_key": key,
                "site": row["site"],
                "feature": int(row["feature"]),
                "cohort": "original_h5",
                "context_type": "natural_firing",
                "document_id": row["document_id"],
                "anchor_id": row["anchor_id"],
                "source_group": row["source_group"],
                "activation": float(row["activation"]),
                "text": row["text"],
                "marked_text": row["text"],
                "discovery_cluster": assignment_index[
                    (key, int(row["context_index"]))
                ],
                "cluster_source": "historical_h5_mistral_cosine_kmeans",
            }
        )
    for key in sorted(old_grouped):
        ordered = sorted(
            old_grouped[key], key=lambda row: (-float(row["activation"]), row["anchor_id"])
        )
        kept = _deduplicate(ordered, contract)
        discovery_counts_after[key] += len(kept)
        discovery_contexts.extend(kept)

    for component in discovery_new:
        key = str(component["component_key"])
        source = []
        for row in component["positive_contexts"]:
            source.append(
                {
                    "row_id": _row_id(key, "discovery_positive", str(row["anchor_id"])),
                    "component_key": key,
                    "site": component["site"],
                    "feature": int(component["feature"]),
                    "cohort": "new_random",
                    "context_type": "natural_firing",
                    "document_id": row["document_id"],
                    "anchor_id": row["anchor_id"],
                    "source_group": row["source_group"],
                    "activation": float(row["activation"]),
                    "text": row["canonical_context"],
                    "marked_text": row["marked_canonical_context"],
                    "discovery_cluster": None,
                    "cluster_source": None,
                }
            )
        discovery_counts_before[key] += len(source)
        kept = _deduplicate(
            sorted(source, key=lambda row: (-float(row["activation"]), row["anchor_id"])),
            contract,
        )
        discovery_counts_after[key] += len(kept)
        discovery_contexts.extend(kept)

    confirmatory_contexts: list[dict[str, Any]] = []
    confirm_counts_before: Counter[tuple[str, str]] = Counter()
    confirm_counts_after: Counter[tuple[str, str]] = Counter()
    field_types = (
        ("positive_contexts", "natural_firing"),
        ("ordinary_negative_candidates", "ordinary_nonfiring"),
        ("hard_negative_candidates", "positive_unselected_hard_negative"),
    )
    for component in confirmatory:
        key = str(component["component_key"])
        for field, context_type in field_types:
            source = []
            for row in component[field]:
                source.append(
                    {
                        "row_id": _row_id(key, context_type, str(row["anchor_id"])),
                        "component_key": key,
                        "site": component["site"],
                        "feature": int(component["feature"]),
                        "cohort": component["cohort"],
                        "context_type": context_type,
                        "document_id": row["document_id"],
                        "anchor_id": row["anchor_id"],
                        "source_group": row["source_group"],
                        "activation": float(row["activation"]),
                        "pre_topk_score": float(row["pre_topk_score"]),
                        "text": row["canonical_context"],
                        "marked_text": row["marked_canonical_context"],
                    }
                )
            confirm_counts_before[(key, context_type)] += len(source)
            ordering = (
                (lambda row: (-float(row["activation"]), row["anchor_id"]))
                if context_type == "natural_firing"
                else (lambda row: row["row_id"])
            )
            kept = _deduplicate(sorted(source, key=ordering), contract)
            confirm_counts_after[(key, context_type)] += len(kept)
            confirmatory_contexts.extend(kept)

    discovery_contexts.sort(key=lambda row: (row["component_key"], row["row_id"]))
    confirmatory_contexts.sort(
        key=lambda row: (row["component_key"], row["context_type"], row["row_id"])
    )
    if {row["document_id"] for row in discovery_contexts} & {
        row["document_id"] for row in confirmatory_contexts
    }:
        raise ValueError("discovery document leaked into deduplicated confirmatory contexts")
    if len({row["row_id"] for row in discovery_contexts + confirmatory_contexts}) != (
        len(discovery_contexts) + len(confirmatory_contexts)
    ):
        raise ValueError("context row IDs are not globally unique")

    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "discovery_contexts.jsonl", discovery_contexts)
    _write_jsonl(
        args.output_dir / "discovery_new_embedding_input.jsonl",
        [row for row in discovery_contexts if row["cohort"] == "new_random"],
    )
    _write_jsonl(
        args.output_dir / "confirmatory_embedding_input.jsonl", confirmatory_contexts
    )
    coverage_rows = []
    for key in sorted(selected_index):
        row = selected_index[key]
        coverage_rows.append(
            {
                "component_key": key,
                "site": row["site"],
                "cohort": row["cohort"],
                "historical_credible_multi_cluster": row[
                    "historical_credible_multi_cluster"
                ],
                "discovery_before_dedup": discovery_counts_before[key],
                "discovery_after_dedup": discovery_counts_after[key],
                **{
                    f"{context_type}_before_dedup": confirm_counts_before[
                        (key, context_type)
                    ]
                    for _, context_type in field_types
                },
                **{
                    f"{context_type}_after_dedup": confirm_counts_after[
                        (key, context_type)
                    ]
                    for _, context_type in field_types
                },
            }
        )
    with (args.output_dir / "component_coverage.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_COHERENCE_CONTEXTS_PREPARED",
        "git_sha": _git_sha(),
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_fingerprint": contract.fingerprint(),
        "preparation_manifest_sha256": sha256_file(
            args.preparation_dir / "preparation_manifest.json"
        ),
        "discovery_extraction_manifest_sha256": sha256_file(
            args.extraction_dir / "discovery-new" / "experiment_manifest.json"
        ),
        "confirmatory_extraction_manifest_sha256": sha256_file(
            args.extraction_dir / "confirmatory" / "experiment_manifest.json"
        ),
        "historical_assignments_sha256": sha256_file(args.old_assignments),
        "discovery_rows_before_dedup": sum(discovery_counts_before.values()),
        "discovery_rows_after_dedup": len(discovery_contexts),
        "confirmatory_rows_before_dedup": sum(confirm_counts_before.values()),
        "confirmatory_rows_after_dedup": len(confirmatory_contexts),
        "discovery_new_embedding_rows": sum(
            row["cohort"] == "new_random" for row in discovery_contexts
        ),
        "confirmatory_embedding_rows": len(confirmatory_contexts),
        "document_disjoint": True,
        "near_duplicate_jaccard_threshold": (
            contract.normalized_text_near_duplicate_jaccard
        ),
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "discovery_contexts.jsonl",
                "discovery_new_embedding_input.jsonl",
                "confirmatory_embedding_input.jsonl",
                "component_coverage.csv",
            )
        },
        "source_extraction_elapsed_seconds": {
            "discovery_new": discovery_manifest["elapsed_seconds"],
            "confirmatory": confirmatory_manifest["elapsed_seconds"],
        },
    }
    _write_json(args.output_dir / "preparation_manifest.json", manifest)
    (args.output_dir / "PREPARED").write_text(
        "ASPD_V3_1_FEATURE_COHERENCE_CONTEXTS_PREPARED\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

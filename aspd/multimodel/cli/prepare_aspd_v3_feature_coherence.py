"""Freeze the ASPD v3.1 feature-coherence component and document manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.feature_coherence import (
    FeatureCoherenceContract,
    component_key,
    csr_unique_document_counts,
    feature_event_values,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_sha(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _load_csr(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        result = {key: data[key] for key in data.files}
    shape = tuple(int(value) for value in result["shape"])
    if shape != (100_352, 8_192):
        raise ValueError(f"unexpected sparse firing shape in {path}: {shape}")
    return result


def _site_seed(base_seed: int, site: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{site}".encode()).digest()[:8], "big")


def prepare(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = FeatureCoherenceContract.load(protocol_path)
    sparse_dir = Path(contract.sparse_discovery_dir)
    paths = {
        "original_embedding_input": Path(contract.original_embedding_input),
        "original_sample_manifest": Path(contract.original_sample_manifest),
        "original_cluster_metrics": Path(contract.original_cluster_metrics),
        "sparse_pre": sparse_dir / "pre_natural_firings.npz",
        "sparse_mid": sparse_dir / "mid_natural_firings.npz",
        "sparse_anchors": sparse_dir / "anchors.npz",
        "sparse_documents": sparse_dir / "documents.json",
    }
    expected = {
        "original_embedding_input": contract.original_embedding_input_sha256,
        "original_sample_manifest": contract.original_sample_manifest_sha256,
        "original_cluster_metrics": contract.original_cluster_metrics_sha256,
        "sparse_pre": contract.sparse_pre_sha256,
        "sparse_mid": contract.sparse_mid_sha256,
        "sparse_anchors": contract.sparse_anchors_sha256,
        "sparse_documents": contract.sparse_documents_sha256,
    }
    for key, path in paths.items():
        _validate_sha(path, expected[key])

    sample_manifest = json.loads(paths["original_sample_manifest"].read_text())
    original_rows = sample_manifest["selected"]
    if len(original_rows) != contract.original_components:
        raise ValueError("original H5 component count changed")
    original_keys = {str(row["component_key"]) for row in original_rows}
    if len(original_keys) != contract.original_components:
        raise ValueError("original H5 component keys are not unique")

    with paths["original_cluster_metrics"].open(encoding="utf-8", newline="") as handle:
        cluster_rows = list(csv.DictReader(handle))
    credible_keys = {
        row["component_key"]
        for row in cluster_rows
        if str(row["credible_multi_cluster"]).lower() == "true"
    }
    if len(credible_keys) != 78 or not credible_keys <= original_keys:
        raise ValueError("historical 78-component H5 cohort changed")

    with np.load(paths["sparse_anchors"]) as anchors:
        document_indices = anchors["document_indices"]
        anchor_ids = anchors["anchor_ids"]
    documents = json.loads(paths["sparse_documents"].read_text())
    if len(documents) != int(document_indices.max()) + 1:
        raise ValueError("document table and anchor document indices disagree")
    document_ids = [str(row["document_id"]) for row in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("discovery document IDs are not unique")

    csr_by_site = {
        "pre": _load_csr(paths["sparse_pre"]),
        "mid": _load_csr(paths["sparse_mid"]),
    }
    selected_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    eligibility: dict[str, Any] = {}
    original_by_key = {str(row["component_key"]): row for row in original_rows}

    for site in ("pre", "mid"):
        csr = csr_by_site[site]
        unique_documents = csr_unique_document_counts(
            csr["indptr"],
            csr["indices"],
            document_indices,
            dictionary_size=contract.dictionary_size,
        )
        event_counts = np.bincount(
            csr["indices"].astype(np.int64, copy=False),
            minlength=contract.dictionary_size,
        )
        eligible = [
            feature
            for feature in range(contract.dictionary_size)
            if unique_documents[feature] >= contract.minimum_discovery_unique_documents
            and component_key(site, feature) not in original_keys
        ]
        rng = np.random.default_rng(_site_seed(contract.component_sampling_seed, site))
        if len(eligible) < contract.additional_components_per_site:
            raise ValueError(f"insufficient eligible random components for {site}")
        additional = sorted(
            int(value)
            for value in rng.choice(
                eligible, size=contract.additional_components_per_site, replace=False
            )
        )
        eligibility[site] = {
            "eligible_after_original_exclusion": len(eligible),
            "minimum_unique_documents": contract.minimum_discovery_unique_documents,
            "sampled": len(additional),
            "sampling_fraction": len(additional) / len(eligible),
            "site_seed": _site_seed(contract.component_sampling_seed, site),
        }

        site_original = sorted(
            int(row["feature"]) for row in original_rows if row["site"] == site
        )
        if len(site_original) != 60:
            raise ValueError(f"historical H5 site count changed for {site}")
        for feature in site_original + additional:
            key = component_key(site, feature)
            cohort = "original_h5" if key in original_keys else "new_random"
            row = {
                "component_key": key,
                "site": site,
                "feature": feature,
                "cohort": cohort,
                "historical_credible_multi_cluster": key in credible_keys,
                "discovery_natural_firing_events": int(event_counts[feature]),
                "discovery_unique_firing_documents": int(unique_documents[feature]),
            }
            if cohort == "original_h5":
                row["historical_frequency_stratum"] = original_by_key[key][
                    "frequency_stratum"
                ]
            selected_rows.append(row)
            values = feature_event_values(
                csr["indptr"], csr["indices"], csr["values"], feature=feature
            )
            if not len(values):
                raise ValueError(f"selected component has no natural firing events: {key}")
            quantiles = np.quantile(values, [0, 1 / 3, 0.5, 2 / 3, 0.9, 1])
            activation_rows.append(
                {
                    "component_key": key,
                    "natural_firing_events": len(values),
                    "minimum": float(quantiles[0]),
                    "q1_over_3": float(quantiles[1]),
                    "median": float(quantiles[2]),
                    "q2_over_3": float(quantiles[3]),
                    "q90": float(quantiles[4]),
                    "maximum": float(quantiles[5]),
                    "threshold_source": "discovery_selected_natural_firing_events",
                }
            )

    selected_rows.sort(key=lambda row: (row["site"], row["feature"]))
    activation_rows.sort(key=lambda row: row["component_key"])
    if len(selected_rows) != 160 or len({row["component_key"] for row in selected_rows}) != 160:
        raise ValueError("combined component sample must contain 160 unique keys")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "selected_components.json", selected_rows)
    _write_json(
        output_dir / "discovery_document_denylist.json",
        {
            "document_count": len(document_ids),
            "document_ids": sorted(document_ids),
            "purpose": "global document exclusion for all confirmatory extraction",
            "source_split": contract.discovery_split,
        },
    )
    _write_json(output_dir / "activation_thresholds.json", activation_rows)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_COHERENCE_PREPARED",
        "source_git_sha": _git_sha(),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "contract_fingerprint": contract.fingerprint(),
        "source_artifacts": {
            key: {"path": str(path), "sha256": expected[key]}
            for key, path in paths.items()
        },
        "discovery": {
            "split": contract.discovery_split,
            "anchors": len(anchor_ids),
            "documents": len(document_ids),
            "document_denylist_applies_to": contract.confirmatory_split,
        },
        "components": {
            "total": len(selected_rows),
            "pre": sum(row["site"] == "pre" for row in selected_rows),
            "mid": sum(row["site"] == "mid" for row in selected_rows),
            "original_h5": sum(row["cohort"] == "original_h5" for row in selected_rows),
            "historical_credible_multi_cluster": sum(
                row["historical_credible_multi_cluster"] for row in selected_rows
            ),
            "new_random": sum(row["cohort"] == "new_random" for row in selected_rows),
            "eligibility": eligibility,
        },
        "scientific_boundaries": [
            "Historical H5 contexts are discovery-only.",
            "Confirmatory contexts must exclude every discovery document ID.",
            "Activation thresholds are frozen from selected natural BatchTopK events.",
            "No clustering result is itself evidence of polysemanticity.",
        ],
    }
    _write_json(output_dir / "preparation_manifest.json", manifest)
    hashes = {
        name: _sha256(output_dir / name)
        for name in (
            "selected_components.json",
            "discovery_document_denylist.json",
            "activation_thresholds.json",
            "preparation_manifest.json",
        )
    }
    _write_json(output_dir / "artifact_hashes.json", hashes)
    (output_dir / "PREPARED").write_text(
        "ASPD_V3_1_FEATURE_COHERENCE_PREPARED\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = _args()
    print(json.dumps(prepare(args.protocol, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

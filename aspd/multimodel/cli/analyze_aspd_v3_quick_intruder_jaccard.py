"""Analyze v3.1 sparse firings and prepare balanced Llama Intruder tasks."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from aspd.multimodel.cli.analyze_aspd_v3_mechanistic import (
    _document_counts,
    _feature_anchor_lists,
    _load_sparse,
    _minhash_candidates,
    _pair_row,
    _sample_pairs,
    _stratified_features,
)
from aspd.multimodel.cli.prepare_aspd_v3_intruder_final import _build_group
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import QuickIntruderJaccardContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _site_diversity(
    *,
    site: str,
    extraction_dir: Path,
    contract: QuickIntruderJaccardContract,
    anchor_documents: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    indptr, indices, _values, shape = _load_sparse(
        extraction_dir / f"{site}_natural_firings.npz"
    )
    firing, counts = _feature_anchor_lists(indptr, indices, shape[1])
    documents = _document_counts(firing, anchor_documents)
    sampled, strata, eligibility = _stratified_features(
        counts,
        documents,
        minimum_fires=contract.minimum_feature_fires,
        minimum_documents=contract.minimum_unique_documents,
        sample_size=contract.features_per_site,
        seed=contract.feature_sampling_seed,
    )
    eligible = np.asarray(sorted(strata), dtype=np.int64)
    rng = np.random.default_rng(
        contract.jaccard_pair_seed + 1_000 + (0 if site == "pre" else 100)
    )
    pair_rows: list[dict[str, Any]] = []
    strata_summary: dict[str, Any] = {}
    for stratum in contract.frequency_strata:
        features = np.asarray(
            [feature for feature, name in strata.items() if name == stratum],
            dtype=np.int64,
        )
        rows = [
            _pair_row(left, right, firing, shape[0], f"stratum_{stratum}")
            for left, right in _sample_pairs(
                features, contract.jaccard_pairs_per_stratum, rng
            )
        ]
        pair_rows.extend(rows)
        strata_summary[stratum] = {
            "eligible_features": int(len(features)),
            "sampled_pairs": len(rows),
            "mean_jaccard": float(np.mean([row["jaccard"] for row in rows])),
            "median_jaccard": float(np.median([row["jaccard"] for row in rows])),
            "mean_excess_coactivation_rate": float(
                np.mean([row["excess_coactivation_rate"] for row in rows])
            ),
        }
    candidates, lsh = _minhash_candidates(
        firing, eligible, seed=contract.jaccard_pair_seed
    )
    candidate_rows = [
        _pair_row(left, right, firing, shape[0], "minhash_lsh_candidate")
        for left, right in sorted(candidates)
    ]
    candidate_rows.sort(
        key=lambda row: (-float(row["jaccard"]), row["left"], row["right"])
    )
    near_duplicates = [
        row
        for row in candidate_rows
        if float(row["jaccard"]) >= contract.near_duplicate_threshold
    ]
    sampled_features = [int(row["feature"]) for row in sampled]
    summary = {
        "site": site,
        "anchors": shape[0],
        "dictionary_capacity": shape[1],
        "natural_firing_events": int(len(indices)),
        "active_features": int(np.count_nonzero(counts)),
        "dead_features": int(shape[1] - np.count_nonzero(counts)),
        "eligible_features": int(eligibility["eligible_features"]),
        "eligible_low": int(eligibility["eligible_low"]),
        "eligible_medium": int(eligibility["eligible_medium"]),
        "eligible_high": int(eligibility["eligible_high"]),
        "sampled_features": len(sampled_features),
        "sampled_feature_ids": sampled_features,
        "jaccard_pair_sample": {
            "rows": len(pair_rows),
            "mean": float(np.mean([row["jaccard"] for row in pair_rows])),
            "median": float(np.median([row["jaccard"] for row in pair_rows])),
            "mean_excess_coactivation_rate": float(
                np.mean([row["excess_coactivation_rate"] for row in pair_rows])
            ),
            "frequency_strata": strata_summary,
        },
        "near_duplicate_search": {
            **lsh,
            "candidate_discovery": "approximate_minhash_lsh",
            "candidate_jaccard_verification": "exact_on_discovered_candidates",
            "near_duplicate_threshold": contract.near_duplicate_threshold,
            "near_duplicate_pairs": len(near_duplicates),
            "top_pairs": candidate_rows[:500],
        },
    }
    return summary, pair_rows, near_duplicates


def main() -> None:
    args = _args()
    contract = QuickIntruderJaccardContract.load(args.protocol)
    manifest = _json(args.extraction_dir / "experiment_manifest.json")
    expected = {
        "status": "ASPD_V3_1_QUICK_EVALUATION_EXTRACTION_COMPLETE",
        "mode": "full",
        "experiment_family": contract.experiment_family,
        "accepted_evaluation_anchors": contract.accepted_evaluation_anchors,
        "checkpoint_step": contract.checkpoint_step,
        "checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
        "natural_gate_only": True,
        "forced_activations": 0,
        "protocol_fingerprint": contract.fingerprint(),
    }
    mismatch = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise ValueError(f"extraction provenance mismatch: {mismatch}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    with np.load(args.extraction_dir / "anchors.npz") as anchors:
        anchor_documents = anchors["document_indices"].astype(np.int64, copy=False)
    summaries: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    all_near: list[dict[str, Any]] = []
    for site in ("pre", "mid"):
        summary, pairs, near = _site_diversity(
            site=site,
            extraction_dir=args.extraction_dir,
            contract=contract,
            anchor_documents=anchor_documents,
        )
        summaries[site] = summary
        all_pairs.extend({"site": site, **row} for row in pairs)
        all_near.extend({"site": site, **row} for row in near)

    task_contract = SimpleNamespace(
        **contract.model_dump(),
        screening_features_per_arm_site=contract.features_per_site,
        arms=("per_anchor_zscore",),
        intruder_examples=contract.contexts_per_feature + 1,
    )
    tasks: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    task_groups: dict[str, Any] = {}
    sampled_artifact = _json(args.extraction_dir / "sampled_features.json")
    for site in ("pre", "mid"):
        site_tasks, site_answers, report = _build_group(
            arm="per_anchor_zscore",
            site=site,
            extraction_dir=args.extraction_dir,
            contract=task_contract,
        )
        expected_features = sorted(
            int(row["feature"]) for row in sampled_artifact["sites"][site]
        )
        observed_features = sorted(int(row["feature"]) for row in site_answers)
        if not set(observed_features).issubset(expected_features):
            raise ValueError(f"{site} task sampling differs from frozen extraction sample")
        tasks.extend(site_tasks)
        answers.extend(site_answers)
        task_groups[site] = report

    _write_jsonl(args.output_dir / "blinded_parent_tasks.jsonl", tasks)
    _write_jsonl(args.output_dir / "parent_answer_key.jsonl", answers)
    with (args.output_dir / "jaccard_pair_sample.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_pairs[0]))
        writer.writeheader()
        writer.writerows(all_pairs)
    with (args.output_dir / "near_duplicate_pairs.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_pairs[0]))
        writer.writeheader()
        writer.writerows(all_near)
    _write_json(
        args.output_dir / "diversity_summary.json",
        {
            "schema_version": 1,
            "status": "ASPD_V3_1_QUICK_JACCARD_COMPLETE",
            "sites": summaries,
            "comparison_protocol": {
                "anchor_and_document_sampling": "matched to 10M frozen Jaccard protocol",
                "pair_sampling": "20000 deterministic pairs per firing-frequency stratum",
                "near_duplicate_counts": "exact Jaccard only within approximate MinHash/LSH candidates",
            },
        },
    )
    preparation = {
        "schema_version": 1,
        "status": "ASPD_V3_1_QUICK_INTRUDER_TASKS_COMPLETE",
        "git_sha": _git_sha(),
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_fingerprint": contract.fingerprint(),
        "extraction_manifest_sha256": sha256_file(
            args.extraction_dir / "experiment_manifest.json"
        ),
        "checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
        "parent_tasks": len(tasks),
        "groups": task_groups,
        "answer_position_counts": {
            str(position): sum(
                int(row["intruder_position"]) == position for row in answers
            )
            for position in range(1, 6)
        },
        "judge_protocol": "frozen Llama-3.3-70B greedy five-rotation forced choice",
        "outputs": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "blinded_parent_tasks.jsonl",
                "parent_answer_key.jsonl",
                "jaccard_pair_sample.csv",
                "near_duplicate_pairs.csv",
                "diversity_summary.json",
            )
        },
    }
    _write_json(args.output_dir / "preparation_manifest.json", preparation)
    _write_json(
        args.output_dir / "COMPLETE",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(args.output_dir / "preparation_manifest.json"),
        },
    )
    print(json.dumps(preparation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

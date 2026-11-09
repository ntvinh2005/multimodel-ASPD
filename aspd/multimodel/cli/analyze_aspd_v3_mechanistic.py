"""Analyze sparse natural firings and build blinded ASPD v3 Intruder tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import MechanisticValidationContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--zscore-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_sparse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    with np.load(path) as artifact:
        indptr = artifact["indptr"].astype(np.int64, copy=False)
        indices = artifact["indices"].astype(np.int64, copy=False)
        values = artifact["values"].astype(np.float32, copy=False)
        shape = tuple(int(value) for value in artifact["shape"])
    if len(shape) != 2 or len(indptr) != shape[0] + 1 or int(indptr[-1]) != len(indices):
        raise ValueError(f"invalid sparse firing artifact: {path}")
    if len(indices) != len(values) or np.any(indices < 0) or np.any(indices >= shape[1]):
        raise ValueError(f"sparse firing coordinates are invalid: {path}")
    if np.any(np.diff(indptr) < 0) or np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError(f"sparse firing values/indptr are invalid: {path}")
    return indptr, indices, values, shape


def _feature_anchor_lists(
    indptr: np.ndarray, indices: np.ndarray, n_features: int
) -> tuple[list[np.ndarray], np.ndarray]:
    rows = np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))
    order = np.argsort(indices, kind="stable")
    sorted_features = indices[order]
    sorted_rows = rows[order]
    counts = np.bincount(sorted_features, minlength=n_features)
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    firing = [
        sorted_rows[int(boundaries[feature]) : int(boundaries[feature + 1])]
        for feature in range(n_features)
    ]
    return firing, counts.astype(np.int64, copy=False)


def _document_counts(
    firing: list[np.ndarray], anchor_documents: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [len(np.unique(anchor_documents[anchors])) if len(anchors) else 0 for anchors in firing],
        dtype=np.int64,
    )


def _stratified_features(
    counts: np.ndarray,
    document_counts: np.ndarray,
    *,
    minimum_fires: int,
    minimum_documents: int,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, str], dict[str, int]]:
    eligible = np.flatnonzero(
        (counts >= minimum_fires) & (document_counts >= minimum_documents)
    )
    ordered = eligible[np.lexsort((eligible, counts[eligible]))]
    parts = np.array_split(ordered, 3)
    names = ("low", "medium", "high")
    strata = {
        int(feature): name
        for name, values in zip(names, parts, strict=True)
        for feature in values
    }
    base, extra = divmod(sample_size, 3)
    quotas = {name: base + int(index < extra) for index, name in enumerate(names)}
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for name, values in zip(names, parts, strict=True):
        take = min(quotas[name], len(values))
        sampled = np.sort(rng.choice(values, size=take, replace=False)) if take else []
        for feature in sampled:
            selected.append(
                {
                    "feature": int(feature),
                    "frequency_stratum": name,
                    "fires": int(counts[feature]),
                    "unique_documents": int(document_counts[feature]),
                }
            )
    return selected, strata, {
        "eligible_features": int(len(eligible)),
        "sampled_features": len(selected),
        **{f"eligible_{name}": int(len(values)) for name, values in zip(names, parts, strict=True)},
    }


def _sample_pairs(features: np.ndarray, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    features = np.asarray(sorted(int(value) for value in features), dtype=np.int64)
    total = len(features) * (len(features) - 1) // 2
    if total <= count:
        return [
            (int(features[left]), int(features[right]))
            for left in range(len(features))
            for right in range(left + 1, len(features))
        ]
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        chosen = rng.choice(features, size=2, replace=False)
        pairs.add(tuple(sorted((int(chosen[0]), int(chosen[1])))))
    return sorted(pairs)


def _pair_row(
    left: int,
    right: int,
    firing: list[np.ndarray],
    n_anchors: int,
    pair_kind: str,
) -> dict[str, Any]:
    left_rows = firing[left]
    right_rows = firing[right]
    intersection = int(np.intersect1d(left_rows, right_rows, assume_unique=True).size)
    union = len(left_rows) + len(right_rows) - intersection
    independent = len(left_rows) * len(right_rows) / n_anchors
    return {
        "pair_kind": pair_kind,
        "left": left,
        "right": right,
        "left_fires": len(left_rows),
        "right_fires": len(right_rows),
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 0.0,
        "independent_intersection": independent,
        "excess_coactivation_rate": (intersection - independent) / n_anchors,
    }


def _minhash_candidates(
    firing: list[np.ndarray], eligible: np.ndarray, *, seed: int
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    """Return LSH candidates for high-overlap pairs without dense all-pairs materialization."""

    permutations = 32
    rows_per_band = 4
    rng = np.random.default_rng(seed)
    multipliers = rng.integers(1, np.iinfo(np.uint64).max, size=permutations, dtype=np.uint64)
    offsets = rng.integers(0, np.iinfo(np.uint64).max, size=permutations, dtype=np.uint64)
    signatures: dict[int, tuple[int, ...]] = {}
    with np.errstate(over="ignore"):
        for feature in eligible:
            anchors = firing[int(feature)].astype(np.uint64, copy=False)
            hashed = anchors[:, None] * multipliers[None, :] + offsets[None, :]
            signatures[int(feature)] = tuple(int(value) for value in hashed.min(axis=0))
    candidates: set[tuple[int, int]] = set()
    oversized_buckets = 0
    for band in range(permutations // rows_per_band):
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        lower = band * rows_per_band
        upper = lower + rows_per_band
        for feature, signature in signatures.items():
            buckets[signature[lower:upper]].append(feature)
        for values in buckets.values():
            if len(values) > 512:
                oversized_buckets += 1
                values = values[:512]
            for left_index, left in enumerate(values):
                for right in values[left_index + 1 :]:
                    candidates.add((min(left, right), max(left, right)))
    return candidates, {
        "minhash_permutations": permutations,
        "lsh_bands": permutations // rows_per_band,
        "lsh_rows_per_band": rows_per_band,
        "candidate_pairs": len(candidates),
        "oversized_buckets_truncated": oversized_buckets,
    }


def _load_contexts(path: Path) -> dict[int, list[dict[str, Any]]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[int(row["feature"])] = list(row["contexts"])
    return result


def _intruder_task(
    *,
    arm: str,
    site: str,
    selected: dict[str, Any],
    sampled: list[dict[str, Any]],
    contexts: dict[int, list[dict[str, Any]]],
    rng: np.random.Generator,
    examples: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    feature = int(selected["feature"])
    target = [row for row in contexts[feature] if str(row.get("context", "")).strip()]
    if len(target) < examples - 1:
        return None
    target = target[: examples - 1]
    target_documents = {row["document_id"] for row in target}
    donors = [
        row
        for row in sampled
        if int(row["feature"]) != feature
        and row["frequency_stratum"] == selected["frequency_stratum"]
    ]
    rng.shuffle(donors)
    candidates = []
    target_groups = {row.get("source_group") for row in target}
    target_length = float(np.mean([len(row["context"]) for row in target]))
    for donor in donors:
        for context in contexts[int(donor["feature"])]:
            if not str(context.get("context", "")).strip():
                continue
            if context["document_id"] in target_documents:
                continue
            candidates.append(
                (
                    int(context.get("source_group") not in target_groups),
                    abs(len(context["context"]) - target_length),
                    int(donor["feature"]),
                    context,
                )
            )
    if not candidates:
        return None
    _group_penalty, _length_penalty, donor_feature, intruder = min(
        candidates, key=lambda row: row[:3]
    )
    position = int(rng.integers(0, examples))
    displayed = [row["context"] for row in target]
    displayed.insert(position, intruder["context"])
    trial_key = hashlib.sha256(
        f"{arm}\0{site}\0{feature}\0{position}".encode()
    ).hexdigest()[:24]
    task = {
        "schema_version": 1,
        "trial_id": trial_key,
        "instruction": (
            "Four excerpts share a common pattern and one is an intruder. "
            "Return only JSON with integer field intruder_position (1-5) and "
            "string field reason."
        ),
        "examples": [
            {"position": index + 1, "text": text} for index, text in enumerate(displayed)
        ],
    }
    answer = {
        "trial_id": trial_key,
        "arm": arm,
        "site": site,
        "feature": feature,
        "frequency_stratum": selected["frequency_stratum"],
        "feature_fires": selected["fires"],
        "feature_unique_documents": selected["unique_documents"],
        "intruder_position": position + 1,
        "donor_feature": donor_feature,
        "target_document_ids": sorted(target_documents),
        "intruder_document_id": intruder["document_id"],
        "matching": {
            "source_group_matched": intruder.get("source_group") in target_groups,
            "target_mean_characters": target_length,
            "intruder_characters": len(intruder["context"]),
        },
    }
    return task, answer


def _analyze_arm_site(
    arm: str,
    directory: Path,
    site: str,
    protocol: MechanisticValidationContract,
    anchor_documents: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    indptr, indices, _values, shape = _load_sparse(directory / f"{site}_natural_firings.npz")
    firing, counts = _feature_anchor_lists(indptr, indices, shape[1])
    documents = _document_counts(firing, anchor_documents)
    selected, strata, eligibility = _stratified_features(
        counts,
        documents,
        minimum_fires=protocol.minimum_feature_fires,
        minimum_documents=protocol.minimum_unique_documents,
        sample_size=protocol.screening_features_per_arm_site,
        seed=protocol.feature_sampling_seed + (0 if arm == "full_only_control" else 1000) + (0 if site == "pre" else 100),
    )
    eligible = np.asarray(sorted(strata), dtype=np.int64)
    rng = np.random.default_rng(
        protocol.jaccard_pair_seed
        + (0 if arm == "full_only_control" else 1000)
        + (0 if site == "pre" else 100)
    )
    pair_rows = []
    for stratum in protocol.frequency_strata:
        features = np.asarray([feature for feature, name in strata.items() if name == stratum])
        for left, right in _sample_pairs(features, protocol.jaccard_pairs_per_stratum, rng):
            pair_rows.append(_pair_row(left, right, firing, shape[0], f"stratum_{stratum}"))
    candidates, lsh = _minhash_candidates(firing, eligible, seed=protocol.jaccard_pair_seed)
    near_rows = [
        _pair_row(left, right, firing, shape[0], "minhash_lsh_candidate")
        for left, right in sorted(candidates)
    ]
    near_rows.sort(key=lambda row: (-float(row["jaccard"]), row["left"], row["right"]))
    near_duplicates = [row for row in near_rows if float(row["jaccard"]) >= 0.5]
    contexts = _load_contexts(directory / f"{site}_feature_contexts.jsonl")
    task_rng = np.random.default_rng(
        protocol.intruder_position_seed
        + (0 if arm == "full_only_control" else 1000)
        + (0 if site == "pre" else 100)
    )
    tasks = []
    answers = []
    for row in selected:
        result = _intruder_task(
            arm=arm,
            site=site,
            selected=row,
            sampled=selected,
            contexts=contexts,
            rng=task_rng,
            examples=protocol.intruder_examples,
        )
        if result is not None:
            task, answer = result
            tasks.append(task)
            answers.append(answer)
    summary = {
        "arm": arm,
        "site": site,
        "anchors": shape[0],
        "features": shape[1],
        "natural_firing_events": int(len(indices)),
        "features_firing": int((counts > 0).sum()),
        "eligibility": eligibility,
        "sampled_features": selected,
        "intruder_tasks": len(tasks),
        "jaccard_pair_sample": {
            "rows": len(pair_rows),
            "mean": float(np.mean([row["jaccard"] for row in pair_rows])),
            "median": float(np.median([row["jaccard"] for row in pair_rows])),
            "mean_excess_coactivation_rate": float(
                np.mean([row["excess_coactivation_rate"] for row in pair_rows])
            ),
        },
        "near_duplicate_search": {
            **lsh,
            "near_duplicate_threshold": 0.5,
            "near_duplicate_pairs": len(near_duplicates),
            "top_pairs": near_rows[:500],
        },
    }
    return summary, tasks, answers + pair_rows


def main() -> None:
    args = _args()
    protocol = MechanisticValidationContract.load(args.protocol)
    directories = {
        "full_only_control": args.control_dir,
        "per_anchor_zscore": args.zscore_dir,
    }
    manifests = {arm: _load_json(path / "experiment_manifest.json") for arm, path in directories.items()}
    for arm, manifest in manifests.items():
        if manifest.get("status") != "MECHANISTIC_TEST_EXTRACTION_COMPLETE":
            raise ValueError(f"{arm} extraction is not a complete untouched-test artifact")
        if manifest.get("arm") != arm or manifest.get("selected_split") != "test":
            raise ValueError(f"{arm} extraction provenance mismatch")
        if manifest.get("forced_activations") != 0 or not manifest.get("natural_gate_only"):
            raise ValueError(f"{arm} extraction contains artificial activations")
    with np.load(args.control_dir / "anchors.npz") as control_anchors:
        anchor_ids = control_anchors["anchor_ids"]
        anchor_documents = control_anchors["document_indices"].astype(np.int64, copy=False)
    with np.load(args.zscore_dir / "anchors.npz") as treatment_anchors:
        if not np.array_equal(anchor_ids, treatment_anchors["anchor_ids"]):
            raise ValueError("control/z-score extraction anchor IDs are not paired")
        if not np.array_equal(anchor_documents, treatment_anchors["document_indices"]):
            raise ValueError("control/z-score document assignments are not paired")
    control_documents = _load_json(args.control_dir / "documents.json")
    treatment_documents = _load_json(args.zscore_dir / "documents.json")
    if control_documents != treatment_documents:
        raise ValueError("control/z-score source-document manifests differ")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    summaries = []
    tasks = []
    answers = []
    pair_rows = []
    for arm, directory in directories.items():
        for site in ("pre", "mid"):
            summary, site_tasks, mixed_rows = _analyze_arm_site(
                arm, directory, site, protocol, anchor_documents
            )
            summaries.append(summary)
            tasks.extend(site_tasks)
            for row in mixed_rows:
                if "trial_id" in row:
                    answers.append(row)
                else:
                    pair_rows.append({"arm": arm, "site": site, **row})
    with (args.output_dir / "intruder_tasks.jsonl").open("x", encoding="utf-8") as handle:
        for row in tasks:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with (args.output_dir / "intruder_answer_key.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for row in answers:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with (args.output_dir / "jaccard_pair_sample.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    _write_json(args.output_dir / "redundancy_and_sampling_summary.json", summaries)
    manifest = {
        "schema_version": 1,
        "status": "MECHANISTIC_SPARSE_ANALYSIS_COMPLETE",
        "protocol_sha256": sha256_file(args.protocol),
        "paired_anchor_count": len(anchor_ids),
        "paired_document_count": len(control_documents),
        "arm_manifests": {
            arm: sha256_file(directory / "experiment_manifest.json")
            for arm, directory in directories.items()
        },
        "outputs": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "intruder_tasks.jsonl",
                "intruder_answer_key.jsonl",
                "jaccard_pair_sample.csv",
                "redundancy_and_sampling_summary.json",
            )
        },
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)


if __name__ == "__main__":
    main()

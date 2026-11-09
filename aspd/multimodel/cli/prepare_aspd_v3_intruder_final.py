"""Prepare blinded, document-disjoint final ASPD v3 Intruder parent tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.cli.analyze_aspd_v3_mechanistic import (
    _document_counts,
    _feature_anchor_lists,
    _load_sparse,
    _stratified_features,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import FinalIntruderValidationContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--zscore-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def _text_jaccard(left: str, right: str) -> float:
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _context_text(row: dict[str, Any]) -> str:
    value = str(row.get("canonical_context") or row.get("context") or "").strip()
    return value


def _context_shape(row: dict[str, Any]) -> tuple[int, int, int]:
    text = _context_text(row)
    ascii_share = sum(ord(char) < 128 for char in text) / max(1, len(text))
    code_markers = sum(text.count(marker) for marker in ("{", "}", "();", "</", "```"))
    line_count = text.count("\n") + 1
    return round(ascii_share * 10), min(code_markers, 9), min(line_count, 9)


def _contexts_by_feature(path: Path) -> dict[int, list[dict[str, Any]]]:
    return {
        int(row["feature"]): list(row["contexts"])
        for row in _jsonl(path)
    }


def _choose_targets(
    contexts: list[dict[str, Any]], *, count: int, maximum_jaccard: float
) -> list[dict[str, Any]] | None:
    chosen: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for row in sorted(
        contexts, key=lambda value: (-float(value["activation"]), value["anchor_id"])
    ):
        text = _context_text(row)
        document = str(row["document_id"])
        if not text or document in seen_documents:
            continue
        if any(_text_jaccard(text, _context_text(other)) > maximum_jaccard for other in chosen):
            continue
        chosen.append(row)
        seen_documents.add(document)
        if len(chosen) == count:
            return chosen
    return None


def _intruder_candidates(
    *,
    target_feature: int,
    stratum: str,
    sampled: list[dict[str, Any]],
    contexts: dict[int, list[dict[str, Any]]],
    target_contexts: list[dict[str, Any]],
    target_firing: np.ndarray,
    maximum_jaccard: float,
    rng: np.random.Generator,
) -> list[tuple[tuple[Any, ...], int, dict[str, Any]]]:
    target_documents = {str(row["document_id"]) for row in target_contexts}
    target_groups = {str(row.get("source_group", "unknown")) for row in target_contexts}
    target_lengths = [len(_context_text(row)) for row in target_contexts]
    target_length = float(np.median(target_lengths))
    target_shapes = Counter(_context_shape(row) for row in target_contexts)
    firing_set = set(int(value) for value in target_firing)
    donors = [
        row for row in sampled
        if int(row["feature"]) != target_feature and row["frequency_stratum"] == stratum
    ]
    rng.shuffle(donors)
    candidates = []
    for donor in donors:
        donor_feature = int(donor["feature"])
        for row in contexts.get(donor_feature, []):
            text = _context_text(row)
            if not text or str(row["document_id"]) in target_documents:
                continue
            if int(row["anchor_index"]) in firing_set:
                continue
            if any(
                _text_jaccard(text, _context_text(target)) > maximum_jaccard
                for target in target_contexts
            ):
                continue
            score = (
                int(str(row.get("source_group", "unknown")) not in target_groups),
                int(_context_shape(row) not in target_shapes),
                abs(len(text) - target_length),
                donor_feature,
                str(row["anchor_id"]),
            )
            candidates.append((score, donor_feature, row))
    return sorted(candidates, key=lambda value: value[0])


def _build_group(
    *,
    arm: str,
    site: str,
    extraction_dir: Path,
    contract: FinalIntruderValidationContract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indptr, indices, _values, shape = _load_sparse(
        extraction_dir / f"{site}_natural_firings.npz"
    )
    document_rows = _json(extraction_dir / "documents.json")
    with np.load(extraction_dir / "anchors.npz") as anchors:
        document_indices = anchors["document_indices"].astype(np.int64, copy=False)
    document_ids = np.asarray(
        [str(row["document_id"]) for row in document_rows], dtype=object
    )
    documents = document_ids[document_indices]
    if len(documents) != shape[0]:
        raise ValueError(f"{arm}/{site}: anchor/document count mismatch")
    firing, counts = _feature_anchor_lists(indptr, indices, shape[1])
    document_counts = _document_counts(firing, documents)
    sampled, _strata, sampling = _stratified_features(
        counts,
        document_counts,
        minimum_fires=contract.minimum_feature_fires,
        minimum_documents=contract.minimum_unique_documents,
        sample_size=contract.screening_features_per_arm_site,
        seed=contract.feature_sampling_seed,
    )
    contexts = _contexts_by_feature(extraction_dir / f"{site}_feature_contexts.jsonl")
    rng = np.random.default_rng(
        contract.intruder_selection_seed
        + contract.arms.index(arm) * 1_000
        + (0 if site == "pre" else 100)
    )
    tasks: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for selected in sampled:
        feature = int(selected["feature"])
        targets = _choose_targets(
            contexts.get(feature, []),
            count=contract.contexts_per_feature,
            maximum_jaccard=contract.maximum_normalized_text_jaccard,
        )
        if targets is None:
            exclusions["insufficient_distinct_nonduplicate_target_contexts"] += 1
            continue
        candidates = _intruder_candidates(
            target_feature=feature,
            stratum=str(selected["frequency_stratum"]),
            sampled=sampled,
            contexts=contexts,
            target_contexts=targets,
            target_firing=firing[feature],
            maximum_jaccard=contract.maximum_normalized_text_jaccard,
            rng=rng,
        )
        if not candidates:
            exclusions["no_matched_inactive_intruder"] += 1
            continue
        score, donor_feature, intruder = candidates[0]
        canonical = list(targets) + [intruder]
        permutation = rng.permutation(contract.intruder_examples)
        displayed = [canonical[int(index)] for index in permutation]
        intruder_position = int(np.flatnonzero(permutation == 4)[0]) + 1
        trial_id = hashlib.sha256(f"{arm}\0{site}\0{feature}".encode()).hexdigest()[:24]
        tasks.append(
            {
                "schema_version": 4,
                "trial_id": trial_id,
                "examples": [
                    {
                        "candidate_id": f"C{position}",
                        "text": _context_text(row),
                    }
                    for position, row in enumerate(displayed, start=1)
                ],
            }
        )
        answers.append(
            {
                "schema_version": 4,
                "trial_id": trial_id,
                "arm": arm,
                "site": site,
                "feature": feature,
                "frequency_stratum": selected["frequency_stratum"],
                "feature_fires": int(selected["fires"]),
                "feature_unique_documents": int(selected["unique_documents"]),
                "intruder_position": intruder_position,
                "intruder_donor_feature": donor_feature,
                "intruder_anchor_index": int(intruder["anchor_index"]),
                "intruder_anchor_id": intruder["anchor_id"],
                "intruder_document_id": intruder["document_id"],
                "target_anchor_indices": [int(row["anchor_index"]) for row in targets],
                "target_anchor_ids": [row["anchor_id"] for row in targets],
                "target_document_ids": [row["document_id"] for row in targets],
                "target_inactive_at_intruder": int(intruder["anchor_index"])
                not in set(int(value) for value in firing[feature]),
                "intruder_match_score": list(score[:3]),
            }
        )
    report = {
        **sampling,
        "dictionary_capacity": shape[1],
        "features_firing": int(np.count_nonzero(counts)),
        "successfully_constructed": len(tasks),
        "construction_exclusions": dict(exclusions),
    }
    return tasks, answers, report


def _validate_extraction(
    directory: Path, contract: FinalIntruderValidationContract
) -> dict[str, Any]:
    manifest = _json(directory / "experiment_manifest.json")
    if manifest["protocol_fingerprint"] != contract.fingerprint():
        raise ValueError(f"protocol mismatch: {directory}")
    if not manifest.get("excluded_documents_disjoint"):
        raise ValueError(f"excluded-document disjointness not certified: {directory}")
    if int(manifest["accepted_evaluation_anchors"]) != contract.accepted_evaluation_anchors:
        raise ValueError(f"evaluation budget mismatch: {directory}")
    return manifest


def main() -> None:
    args = _args()
    contract = FinalIntruderValidationContract.load(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    control_manifest = _validate_extraction(args.control_dir, contract)
    zscore_manifest = _validate_extraction(args.zscore_dir, contract)
    if sha256_file(args.control_dir / "anchors.npz") != sha256_file(
        args.zscore_dir / "anchors.npz"
    ):
        raise ValueError("arms do not share identical evaluation anchors")
    args.output_dir.mkdir(parents=True)
    all_tasks: list[dict[str, Any]] = []
    all_answers: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    for arm, directory in (
        ("full_only_control", args.control_dir),
        ("per_anchor_zscore", args.zscore_dir),
    ):
        for site in ("pre", "mid"):
            tasks, answers, report = _build_group(
                arm=arm,
                site=site,
                extraction_dir=directory,
                contract=contract,
            )
            all_tasks.extend(tasks)
            all_answers.extend(answers)
            groups[f"{arm}.{site}"] = report
    _write_jsonl(args.output_dir / "blinded_parent_tasks.jsonl", all_tasks)
    _write_jsonl(args.output_dir / "parent_answer_key.jsonl", all_answers)
    manifest = {
        "schema_version": 4,
        "git_sha": _git_sha(),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_fingerprint": contract.fingerprint(),
        "control_extraction_manifest_sha256": sha256_file(
            args.control_dir / "experiment_manifest.json"
        ),
        "zscore_extraction_manifest_sha256": sha256_file(
            args.zscore_dir / "experiment_manifest.json"
        ),
        "control_checkpoint_manifest_sha256": control_manifest[
            "checkpoint_manifest_sha256"
        ],
        "zscore_checkpoint_manifest_sha256": zscore_manifest[
            "checkpoint_manifest_sha256"
        ],
        "matched_anchor_sha256": sha256_file(args.control_dir / "anchors.npz"),
        "excluded_document_manifest_sha256": contract.excluded_document_manifest_sha256,
        "excluded_document_count": contract.excluded_document_count,
        "parent_tasks": len(all_tasks),
        "groups": groups,
    }
    _write_json(args.output_dir / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

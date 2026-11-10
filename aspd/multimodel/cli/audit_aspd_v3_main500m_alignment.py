"""Bounded real-tokenizer audit for the ASPD v3.1 three-way alignment contract."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.alignment import AlignmentResult, align_canonical_text
from aspd.multimodel.v3.data import HashSplitPolicy, SourceDocument, write_json_immutable
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.models import load_v3_tokenizer
from aspd.multimodel.v3.production_stream import load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-documents", type=int, default=2_000)
    parser.add_argument("--sample-documents", type=int, default=512)
    parser.add_argument("--replay-documents", type=int, default=64)
    parser.add_argument("--max-document-characters", type=int, default=100_000)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_record(row: Mapping[str, Any], ordinal: int) -> dict[str, Any] | None:
    text = row.get("text")
    if not isinstance(text, str) or not text:
        return None
    meta = row.get("meta") or {}
    source_group = str(
        meta.get("pile_set_name") if isinstance(meta, Mapping) else "unknown"
    )
    if not source_group or source_group == "None":
        source_group = "unknown"
    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "ordinal": ordinal,
        "document_id": content_sha256,
        "source_group": source_group,
        "text": text,
    }


def _deterministic_sample(
    rows: Iterable[Mapping[str, Any]],
    *,
    scan_documents: int,
    sample_documents: int,
    max_document_characters: int,
    split_policy: HashSplitPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scan_documents <= 0 or sample_documents <= 0:
        raise ValueError("alignment audit document budgets must be positive")
    if sample_documents > scan_documents:
        raise ValueError("sample_documents cannot exceed scan_documents")
    if max_document_characters <= 0:
        raise ValueError("max_document_characters must be positive")
    heap: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    counters: Counter[str] = Counter()
    source_documents: Counter[str] = Counter()
    replay_prefix: list[str] = []
    rows_consumed = 0
    for ordinal, row in enumerate(rows):
        if rows_consumed >= scan_documents:
            break
        rows_consumed += 1
        record = _row_record(row, ordinal)
        if record is None:
            counters["invalid_or_empty_text"] += 1
            continue
        if len(replay_prefix) < 64:
            replay_prefix.append(record["document_id"])
        document_id = str(record["document_id"])
        if document_id in seen:
            counters["exact_duplicate"] += 1
            continue
        seen.add(document_id)
        if len(str(record["text"])) > max_document_characters:
            counters["over_character_budget"] += 1
            continue
        document = SourceDocument(
            document_id=document_id,
            source_group=str(record["source_group"]),
            text=str(record["text"]),
            source_row_id=f"alignment-audit:{ordinal}",
        )
        split = split_policy.split(document)
        counters[f"split:{split}"] += 1
        if split != "train":
            continue
        source_documents[document.source_group] += 1
        rank = int(document_id, 16)
        item = (-rank, document_id, record)
        if len(heap) < sample_documents:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    sample = [item[2] for item in sorted(heap, key=lambda value: value[1])]
    return sample, {
        "rows_consumed": rows_consumed,
        "unique_valid_documents": len(seen),
        "counters": dict(sorted(counters.items())),
        "train_source_documents": dict(sorted(source_documents.items())),
        "initial_valid_document_hashes": replay_prefix,
    }


def _alignment_digest(result: AlignmentResult) -> str:
    return _canonical_sha256(
        [
            {
                "anchor_id": anchor.anchor_id,
                "character_boundary": anchor.character_boundary,
                "byte_boundary": anchor.byte_boundary,
                "prefix_sha256": anchor.canonical_prefix_sha256,
                "positions": {
                    key: {
                        "token_index": value.token_index,
                        "token_id": value.token_id,
                        "character_boundary": value.character_boundary,
                        "byte_boundary": value.byte_boundary,
                    }
                    for key, value in anchor.positions.items()
                },
            }
            for anchor in result.anchors
        ]
    )


def _validate_boundaries(
    text: str,
    result: AlignmentResult,
    model_order: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    previous_boundary = 0
    for anchor in result.anchors:
        boundary = anchor.character_boundary
        prefix = text[:boundary]
        expected_bytes = len(prefix.encode("utf-8"))
        expected_sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        if not previous_boundary < boundary <= len(text):
            failures.append("character_boundaries_not_strictly_increasing")
        if anchor.byte_boundary != expected_bytes:
            failures.append("anchor_utf8_byte_boundary_mismatch")
        if anchor.canonical_prefix_sha256 != expected_sha:
            failures.append("anchor_prefix_sha256_mismatch")
        if tuple(anchor.positions) != tuple(model_order):
            failures.append("anchor_model_order_mismatch")
        for key, position in anchor.positions.items():
            if position.model_key != key:
                failures.append("position_model_identity_mismatch")
            if position.character_boundary != boundary:
                failures.append("position_character_boundary_mismatch")
            if position.byte_boundary != expected_bytes:
                failures.append("position_utf8_byte_boundary_mismatch")
        previous_boundary = boundary
    return failures


def _distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {
        key: value / total
        for key, value in sorted(counter.items())
    } if total else {}


def _total_variation(left: Counter[str], right: Counter[str]) -> float:
    left_distribution = _distribution(left)
    right_distribution = _distribution(right)
    keys = set(left_distribution) | set(right_distribution)
    return 0.5 * sum(
        abs(left_distribution.get(key, 0.0) - right_distribution.get(key, 0.0))
        for key in keys
    )


def _capacity_projection(
    accepted_per_document: Sequence[int],
    *,
    dataset_rows: int | None,
    train_fraction: float,
) -> dict[str, Any]:
    if not accepted_per_document:
        return {"status": "UNAVAILABLE_NO_DOCUMENTS"}
    mean = statistics.fmean(accepted_per_document)
    standard_error = (
        statistics.stdev(accepted_per_document) / math.sqrt(len(accepted_per_document))
        if len(accepted_per_document) > 1
        else 0.0
    )
    lower95 = max(0.0, mean - 1.96 * standard_error)
    required_at_mean = math.inf if mean == 0 else math.ceil(500_000_000 / mean)
    required_at_lower95 = math.inf if lower95 == 0 else math.ceil(500_000_000 / lower95)
    payload: dict[str, Any] = {
        "mean_accepted_anchors_per_document": mean,
        "standard_error": standard_error,
        "lower95_accepted_anchors_per_document": lower95,
        "documents_required_at_mean": (
            None if math.isinf(required_at_mean) else required_at_mean
        ),
        "documents_required_at_lower95": (
            None if math.isinf(required_at_lower95) else required_at_lower95
        ),
    }
    if dataset_rows is None:
        payload["status"] = "DATASET_ROW_COUNT_UNAVAILABLE"
        return payload
    train_rows = math.floor(dataset_rows * train_fraction)
    projected_lower95 = lower95 * train_rows
    payload.update(
        {
            "dataset_rows": dataset_rows,
            "projected_train_rows": train_rows,
            "projected_accepted_anchors_lower95": projected_lower95,
            "margin_over_500m_lower95": projected_lower95 - 500_000_000,
            "status": (
                "PROJECTED_CAPACITY_PASS"
                if projected_lower95 >= 500_000_000
                else "PROJECTED_CAPACITY_FAIL"
            ),
        }
    )
    return payload


def _dataset_rows(dataset: Any) -> int | None:
    splits = getattr(getattr(dataset, "info", None), "splits", None)
    if splits is None or "train" not in splits:
        return None
    value = getattr(splits["train"], "num_examples", None)
    return None if value is None else int(value)


def _replay_prefix(dataset: Any, count: int) -> list[str]:
    hashes: list[str] = []
    for ordinal, row in enumerate(dataset):
        record = _row_record(row, ordinal)
        if record is not None:
            hashes.append(str(record["document_id"]))
        if len(hashes) >= count:
            break
    return hashes


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_main500m_config(args.config)
    registry = build_model_registry_2b(config)
    model_order = tuple(config.architecture.model_order)
    tokenizers = {
        key: load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
        for key in model_order
    }
    split_policy = HashSplitPolicy(
        modulus=config.data.split_modulus,
        validation_buckets=frozenset(config.data.validation_buckets),
        test_buckets=frozenset(config.data.test_buckets),
        seed=config.data.split_seed,
    )
    dataset = load_pinned_pile_dataset(
        config.data.dataset_id,
        config.data.dataset_revision,
    )
    dataset_rows = _dataset_rows(dataset)
    sample, scan = _deterministic_sample(
        dataset,
        scan_documents=args.scan_documents,
        sample_documents=args.sample_documents,
        max_document_characters=args.max_document_characters,
        split_policy=split_policy,
    )
    if len(sample) != args.sample_documents:
        raise RuntimeError(
            f"alignment audit obtained {len(sample)} sampled train documents; "
            f"expected {args.sample_documents}"
        )

    rejection_counts: Counter[str] = Counter()
    source_documents: Counter[str] = Counter()
    source_candidates: Counter[str] = Counter()
    source_accepted: Counter[str] = Counter()
    truncated_models: Counter[str] = Counter()
    prefix_verifiers: Counter[str] = Counter()
    boundary_failures: Counter[str] = Counter()
    accepted_per_document: list[int] = []
    document_rows: list[dict[str, Any]] = []
    deterministic = True
    raw_token_totals: Counter[str] = Counter()
    for record in sample:
        text = str(record["text"])
        result = align_canonical_text(
            text,
            document_id=str(record["document_id"]),
            source_group=str(record["source_group"]),
            tokenizers=tokenizers,
            model_order=model_order,
            max_length=config.data.sequence_length,
        )
        repeated = align_canonical_text(
            text,
            document_id=str(record["document_id"]),
            source_group=str(record["source_group"]),
            tokenizers=tokenizers,
            model_order=model_order,
            max_length=config.data.sequence_length,
        )
        digest = _alignment_digest(result)
        repeated_digest = _alignment_digest(repeated)
        deterministic &= digest == repeated_digest
        report = result.report()
        failures = _validate_boundaries(text, result, model_order)
        boundary_failures.update(failures)
        rejection_counts.update(result.rejection_counts)
        truncated_models.update(report["truncated_models"])
        source = str(record["source_group"])
        candidates = int(report["candidate_canonical_boundaries"])
        accepted = len(result.anchors)
        source_documents[source] += 1
        source_candidates[source] += candidates
        source_accepted[source] += accepted
        accepted_per_document.append(accepted)
        for key, tokenized in result.tokenized.items():
            raw_token_totals[key] += len(tokenized.input_ids)
            prefix_verifiers[tokenized.prefix_verifier] += 1
        document_rows.append(
            {
                "document_id": record["document_id"],
                "source_group": source,
                "accepted_anchors": accepted,
                "candidate_boundaries": candidates,
                "alignment_digest": digest,
                "raw_tokens": {
                    key: len(value.input_ids)
                    for key, value in result.tokenized.items()
                },
            }
        )

    replay_dataset = load_pinned_pile_dataset(
        config.data.dataset_id,
        config.data.dataset_revision,
    )
    replay_observed = _replay_prefix(replay_dataset, args.replay_documents)
    replay_expected = scan["initial_valid_document_hashes"][: args.replay_documents]
    replay_pass = replay_observed == replay_expected
    total_candidates = sum(row["candidate_boundaries"] for row in document_rows)
    total_accepted = sum(accepted_per_document)
    zero_anchor_documents = sum(value == 0 for value in accepted_per_document)
    train_fraction = 1.0 - (
        len(config.data.validation_buckets) + len(config.data.test_buckets)
    ) / config.data.split_modulus
    capacity = _capacity_projection(
        accepted_per_document,
        dataset_rows=dataset_rows,
        train_fraction=train_fraction,
    )
    invariant_pass = not boundary_failures
    engineering_status = (
        "PASS"
        if invariant_pass
        and deterministic
        and replay_pass
        and total_accepted > 0
        and zero_anchor_documents < len(document_rows)
        else "FAIL"
    )
    payload = {
        "schema_version": 1,
        "status": engineering_status,
        "scope": "REAL_PINNED_THREE_TOKENIZER_ALIGNMENT_AUDIT",
        "config": {
            "path": str(args.config),
            "fingerprint": config.fingerprint(),
            "execution_contract_fingerprint": config.execution_contract_fingerprint(),
        },
        "dataset": {
            "id": config.data.dataset_id,
            "revision": config.data.dataset_revision,
            "reported_train_rows": dataset_rows,
        },
        "tokenizers": [
            {
                "model_key": model.key,
                "id": model.tokenizer_id,
                "revision": model.tokenizer_revision,
                "class": type(tokenizers[model.key]).__name__,
                "is_fast": bool(tokenizers[model.key].is_fast),
            }
            for model in config.models
        ],
        "policy": {
            "canonical_text": "identity",
            "alignment": config.data.canonical_text_policy,
            "acceptance": config.data.alignment_acceptance,
            "sequence_length": config.data.sequence_length,
        },
        "sample": {
            "scan_documents": args.scan_documents,
            "sample_documents": len(document_rows),
            "selection": "lowest_content_sha256_unique_train_documents_within_scanned_prefix",
            "max_document_characters": args.max_document_characters,
            "scan": scan,
            "documents": document_rows,
        },
        "results": {
            "accepted_anchors": total_accepted,
            "candidate_canonical_boundaries": total_candidates,
            "boundary_weighted_acceptance_rate": (
                total_accepted / total_candidates if total_candidates else 0.0
            ),
            "zero_anchor_documents": zero_anchor_documents,
            "zero_anchor_document_fraction": zero_anchor_documents / len(document_rows),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "truncated_models": dict(sorted(truncated_models.items())),
            "raw_token_totals": dict(sorted(raw_token_totals.items())),
            "prefix_verifiers": dict(sorted(prefix_verifiers.items())),
            "boundary_invariant_failures": dict(sorted(boundary_failures.items())),
            "boundary_invariants": "PASS" if invariant_pass else "FAIL",
            "within_process_repeatability": "PASS" if deterministic else "FAIL",
            "dataset_stream_prefix_replay": "PASS" if replay_pass else "FAIL",
        },
        "selection_bias": {
            "source_document_distribution": _distribution(source_documents),
            "source_candidate_boundary_distribution": _distribution(source_candidates),
            "source_accepted_anchor_distribution": _distribution(source_accepted),
            "alignment_filter_total_variation": _total_variation(
                source_candidates,
                source_accepted,
            ),
            "document_to_accepted_total_variation": _total_variation(
                source_documents,
                source_accepted,
            ),
            "accepted_anchors_by_source": dict(sorted(source_accepted.items())),
            "candidate_boundaries_by_source": dict(sorted(source_candidates.items())),
        },
        "capacity_projection": capacity,
        "evidence_boundary": (
            "Real pinned tokenizers and a deterministic bounded prefix of the pinned dataset. "
            "Capacity and source-bias values are estimates from the recorded sample, not a full "
            "500M materialization."
        ),
    }
    payload["report_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    args = _parse_args()
    if args.replay_documents <= 0 or args.replay_documents > 64:
        raise ValueError("replay_documents must be in 1..64")
    payload = run_audit(args)
    write_json_immutable(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

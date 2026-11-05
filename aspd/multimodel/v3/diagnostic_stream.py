"""Deterministic validation-only anchor stream for ASPD v3 diagnostics.

The training stream is intentionally not generalized: changing its provenance payload
would make historical checkpoint state incompatible.  This separate stream applies the
same token alignment and document hash split while accepting only the frozen validation
buckets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.alignment import AlignmentResult, FastTokenizer, align_canonical_text
from aspd.multimodel.v3.data import HashSplitPolicy, SourceDocument
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B
from aspd.multimodel.v3.production_stream import (
    AlignedAnchorPool,
    ModelPoolInputs,
    SQLiteStreamDeduplicator,
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ValidationAlignedAnchorStream:
    """Read-only validation stream with checkpointable source/dedup state."""

    STATE_SCHEMA = 1

    def __init__(
        self,
        *,
        dataset: Any,
        dataset_id: str,
        dataset_revision: str,
        tokenizers: Mapping[str, FastTokenizer],
        tokenizer_revisions: Mapping[str, str],
        split_policy: HashSplitPolicy,
        dedup_path: str | Path,
        sequence_length: int = 512,
        model_order: tuple[str, str, str] = MODEL_ORDER_2B,
    ):
        self.model_order = tuple(model_order)
        if self.model_order != MODEL_ORDER_2B:
            raise ValueError(f"validation stream requires model order {MODEL_ORDER_2B}")
        if tuple(tokenizers) != self.model_order or tuple(tokenizer_revisions) != self.model_order:
            raise ValueError("validation stream tokenizer order mismatch")
        if not hasattr(dataset, "state_dict") or not hasattr(dataset, "load_state_dict"):
            raise TypeError("pinned validation dataset must expose resumable state")
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision
        self.tokenizers = dict(tokenizers)
        self.tokenizer_revisions = dict(tokenizer_revisions)
        self.split_policy = split_policy
        self.sequence_length = sequence_length
        self.provenance = {
            "schema_version": self.STATE_SCHEMA,
            "purpose": "aspd_v3_500m_read_only_diagnostic",
            "selected_split": "validation",
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "tokenizer_revisions": dict(tokenizer_revisions),
            "model_order": list(self.model_order),
            "split": {
                "modulus": split_policy.modulus,
                "validation_buckets": sorted(split_policy.validation_buckets),
                "test_buckets": sorted(split_policy.test_buckets),
                "seed": split_policy.seed,
            },
            "sequence_length": sequence_length,
            "canonical_text_policy": "identity_exact_three_way_prefix",
        }
        self.provenance_sha256 = _canonical_sha256(self.provenance)
        self.dedup = SQLiteStreamDeduplicator(
            dedup_path, provenance_sha256=self.provenance_sha256
        )
        self._iterator = iter(dataset)
        self._pending: dict[str, Any] | None = None
        self.documents_read = 0
        self.documents_by_split = {"train": 0, "validation": 0, "test": 0}
        self.exact_duplicates_rejected = 0
        self.zero_anchor_documents = 0

    def close(self) -> None:
        self.dedup.close()

    def _next_validation_alignment(self) -> tuple[SourceDocument, AlignmentResult]:
        while True:
            source_state_before = self.dataset.state_dict()
            try:
                row = next(self._iterator)
            except StopIteration as error:
                raise RuntimeError("pinned dataset exhausted before validation budget") from error
            self.documents_read += 1
            text = row.get("text")
            meta = row.get("meta") or {}
            if not isinstance(text, str) or not text:
                continue
            source_group = str(meta.get("pile_set_name") or "unknown")
            content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source_row_id = _canonical_sha256(
                {"dataset_state_before": source_state_before, "content_sha256": content_sha256}
            )
            document = SourceDocument(
                document_id=content_sha256,
                source_group=source_group,
                text=text,
                source_row_id=source_row_id,
            )
            # Match training: exact duplicates are rejected before split assignment, so
            # repeated content can never occur in both train and validation.
            if not self.dedup.add_if_unique(document):
                self.exact_duplicates_rejected += 1
                continue
            split = self.split_policy.split(document)
            self.documents_by_split[split] += 1
            if split != "validation":
                continue
            alignment = align_canonical_text(
                text,
                document_id=document.document_id,
                source_group=document.source_group,
                tokenizers=self.tokenizers,
                model_order=self.model_order,
                max_length=self.sequence_length,
            )
            if not alignment.anchors:
                self.zero_anchor_documents += 1
                continue
            return document, alignment

    @staticmethod
    def _pending_payload(document: SourceDocument, next_anchor: int) -> dict[str, Any]:
        return {
            "document_id": document.document_id,
            "source_group": document.source_group,
            "text": document.text,
            "source_row_id": document.source_row_id,
            "next_anchor": next_anchor,
        }

    def _restore_pending(self) -> tuple[SourceDocument, AlignmentResult, int] | None:
        if self._pending is None:
            return None
        document = SourceDocument(
            document_id=str(self._pending["document_id"]),
            source_group=str(self._pending["source_group"]),
            text=str(self._pending["text"]),
            source_row_id=str(self._pending["source_row_id"]),
        )
        alignment = align_canonical_text(
            document.text,
            document_id=document.document_id,
            source_group=document.source_group,
            tokenizers=self.tokenizers,
            model_order=self.model_order,
            max_length=self.sequence_length,
        )
        next_anchor = int(self._pending["next_anchor"])
        if not 0 <= next_anchor < len(alignment.anchors):
            raise ValueError("pending validation cursor is outside its anchor list")
        return document, alignment, next_anchor

    def next_pool(self, *, pool_anchors: int = 2048) -> AlignedAnchorPool:
        if pool_anchors <= 0:
            raise ValueError("pool_anchors must be positive")
        anchor_ids: list[str] = []
        document_ids: list[str] = []
        observations: list[dict[str, Any]] = []
        model_ids: dict[str, list[tuple[int, ...]]] = {key: [] for key in self.model_order}
        model_masks: dict[str, list[tuple[int, ...]]] = {key: [] for key in self.model_order}
        model_documents: dict[str, list[int]] = {key: [] for key in self.model_order}
        model_tokens: dict[str, list[int]] = {key: [] for key in self.model_order}

        restored = self._restore_pending()
        while len(anchor_ids) < pool_anchors:
            if restored is None:
                document, alignment = self._next_validation_alignment()
                start = 0
            else:
                document, alignment, start = restored
                restored = None
            count = min(pool_anchors - len(anchor_ids), len(alignment.anchors) - start)
            selected = alignment.anchors[start : start + count]
            document_index = len(document_ids)
            document_ids.append(document.document_id)
            observations.append(
                {
                    "observation_id": document.source_row_id,
                    "document_id": document.document_id,
                    "raw_tokens": {
                        key: len(alignment.tokenized[key].input_ids) for key in self.model_order
                    },
                }
            )
            for key in self.model_order:
                tokenized = alignment.tokenized[key]
                model_ids[key].append(tokenized.input_ids)
                model_masks[key].append(tokenized.attention_mask)
                model_documents[key].extend([document_index] * count)
                model_tokens[key].extend(anchor.positions[key].token_index for anchor in selected)
            anchor_ids.extend(anchor.anchor_id for anchor in selected)
            if start + count < len(alignment.anchors):
                self._pending = self._pending_payload(document, start + count)
                restored = self._restore_pending()
            else:
                self._pending = None

        models = {
            key: ModelPoolInputs(
                input_ids=tuple(model_ids[key]),
                attention_masks=tuple(model_masks[key]),
                anchor_document_indices=tuple(model_documents[key]),
                anchor_token_indices=tuple(model_tokens[key]),
            )
            for key in self.model_order
        }
        pool = AlignedAnchorPool(
            anchor_ids=tuple(anchor_ids),
            document_ids=tuple(document_ids),
            models=models,
            source_observations=tuple(observations),
        )
        pool.validate(
            expected_anchors=pool_anchors,
            sequence_length=self.sequence_length,
            model_order=self.model_order,
        )
        return pool

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA,
            "provenance_sha256": self.provenance_sha256,
            "dataset_state": self.dataset.state_dict(),
            "dedup_high_water": self.dedup.high_water,
            "pending": self._pending,
            "documents_read": self.documents_read,
            "documents_by_split": dict(self.documents_by_split),
            "exact_duplicates_rejected": self.exact_duplicates_rejected,
            "zero_anchor_documents": self.zero_anchor_documents,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != self.STATE_SCHEMA:
            raise ValueError("unsupported validation stream state schema")
        if state.get("provenance_sha256") != self.provenance_sha256:
            raise ValueError("validation stream provenance mismatch")
        self.dataset.load_state_dict(dict(state["dataset_state"]))
        self._iterator = iter(self.dataset)
        self.dedup.rollback(int(state["dedup_high_water"]))
        self._pending = None if state.get("pending") is None else dict(state["pending"])
        self.documents_read = int(state["documents_read"])
        self.documents_by_split = {
            key: int(value) for key, value in state["documents_by_split"].items()
        }
        self.exact_duplicates_rejected = int(state["exact_duplicates_rejected"])
        self.zero_anchor_documents = int(state["zero_anchor_documents"])

"""On-the-fly, exact-resume ASPD v3 streaming for the pinned Pile revision.

Only rank zero owns this stream.  It broadcasts :class:`AlignedAnchorPool` payloads to the
other FSDP ranks, so every logical target sees the same accepted-anchor exposures.  The source
dataset remains an iterable dataset; no per-anchor JSONL materialization is performed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.alignment import AlignmentResult, FastTokenizer, align_canonical_text
from aspd.multimodel.v3.data import HashSplitPolicy, SourceDocument

MODEL_ORDER = ("qwen", "gemma", "phi")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelPoolInputs:
    """One model's bounded document batch and accepted-anchor gather map."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_masks: tuple[tuple[int, ...], ...]
    anchor_document_indices: tuple[int, ...]
    anchor_token_indices: tuple[int, ...]

    def validate(self, anchor_count: int, sequence_length: int) -> None:
        if len(self.input_ids) != len(self.attention_masks) or not self.input_ids:
            raise ValueError("model pool must contain matching non-empty document arrays")
        if len(self.anchor_document_indices) != anchor_count or len(self.anchor_token_indices) != anchor_count:
            raise ValueError("model pool gather maps do not match accepted anchors")
        for ids, mask in zip(self.input_ids, self.attention_masks, strict=True):
            if len(ids) != len(mask) or not ids or len(ids) > sequence_length:
                raise ValueError("model pool contains an invalid native token sequence")
        for document_index, token_index in zip(
            self.anchor_document_indices, self.anchor_token_indices, strict=True
        ):
            if not 0 <= document_index < len(self.input_ids):
                raise ValueError("anchor document index lies outside the pool")
            if not 0 <= token_index < len(self.input_ids[document_index]):
                raise ValueError("anchor token index lies outside its native sequence")


@dataclass(frozen=True)
class AlignedAnchorPool:
    """Exactly one BatchTopK pool on a shared canonical-anchor axis."""

    anchor_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    models: Mapping[str, ModelPoolInputs]
    source_observations: tuple[Mapping[str, Any], ...]

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_ids)

    def validate(
        self,
        *,
        expected_anchors: int,
        sequence_length: int,
        model_order: tuple[str, str, str] = MODEL_ORDER,
    ) -> None:
        if self.anchor_count != expected_anchors:
            raise ValueError(
                f"BatchTopK pool requires exactly {expected_anchors} accepted anchors; "
                f"observed {self.anchor_count}"
            )
        if len(set(self.anchor_ids)) != self.anchor_count:
            raise ValueError("one BatchTopK pool cannot contain duplicate anchor IDs")
        if tuple(self.models) != tuple(model_order):
            raise ValueError(f"model pool order must be {tuple(model_order)}")
        for model in self.models.values():
            model.validate(self.anchor_count, sequence_length)

    def to_payload(self) -> dict[str, Any]:
        return {
            "anchor_ids": list(self.anchor_ids),
            "document_ids": list(self.document_ids),
            # Checkpoint sidecars use canonical JSON with sorted mapping keys.
            # Persist the scientifically significant model order explicitly.
            "model_order": list(self.models),
            "models": {key: asdict(value) for key, value in self.models.items()},
            "source_observations": [dict(value) for value in self.source_observations],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AlignedAnchorPool:
        payload_models = payload["models"]
        if not isinstance(payload_models, Mapping):
            raise TypeError("pool models payload must be a mapping")
        if "model_order" in payload:
            model_order = tuple(str(key) for key in payload["model_order"])
        elif set(payload_models) == {"qwen", "gemma", "smol"}:
            # Compatibility with the first 2B preflight checkpoint, written
            # before the explicit order field was added.
            model_order = ("qwen", "gemma", "smol")
        elif set(payload_models) == set(MODEL_ORDER):
            model_order = MODEL_ORDER
        else:
            model_order = tuple(str(key) for key in payload_models)
        if set(model_order) != set(payload_models) or len(model_order) != len(payload_models):
            raise ValueError("pool model_order must name every model exactly once")
        models = {
            key: ModelPoolInputs(
                input_ids=tuple(tuple(int(v) for v in row) for row in value["input_ids"]),
                attention_masks=tuple(
                    tuple(int(v) for v in row) for row in value["attention_masks"]
                ),
                anchor_document_indices=tuple(
                    int(v) for v in value["anchor_document_indices"]
                ),
                anchor_token_indices=tuple(int(v) for v in value["anchor_token_indices"]),
            )
            for key in model_order
            for value in (payload_models[key],)
        }
        return cls(
            anchor_ids=tuple(str(v) for v in payload["anchor_ids"]),
            document_ids=tuple(str(v) for v in payload["document_ids"]),
            models=models,
            source_observations=tuple(dict(v) for v in payload["source_observations"]),
        )


class SQLiteStreamDeduplicator:
    """Exact document deduplication with a rollbackable checkpoint high-water mark."""

    def __init__(self, path: str | Path, *, provenance_sha256: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One bounded prefetch worker owns stream mutation at a time. Checkpoint state
        # is read only after its future resolves, so cross-thread access is serialized.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        # Blue is a shared parallel filesystem. SQLite WAL relies on a local shared-memory
        # wal-index and can SIGBUS there; rollback journaling preserves exact transactions.
        journal_mode = self.connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(journal_mode).lower() != "delete":
            raise RuntimeError("stream deduplication database requires DELETE journaling")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                provenance_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                content_sha256 TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL UNIQUE
            );
            """
        )
        row = self.connection.execute(
            "SELECT provenance_sha256 FROM metadata WHERE singleton=1"
        ).fetchone()
        if row is None:
            with self.connection:
                self.connection.execute("INSERT INTO metadata VALUES (1,?)", (provenance_sha256,))
        elif row[0] != provenance_sha256:
            self.connection.close()
            raise ValueError("deduplication index provenance does not match the pinned stream")
        self.provenance_sha256 = provenance_sha256

    def close(self) -> None:
        self.connection.close()

    def add_if_unique(self, document: SourceDocument) -> bool:
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO documents(content_sha256,document_id) VALUES (?,?)",
                    (document.content_sha256, document.document_id),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    @property
    def high_water(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(ordinal),0) FROM documents").fetchone()
        return int(row[0])

    def rollback(self, high_water: int) -> None:
        if high_water < 0 or high_water > self.high_water:
            raise ValueError("deduplication rollback high-water mark is invalid")
        with self.connection:
            self.connection.execute("DELETE FROM documents WHERE ordinal > ?", (high_water,))


class PileAlignedAnchorStream:
    """Deterministic bounded-memory document stream with exact tokenizer alignment."""

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
        model_order: tuple[str, str, str] = MODEL_ORDER,
    ):
        self.model_order = tuple(model_order)
        if len(self.model_order) != 3:
            raise ValueError("production stream requires exactly three tokenizer families")
        if tuple(tokenizers) != self.model_order or tuple(tokenizer_revisions) != self.model_order:
            raise ValueError(f"production stream requires ordered {self.model_order} tokenizers")
        if not hasattr(dataset, "state_dict") or not hasattr(dataset, "load_state_dict"):
            raise TypeError("pinned dataset stream must expose state_dict/load_state_dict")
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision
        self.tokenizers = dict(tokenizers)
        self.tokenizer_revisions = dict(tokenizer_revisions)
        self.split_policy = split_policy
        self.sequence_length = sequence_length
        self.provenance = {
            "schema_version": self.STATE_SCHEMA,
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
        self._iterator = iter(self.dataset)
        self._pending: dict[str, Any] | None = None
        self.documents_read = 0
        self.documents_by_split = {"train": 0, "validation": 0, "test": 0}
        self.exact_duplicates_rejected = 0
        self.zero_anchor_documents = 0

    def close(self) -> None:
        self.dedup.close()

    def _next_train_alignment(self) -> tuple[SourceDocument, AlignmentResult]:
        while True:
            source_state_before = self.dataset.state_dict()
            try:
                row = next(self._iterator)
            except StopIteration as error:
                raise RuntimeError("pinned Pile stream exhausted before the exposure budget") from error
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
            if not self.dedup.add_if_unique(document):
                self.exact_duplicates_rejected += 1
                continue
            split = self.split_policy.split(document)
            self.documents_by_split[split] += 1
            if split != "train":
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
            raise ValueError("pending aligned-document cursor lies outside its anchor list")
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
                document, alignment = self._next_train_alignment()
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
            raise ValueError("unsupported production stream checkpoint schema")
        if state.get("provenance_sha256") != self.provenance_sha256:
            raise ValueError("production stream checkpoint provenance mismatch")
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


def load_pinned_pile_dataset(dataset_id: str, revision: str) -> Any:
    """Create the immutable Hugging Face iterable without applying random operators."""

    from datasets import load_dataset

    return load_dataset(dataset_id, revision=revision, split="train", streaming=True)


DatasetFactory = Callable[[str, str], Any]

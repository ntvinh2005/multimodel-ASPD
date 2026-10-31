"""Bounded-memory document splitting, deduplication, and v3 manifest infrastructure."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from aspd.multimodel.v3.alignment import AlignmentResult

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_group: str
    text: str
    source_row_id: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HashSplitPolicy:
    """Explicit candidate split buckets; no split proportions are silently selected."""

    modulus: int
    validation_buckets: frozenset[int]
    test_buckets: frozenset[int]
    seed: int

    def __post_init__(self) -> None:
        if self.modulus <= 2:
            raise ValueError("split modulus must leave room for train/validation/test")
        all_reserved = self.validation_buckets | self.test_buckets
        if not self.validation_buckets or not self.test_buckets:
            raise ValueError("validation and test bucket sets must both be explicit and non-empty")
        if self.validation_buckets & self.test_buckets:
            raise ValueError("validation and test split buckets overlap")
        if any(bucket < 0 or bucket >= self.modulus for bucket in all_reserved):
            raise ValueError("split bucket lies outside the configured modulus")

    def split(self, document: SourceDocument) -> SplitName:
        payload = (
            f"{self.seed}\0{document.source_group}\0{document.document_id}"
        ).encode()
        bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % self.modulus
        if bucket in self.validation_buckets:
            return "validation"
        if bucket in self.test_buckets:
            return "test"
        return "train"


def _word_shingles(text: str, width: int) -> frozenset[int]:
    words = text.casefold().split()
    if not words:
        return frozenset()
    if len(words) < width:
        values = [" ".join(words)]
    else:
        values = [" ".join(words[index : index + width]) for index in range(len(words) - width + 1)]
    return frozenset(
        int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
        for value in values
    )


class SQLiteDuplicateIndex:
    """Disk-backed exact and LSH-screened near-duplicate index with bounded RAM use."""

    def __init__(
        self,
        path: str | Path,
        *,
        shingle_width: int = 5,
        signature_size: int = 16,
        band_size: int = 4,
        near_duplicate_threshold: float = 0.85,
    ):
        if signature_size % band_size:
            raise ValueError("MinHash signature_size must be divisible by band_size")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shingle_width = shingle_width
        self.signature_size = signature_size
        self.band_size = band_size
        self.threshold = near_duplicate_threshold
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                content_sha256 TEXT NOT NULL UNIQUE,
                shingles_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bands (
                band_key TEXT NOT NULL,
                document_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS bands_key_idx ON bands(band_key);
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteDuplicateIndex:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _signature(self, shingles: frozenset[int]) -> tuple[int, ...]:
        if not shingles:
            return tuple(0 for _ in range(self.signature_size))
        mask = (1 << 64) - 1
        return tuple(
            min(((value ^ (0x9E3779B97F4A7C15 * (seed + 1))) & mask) for value in shingles)
            for seed in range(self.signature_size)
        )

    def _bands(self, signature: tuple[int, ...]) -> tuple[str, ...]:
        result = []
        for start in range(0, self.signature_size, self.band_size):
            payload = json.dumps(signature[start : start + self.band_size]).encode("ascii")
            result.append(hashlib.sha256(payload).hexdigest())
        return tuple(result)

    def classify(self, document: SourceDocument, split: SplitName) -> dict[str, Any]:
        exact = self.connection.execute(
            "SELECT document_id, split FROM documents WHERE content_sha256 = ?",
            (document.content_sha256,),
        ).fetchone()
        if exact is not None:
            return {
                "status": "EXACT_DUPLICATE",
                "matched_document_id": exact[0],
                "matched_split": exact[1],
            }
        shingles = _word_shingles(document.text, self.shingle_width)
        signature = self._signature(shingles)
        band_keys = self._bands(signature)
        placeholders = ",".join("?" for _ in band_keys)
        candidates = self.connection.execute(
            f"SELECT DISTINCT d.document_id, d.split, d.shingles_json "
            f"FROM bands b JOIN documents d ON d.document_id=b.document_id "
            f"WHERE b.band_key IN ({placeholders})",
            band_keys,
        ).fetchall()
        best: tuple[float, str, str] | None = None
        for candidate_id, candidate_split, encoded in candidates:
            other = frozenset(json.loads(encoded))
            union = len(shingles | other)
            similarity = len(shingles & other) / max(1, union)
            if best is None or similarity > best[0]:
                best = (similarity, candidate_id, candidate_split)
        if best is not None and best[0] >= self.threshold:
            return {
                "status": "NEAR_DUPLICATE",
                "similarity": best[0],
                "matched_document_id": best[1],
                "matched_split": best[2],
            }
        with self.connection:
            self.connection.execute(
                "INSERT INTO documents VALUES (?,?,?,?)",
                (
                    document.document_id,
                    split,
                    document.content_sha256,
                    json.dumps(sorted(shingles)),
                ),
            )
            self.connection.executemany(
                "INSERT INTO bands VALUES (?,?)",
                [(key, document.document_id) for key in band_keys],
            )
        return {"status": "UNIQUE"}


@dataclass
class ExposureCounters:
    accepted_aligned_anchor_exposures: int = 0
    unique_aligned_anchors: int = 0
    raw_tokens_qwen: int = 0
    raw_tokens_gemma: int = 0
    raw_tokens_phi: int = 0
    optimization_steps: int = 0
    repeated_anchor_exposures: int = 0
    newly_observed_anchor_exposures: int = 0

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise AssertionError("v3 exposure counters must be non-negative")
        if self.accepted_aligned_anchor_exposures != (
            self.repeated_anchor_exposures + self.newly_observed_anchor_exposures
        ):
            raise AssertionError("v3 exposure counters lost repeated/new accounting")
        if self.unique_aligned_anchors != self.newly_observed_anchor_exposures:
            raise AssertionError("v3 unique anchors disagree with durable first observations")


@dataclass(frozen=True)
class ExposureLedgerReceipt:
    """Compact checkpoint receipt for one exact exposure-ledger high-water mark."""

    schema_version: int
    ledger_uuid: str
    dataset_manifest_sha256: str
    world_size: int
    optimizer_step: int
    receipt_sha256: str
    counters: ExposureCounters
    rank_exposures: dict[int, int]

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported exposure-ledger receipt schema")
        if self.world_size <= 0 or set(self.rank_exposures) != set(range(self.world_size)):
            raise ValueError("exposure-ledger receipt has incomplete rank counters")
        if self.optimizer_step != self.counters.optimization_steps:
            raise ValueError("exposure-ledger receipt step and counters disagree")
        self.counters.validate()
        if sum(self.rank_exposures.values()) != self.counters.accepted_aligned_anchor_exposures:
            raise ValueError("exposure-ledger receipt rank/global totals disagree")
        for name, value in (
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid exposure-ledger receipt fingerprint: {name}")
        try:
            uuid.UUID(self.ledger_uuid)
        except ValueError as error:
            raise ValueError("invalid exposure-ledger UUID") from error

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "ledger_uuid": self.ledger_uuid,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "world_size": self.world_size,
            "optimizer_step": self.optimizer_step,
            "receipt_sha256": self.receipt_sha256,
            "counters": asdict(self.counters),
            "rank_exposures": {
                str(rank): count for rank, count in sorted(self.rank_exposures.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExposureLedgerReceipt:
        receipt = cls(
            schema_version=int(payload["schema_version"]),
            ledger_uuid=str(payload["ledger_uuid"]),
            dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
            world_size=int(payload["world_size"]),
            optimizer_step=int(payload["optimizer_step"]),
            receipt_sha256=str(payload["receipt_sha256"]),
            counters=ExposureCounters(**payload["counters"]),
            rank_exposures={
                int(rank): int(count) for rank, count in payload["rank_exposures"].items()
            },
        )
        receipt.validate()
        return receipt


class SQLiteExposureLedger:
    """Exact O(batch) exposure accounting with compact checkpoint receipts.

    The hot path performs one batched membership insert plus constant-size counter updates.
    Full-history scans are available only through :meth:`audit_full_consistency`.  A single
    rank-zero writer owns global accounting; future distributed training must aggregate all
    rank-local IDs before invoking this class.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_manifest_sha256: str,
        world_size: int = 1,
        writer_rank: int = 0,
    ):
        if len(dataset_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_manifest_sha256
        ):
            raise ValueError("exposure ledger requires a lowercase SHA-256 manifest identity")
        if world_size <= 0:
            raise ValueError("exposure ledger world_size must be positive")
        if writer_rank != 0:
            raise ValueError("only rank 0 may own the global v3 exposure ledger")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.world_size = world_size
        self.writer_rank = writer_rank
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS exposure_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS seen_anchors (
                anchor_id TEXT PRIMARY KEY,
                first_step INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS seen_anchors_first_step_idx
                ON seen_anchors(first_step);
            CREATE TABLE IF NOT EXISTS optimizer_steps (
                step_id INTEGER PRIMARY KEY,
                batch_sha256 TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                newly_observed INTEGER NOT NULL,
                repeated INTEGER NOT NULL,
                cross_rank_duplicates INTEGER NOT NULL,
                per_rank_counts_json TEXT NOT NULL,
                cumulative_accepted INTEGER NOT NULL,
                cumulative_newly_observed INTEGER NOT NULL,
                cumulative_repeated INTEGER NOT NULL,
                rank_exposures_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alignment_observations (
                observation_id TEXT PRIMARY KEY,
                payload_sha256 TEXT NOT NULL,
                raw_tokens_qwen INTEGER NOT NULL,
                raw_tokens_gemma INTEGER NOT NULL,
                raw_tokens_phi INTEGER NOT NULL,
                observed_before_step INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exposure_totals (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                accepted INTEGER NOT NULL,
                newly_observed INTEGER NOT NULL,
                repeated INTEGER NOT NULL,
                optimization_steps INTEGER NOT NULL,
                raw_tokens_qwen INTEGER NOT NULL,
                raw_tokens_gemma INTEGER NOT NULL,
                raw_tokens_phi INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rank_exposures (
                rank INTEGER PRIMARY KEY,
                accepted INTEGER NOT NULL
            );
            """
        )
        existing = dict(self.connection.execute("SELECT key,value FROM exposure_metadata"))
        if existing:
            expected_fixed = {
                "schema_version": str(self.SCHEMA_VERSION),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "world_size": str(world_size),
                "global_writer_rank": "0",
            }
            observed_fixed = {key: existing.get(key) for key in expected_fixed}
            if observed_fixed != expected_fixed or set(existing) != set(expected_fixed) | {
                "ledger_uuid",
                "genesis_receipt_sha256",
            }:
                self.connection.close()
                raise ValueError(
                    "exposure ledger provenance mismatch or unsupported schema: "
                    f"expected={expected_fixed}, observed={existing}"
                )
            self.ledger_uuid = existing["ledger_uuid"]
            self.genesis_receipt_sha256 = self._genesis_receipt(self.ledger_uuid)
            if existing["genesis_receipt_sha256"] != self.genesis_receipt_sha256:
                self.connection.close()
                raise ValueError("exposure ledger genesis receipt is corrupt")
        else:
            self.ledger_uuid = str(uuid.uuid4())
            self.genesis_receipt_sha256 = self._genesis_receipt(self.ledger_uuid)
            metadata = {
                "schema_version": str(self.SCHEMA_VERSION),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "world_size": str(world_size),
                "global_writer_rank": "0",
                "ledger_uuid": self.ledger_uuid,
                "genesis_receipt_sha256": self.genesis_receipt_sha256,
            }
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO exposure_metadata(key,value) VALUES (?,?)", metadata.items()
                )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO exposure_totals VALUES (1,0,0,0,0,0,0,0)"
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO rank_exposures(rank,accepted) VALUES (?,0)",
                ((rank,) for rank in range(world_size)),
            )
        self._validate_constant_time_state()

    def _genesis_receipt(self, ledger_uuid: str) -> str:
        payload = (
            f"aspd-v3-exposure-ledger-v3\0{ledger_uuid}\0"
            f"{self.dataset_manifest_sha256}\0{self.world_size}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _step_receipt(
        previous_receipt: str,
        step_id: int,
        batch_sha256: str,
        accepted: int,
        newly_observed: int,
        repeated: int,
        cumulative: tuple[int, int, int],
        rank_exposures_json: str,
    ) -> str:
        payload = json.dumps(
            {
                "previous_receipt": previous_receipt,
                "step_id": step_id,
                "batch_sha256": batch_sha256,
                "accepted": accepted,
                "newly_observed": newly_observed,
                "repeated": repeated,
                "cumulative": cumulative,
                "rank_exposures_json": rank_exposures_json,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteExposureLedger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def snapshot(self) -> ExposureCounters:
        row = self.connection.execute(
            "SELECT accepted,newly_observed,repeated,optimization_steps,"
            "raw_tokens_qwen,raw_tokens_gemma,raw_tokens_phi "
            "FROM exposure_totals WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise ValueError("exposure ledger is missing its durable totals")
        accepted, new, repeated, steps, qwen, gemma, phi = (int(value) for value in row)
        return ExposureCounters(
            accepted_aligned_anchor_exposures=accepted,
            unique_aligned_anchors=new,
            raw_tokens_qwen=qwen,
            raw_tokens_gemma=gemma,
            raw_tokens_phi=phi,
            optimization_steps=steps,
            repeated_anchor_exposures=repeated,
            newly_observed_anchor_exposures=new,
        )

    def rank_exposure_counts(self) -> dict[int, int]:
        return {
            int(rank): int(accepted)
            for rank, accepted in self.connection.execute(
                "SELECT rank,accepted FROM rank_exposures ORDER BY rank"
            )
        }

    def _validate_constant_time_state(self) -> None:
        counters = self.snapshot()
        counters.validate()
        ranks = self.rank_exposure_counts()
        if set(ranks) != set(range(self.world_size)):
            raise ValueError("exposure ledger is missing rank counters")
        if sum(ranks.values()) != counters.accepted_aligned_anchor_exposures:
            raise ValueError("exposure ledger per-rank/global totals disagree")
        if counters.optimization_steps == 0:
            if any(
                (
                    counters.accepted_aligned_anchor_exposures,
                    counters.newly_observed_anchor_exposures,
                    counters.repeated_anchor_exposures,
                )
            ):
                raise ValueError("empty exposure ledger has nonzero optimizer counters")
            return
        last = self.connection.execute(
            "SELECT cumulative_accepted,cumulative_newly_observed,cumulative_repeated,"
            "rank_exposures_json FROM optimizer_steps WHERE step_id=?",
            (counters.optimization_steps - 1,),
        ).fetchone()
        if last is None:
            raise ValueError("exposure ledger is missing its high-water optimizer step")
        if tuple(int(value) for value in last[:3]) != (
            counters.accepted_aligned_anchor_exposures,
            counters.newly_observed_anchor_exposures,
            counters.repeated_anchor_exposures,
        ) or {int(key): int(value) for key, value in json.loads(last[3]).items()} != ranks:
            raise ValueError("exposure ledger high-water receipt and totals disagree")

    def audit_full_consistency(self) -> dict[str, int | str]:
        """Perform an explicit O(history) audit; never called by the optimizer hot path."""

        counters = self.snapshot()
        ranks = {rank: 0 for rank in range(self.world_size)}
        cumulative = [0, 0, 0]
        previous_receipt = self.genesis_receipt_sha256
        expected_step = 0
        cursor = self.connection.execute(
            "SELECT step_id,batch_sha256,accepted,newly_observed,repeated,per_rank_counts_json,"
            "cumulative_accepted,cumulative_newly_observed,cumulative_repeated,"
            "rank_exposures_json,receipt_sha256 FROM optimizer_steps ORDER BY step_id"
        )
        for row in cursor:
            step_id = int(row[0])
            if step_id != expected_step:
                raise ValueError("exposure ledger optimizer steps are not contiguous")
            per_rank = {int(key): int(value) for key, value in json.loads(row[5]).items()}
            for rank in range(self.world_size):
                ranks[rank] += per_rank[rank]
            cumulative[0] += int(row[2])
            cumulative[1] += int(row[3])
            cumulative[2] += int(row[4])
            rank_json = json.dumps(ranks, sort_keys=True, separators=(",", ":"))
            receipt = self._step_receipt(
                previous_receipt,
                step_id,
                str(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                tuple(cumulative),
                rank_json,
            )
            if tuple(int(value) for value in row[6:9]) != tuple(cumulative):
                raise ValueError("exposure ledger cumulative step counters are corrupt")
            if str(row[9]) != rank_json or str(row[10]) != receipt:
                raise ValueError("exposure ledger receipt hash chain is corrupt")
            previous_receipt = receipt
            expected_step += 1
        durable_unique = int(
            self.connection.execute("SELECT COUNT(*) FROM seen_anchors").fetchone()[0]
        )
        raw_totals = tuple(
            int(value)
            for value in self.connection.execute(
                "SELECT COALESCE(SUM(raw_tokens_qwen),0),"
                "COALESCE(SUM(raw_tokens_gemma),0),COALESCE(SUM(raw_tokens_phi),0) "
                "FROM alignment_observations"
            ).fetchone()
        )
        if expected_step != counters.optimization_steps:
            raise ValueError("exposure ledger step count and totals disagree")
        if tuple(cumulative) != (
            counters.accepted_aligned_anchor_exposures,
            counters.newly_observed_anchor_exposures,
            counters.repeated_anchor_exposures,
        ):
            raise ValueError("exposure ledger historical and aggregate counters disagree")
        if durable_unique != counters.unique_aligned_anchors or ranks != self.rank_exposure_counts():
            raise ValueError("exposure ledger membership/rank history is inconsistent")
        if raw_totals != (
            counters.raw_tokens_qwen,
            counters.raw_tokens_gemma,
            counters.raw_tokens_phi,
        ):
            raise ValueError("exposure ledger alignment-token history is inconsistent")
        return {
            "status": "PASS",
            "optimizer_steps": expected_step,
            "unique_aligned_anchors": durable_unique,
            "receipt_sha256": previous_receipt,
        }

    def checkpoint_receipt(self, *, expected_optimizer_steps: int) -> ExposureLedgerReceipt:
        counters = self.snapshot()
        if counters.optimization_steps != expected_optimizer_steps:
            raise ValueError(
                "exposure ledger receipt step does not match trainer checkpoint: "
                f"ledger={counters.optimization_steps}, trainer={expected_optimizer_steps}"
            )
        if counters.optimization_steps:
            row = self.connection.execute(
                "SELECT receipt_sha256 FROM optimizer_steps WHERE step_id=?",
                (counters.optimization_steps - 1,),
            ).fetchone()
            if row is None:
                raise ValueError("exposure ledger high-water receipt is missing")
            receipt_sha256 = str(row[0])
        else:
            receipt_sha256 = self.genesis_receipt_sha256
        return ExposureLedgerReceipt(
            schema_version=2,
            ledger_uuid=self.ledger_uuid,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            world_size=self.world_size,
            optimizer_step=counters.optimization_steps,
            receipt_sha256=receipt_sha256,
            counters=counters,
            rank_exposures=self.rank_exposure_counts(),
        )

    def write_checkpoint_receipt(
        self, path: str | Path, *, expected_optimizer_steps: int
    ) -> ExposureLedgerReceipt:
        receipt = self.checkpoint_receipt(expected_optimizer_steps=expected_optimizer_steps)
        write_json_immutable(path, receipt.payload())
        return receipt

    def recover_to_receipt(self, receipt: ExposureLedgerReceipt) -> ExposureCounters:
        """Rollback only post-checkpoint deltas; the full membership DB is never copied."""

        receipt.validate()
        if (
            receipt.ledger_uuid != self.ledger_uuid
            or receipt.dataset_manifest_sha256 != self.dataset_manifest_sha256
            or receipt.world_size != self.world_size
        ):
            raise ValueError("exposure-ledger receipt provenance mismatch")
        current = self.snapshot()
        if current.optimization_steps < receipt.optimizer_step:
            raise ValueError("live exposure ledger is behind the checkpoint receipt")
        if receipt.optimizer_step:
            row = self.connection.execute(
                "SELECT receipt_sha256 FROM optimizer_steps WHERE step_id=?",
                (receipt.optimizer_step - 1,),
            ).fetchone()
            observed_receipt = None if row is None else str(row[0])
        else:
            observed_receipt = self.genesis_receipt_sha256
        if observed_receipt != receipt.receipt_sha256:
            raise ValueError("live exposure ledger does not contain the checkpoint receipt")
        raw_counts_match = (
            current.raw_tokens_qwen,
            current.raw_tokens_gemma,
            current.raw_tokens_phi,
        ) == (
            receipt.counters.raw_tokens_qwen,
            receipt.counters.raw_tokens_gemma,
            receipt.counters.raw_tokens_phi,
        )
        if current.optimization_steps == receipt.optimizer_step and raw_counts_match:
            self._validate_constant_time_state()
            return current

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "DELETE FROM optimizer_steps WHERE step_id>=?", (receipt.optimizer_step,)
            )
            self.connection.execute(
                "DELETE FROM seen_anchors WHERE first_step>=?", (receipt.optimizer_step,)
            )
            self.connection.execute(
                "DELETE FROM alignment_observations WHERE observed_before_step>=?",
                (receipt.optimizer_step,),
            )
            counters = receipt.counters
            self.connection.execute(
                "UPDATE exposure_totals SET accepted=?,newly_observed=?,repeated=?,"
                "optimization_steps=?,raw_tokens_qwen=?,raw_tokens_gemma=?,raw_tokens_phi=? "
                "WHERE singleton=1",
                (
                    counters.accepted_aligned_anchor_exposures,
                    counters.newly_observed_anchor_exposures,
                    counters.repeated_anchor_exposures,
                    counters.optimization_steps,
                    counters.raw_tokens_qwen,
                    counters.raw_tokens_gemma,
                    counters.raw_tokens_phi,
                ),
            )
            self.connection.executemany(
                "UPDATE rank_exposures SET accepted=? WHERE rank=?",
                ((receipt.rank_exposures[rank], rank) for rank in range(self.world_size)),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        self._validate_constant_time_state()
        return self.snapshot()

    def observe_raw_tokens(
        self,
        observation_id: str,
        counts: Mapping[str, int],
        *,
        before_optimizer_step: int | None = None,
    ) -> None:
        if not observation_id:
            raise ValueError("alignment observation requires a stable global identity")
        current_step = self.snapshot().optimization_steps
        if before_optimizer_step is None:
            before_optimizer_step = current_step
        if before_optimizer_step != current_step:
            raise ValueError("raw-token observation must be attached to the current step boundary")
        counts = {key: int(counts[key]) for key in ("qwen", "gemma", "phi")}
        if any(value < 0 for value in counts.values()):
            raise ValueError("raw-token observations cannot be negative")
        payload = json.dumps(counts, sort_keys=True, separators=(",", ":"))
        payload_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT payload_sha256 FROM alignment_observations WHERE observation_id=?",
                (observation_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload_sha256:
                    raise ValueError("alignment observation identity was reused with new counts")
                self.connection.rollback()
                return
            self.connection.execute(
                "INSERT INTO alignment_observations VALUES (?,?,?,?,?,?)",
                (
                    observation_id,
                    payload_sha256,
                    counts["qwen"],
                    counts["gemma"],
                    counts["phi"],
                    before_optimizer_step,
                ),
            )
            self.connection.execute(
                "UPDATE exposure_totals SET raw_tokens_qwen=raw_tokens_qwen+?, "
                "raw_tokens_gemma=raw_tokens_gemma+?, raw_tokens_phi=raw_tokens_phi+? "
                "WHERE singleton=1",
                (counts["qwen"], counts["gemma"], counts["phi"]),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def observe_alignment(self, observation_id: str, result: AlignmentResult) -> None:
        self.observe_raw_tokens(
            observation_id,
            {
                key: int(sum(result.tokenized[key].attention_mask))
                for key in ("qwen", "gemma", "phi")
            },
        )

    def observe_optimizer_step(self, step_id: int, anchor_ids: Sequence[str]) -> ExposureCounters:
        return self.observe_global_optimizer_step(step_id, {0: anchor_ids})

    def observe_global_optimizer_step(
        self,
        step_id: int,
        per_rank_anchor_ids: Mapping[int, Sequence[str]],
    ) -> ExposureCounters:
        if step_id < 0:
            raise ValueError("optimizer step identity must be non-negative")
        expected_ranks = set(range(self.world_size))
        if set(per_rank_anchor_ids) != expected_ranks:
            raise ValueError(
                "global exposure accounting requires exactly one batch from every rank: "
                f"expected={sorted(expected_ranks)}, observed={sorted(per_rank_anchor_ids)}"
            )
        normalized = {
            rank: [str(anchor_id) for anchor_id in per_rank_anchor_ids[rank]]
            for rank in range(self.world_size)
        }
        flat = [anchor_id for rank in range(self.world_size) for anchor_id in normalized[rank]]
        if not flat or any(not anchor_id for anchor_id in flat):
            raise ValueError("an optimizer step must contain nonempty stable anchor IDs")
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        batch_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        per_rank_counts_json = json.dumps(
            {rank: len(values) for rank, values in normalized.items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        current = self.snapshot()
        if step_id < current.optimization_steps:
            existing = self.connection.execute(
                "SELECT batch_sha256 FROM optimizer_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if existing is None or existing[0] != batch_sha256:
                raise ValueError("optimizer step identity was reused with a different batch")
            return current
        if step_id != current.optimization_steps:
            raise ValueError(
                "optimizer steps must be committed contiguously: "
                f"expected={current.optimization_steps}, observed={step_id}"
            )
        anchor_ranks: dict[str, set[int]] = {}
        for rank, anchor_ids in normalized.items():
            for anchor_id in set(anchor_ids):
                anchor_ranks.setdefault(anchor_id, set()).add(rank)
        cross_rank_duplicates = sum(
            len(ranks) - 1 for ranks in anchor_ranks.values() if len(ranks) > 1
        )
        rank_counts = self.rank_exposure_counts()
        for rank in range(self.world_size):
            rank_counts[rank] += len(normalized[rank])
        rank_json = json.dumps(rank_counts, sort_keys=True, separators=(",", ":"))
        previous_receipt = (
            self.genesis_receipt_sha256
            if step_id == 0
            else str(
                self.connection.execute(
                    "SELECT receipt_sha256 FROM optimizer_steps WHERE step_id=?", (step_id - 1,)
                ).fetchone()[0]
            )
        )

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            changes_before = self.connection.total_changes
            self.connection.executemany(
                "INSERT OR IGNORE INTO seen_anchors(anchor_id,first_step) VALUES (?,?)",
                ((anchor_id, step_id) for anchor_id in flat),
            )
            newly_observed = self.connection.total_changes - changes_before
            accepted = len(flat)
            repeated = accepted - newly_observed
            cumulative = (
                current.accepted_aligned_anchor_exposures + accepted,
                current.newly_observed_anchor_exposures + newly_observed,
                current.repeated_anchor_exposures + repeated,
            )
            receipt_sha256 = self._step_receipt(
                previous_receipt,
                step_id,
                batch_sha256,
                accepted,
                newly_observed,
                repeated,
                cumulative,
                rank_json,
            )
            self.connection.execute(
                "INSERT INTO optimizer_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    step_id,
                    batch_sha256,
                    accepted,
                    newly_observed,
                    repeated,
                    cross_rank_duplicates,
                    per_rank_counts_json,
                    *cumulative,
                    rank_json,
                    receipt_sha256,
                ),
            )
            self.connection.execute(
                "UPDATE exposure_totals SET accepted=?,newly_observed=?,repeated=?,"
                "optimization_steps=? WHERE singleton=1",
                (*cumulative, step_id + 1),
            )
            self.connection.executemany(
                "UPDATE rank_exposures SET accepted=? WHERE rank=?",
                ((rank_counts[rank], rank) for rank in range(self.world_size)),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return ExposureCounters(
            accepted_aligned_anchor_exposures=cumulative[0],
            unique_aligned_anchors=cumulative[1],
            raw_tokens_qwen=current.raw_tokens_qwen,
            raw_tokens_gemma=current.raw_tokens_gemma,
            raw_tokens_phi=current.raw_tokens_phi,
            optimization_steps=step_id + 1,
            repeated_anchor_exposures=cumulative[2],
            newly_observed_anchor_exposures=cumulative[1],
        )


class ManifestAccumulator:
    """Disk-backed manifest accounting; document identities never accumulate in RAM."""

    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        split_policy: HashSplitPolicy,
        tokenizer_revisions: dict[str, str],
        ledger_path: str | Path,
    ):
        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision
        self.split_policy = split_policy
        self.tokenizer_revisions = tokenizer_revisions
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS manifest_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton=1),
                provenance_json TEXT NOT NULL,
                provenance_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS split_ledger (
                document_id TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                source_group TEXT NOT NULL,
                source_row_id TEXT NOT NULL,
                anchor_count INTEGER NOT NULL,
                rejection_counts_json TEXT NOT NULL
            );
            """
        )
        columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(split_ledger)")
        }
        required_columns = {
            "document_id",
            "split",
            "content_sha256",
            "source_group",
            "source_row_id",
            "anchor_count",
            "rejection_counts_json",
        }
        if columns != required_columns:
            self.connection.close()
            raise ValueError(
                "existing v3 manifest ledger lacks resumable aggregate fields; "
                f"expected={sorted(required_columns)}, observed={sorted(columns)}"
            )
        provenance = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "split_policy": {
                "modulus": split_policy.modulus,
                "validation_buckets": sorted(split_policy.validation_buckets),
                "test_buckets": sorted(split_policy.test_buckets),
                "seed": split_policy.seed,
            },
            "tokenizer_revisions": dict(sorted(tokenizer_revisions.items())),
        }
        provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        provenance_sha256 = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
        existing = self.connection.execute(
            "SELECT provenance_json,provenance_sha256 FROM manifest_metadata WHERE singleton=1"
        ).fetchone()
        if existing is None:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO manifest_metadata VALUES (1,?,?)",
                    (provenance_json, provenance_sha256),
                )
        elif existing != (provenance_json, provenance_sha256):
            self.connection.close()
            raise ValueError("v3 manifest ledger provenance does not match the requested resume")
        self.documents_by_split = {"train": 0, "validation": 0, "test": 0}
        self.anchors_by_split = {"train": 0, "validation": 0, "test": 0}
        self.rejection_counts: dict[str, int] = {}
        self._restore_aggregates()

    def _restore_aggregates(self) -> None:
        rows = self.connection.execute(
            "SELECT split,COUNT(*),COALESCE(SUM(anchor_count),0) "
            "FROM split_ledger GROUP BY split"
        ).fetchall()
        for split, documents, anchors in rows:
            if split not in self.documents_by_split:
                raise ValueError(f"v3 manifest ledger contains an invalid split: {split}")
            self.documents_by_split[split] = int(documents)
            self.anchors_by_split[split] = int(anchors)
        for (encoded,) in self.connection.execute(
            "SELECT rejection_counts_json FROM split_ledger"
        ):
            for reason, count in json.loads(encoded).items():
                self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + int(count)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ManifestAccumulator:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def add(self, document: SourceDocument, split: SplitName, alignment: AlignmentResult) -> None:
        if alignment.document_id != document.document_id:
            raise ValueError("alignment/document identity mismatch in v3 manifest ledger")
        if alignment.source_group != document.source_group:
            raise ValueError("alignment/document source-group mismatch in v3 manifest ledger")
        if split not in self.documents_by_split:
            raise ValueError(f"invalid v3 manifest split: {split}")
        existing = self.connection.execute(
            "SELECT split FROM split_ledger WHERE document_id = ?", (document.document_id,)
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"document already assigned to split {existing[0]}: {document.document_id}"
            )
        expected_split = self.split_policy.split(document)
        if split != expected_split:
            raise ValueError(
                f"document split disagrees with deterministic policy: "
                f"document={document.document_id}, expected={expected_split}, observed={split}"
            )
        with self.connection:
            self.connection.execute(
                "INSERT INTO split_ledger VALUES (?,?,?,?,?,?,?)",
                (
                    document.document_id,
                    split,
                    document.content_sha256,
                    document.source_group,
                    document.source_row_id,
                    len(alignment.anchors),
                    json.dumps(alignment.rejection_counts, sort_keys=True),
                ),
            )
        self.documents_by_split[split] += 1
        self.anchors_by_split[split] += len(alignment.anchors)
        for reason, count in alignment.rejection_counts.items():
            self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + count

    def manifest(self) -> dict[str, Any]:
        ledger_rows = int(
            self.connection.execute("SELECT COUNT(*) FROM split_ledger").fetchone()[0]
        )
        if ledger_rows != sum(self.documents_by_split.values()):
            raise ValueError("v3 split ledger and manifest document counts disagree")
        payload = {
            "schema_version": 3,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split_unit": "document",
            "split_policy": {
                "modulus": self.split_policy.modulus,
                "validation_buckets": sorted(self.split_policy.validation_buckets),
                "test_buckets": sorted(self.split_policy.test_buckets),
                "seed": self.split_policy.seed,
            },
            "tokenizer_revisions": dict(sorted(self.tokenizer_revisions.items())),
            "documents_by_split": self.documents_by_split,
            "anchors_by_split": self.anchors_by_split,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "source_groups": [
                row[0]
                for row in self.connection.execute(
                    "SELECT DISTINCT source_group FROM split_ledger ORDER BY source_group"
                )
            ],
            "organism_training_data": False,
            "split_ledger_rows": ledger_rows,
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


def write_json_immutable(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable v3 artifact: {target}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, target)
        except OSError:
            if os.name != "nt":
                raise
            temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)


def counters_payload(counters: ExposureCounters) -> dict[str, int]:
    return {key: int(value) for key, value in asdict(counters).items()}

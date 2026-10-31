"""Bounded-memory document splitting, deduplication, and v3 manifest infrastructure."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
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
        if self.accepted_aligned_anchor_exposures != (
            self.repeated_anchor_exposures + self.newly_observed_anchor_exposures
        ):
            raise AssertionError("v3 exposure counters lost repeated/new accounting")
        if self.unique_aligned_anchors != self.newly_observed_anchor_exposures:
            raise AssertionError("v3 unique anchors disagree with durable first observations")


class SQLiteExposureLedger:
    """Exact bounded-memory global exposure accounting with transactional restart safety.

    One designated rank-zero writer receives the complete global optimizer batch.  This
    prevents per-rank unique counts from being naively summed when an anchor is duplicated
    across ranks.  A future distributed trainer must aggregate IDs before calling this API.
    """

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
            CREATE TABLE IF NOT EXISTS optimizer_steps (
                step_id INTEGER PRIMARY KEY,
                batch_sha256 TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                newly_observed INTEGER NOT NULL,
                repeated INTEGER NOT NULL,
                cross_rank_duplicates INTEGER NOT NULL,
                per_rank_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alignment_observations (
                observation_id TEXT PRIMARY KEY,
                payload_sha256 TEXT NOT NULL,
                raw_tokens_qwen INTEGER NOT NULL,
                raw_tokens_gemma INTEGER NOT NULL,
                raw_tokens_phi INTEGER NOT NULL
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
        expected_metadata = {
            "schema_version": "1",
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "world_size": str(world_size),
            "global_writer_rank": "0",
        }
        existing = dict(self.connection.execute("SELECT key, value FROM exposure_metadata"))
        if existing and existing != expected_metadata:
            self.connection.close()
            raise ValueError(
                "exposure ledger provenance mismatch: "
                f"expected={expected_metadata}, observed={existing}"
            )
        with self.connection:
            if not existing:
                self.connection.executemany(
                    "INSERT INTO exposure_metadata(key,value) VALUES (?,?)",
                    expected_metadata.items(),
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO exposure_totals VALUES (1,0,0,0,0,0,0,0)"
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO rank_exposures(rank,accepted) VALUES (?,0)",
                ((rank,) for rank in range(world_size)),
            )
        self._validate_durable_state()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteExposureLedger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _validate_durable_state(self) -> None:
        counters = self.snapshot()
        counters.validate()
        durable_unique = int(
            self.connection.execute("SELECT COUNT(*) FROM seen_anchors").fetchone()[0]
        )
        durable_steps = int(
            self.connection.execute("SELECT COUNT(*) FROM optimizer_steps").fetchone()[0]
        )
        step_totals = tuple(
            int(value)
            for value in self.connection.execute(
                "SELECT COALESCE(SUM(accepted),0),COALESCE(SUM(newly_observed),0),"
                "COALESCE(SUM(repeated),0) FROM optimizer_steps"
            ).fetchone()
        )
        rank_total = int(
            self.connection.execute(
                "SELECT COALESCE(SUM(accepted),0) FROM rank_exposures"
            ).fetchone()[0]
        )
        if durable_unique != counters.unique_aligned_anchors:
            raise ValueError("exposure ledger unique-anchor table and totals disagree")
        if durable_steps != counters.optimization_steps:
            raise ValueError("exposure ledger optimizer-step table and totals disagree")
        expected_step_totals = (
            counters.accepted_aligned_anchor_exposures,
            counters.newly_observed_anchor_exposures,
            counters.repeated_anchor_exposures,
        )
        if step_totals != expected_step_totals:
            raise ValueError("exposure ledger step rows and totals disagree")
        if rank_total != counters.accepted_aligned_anchor_exposures:
            raise ValueError("exposure ledger per-rank and global accepted totals disagree")

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

    def backup_to(self, path: str | Path, *, expected_optimizer_steps: int) -> Path:
        """Create an immutable SQLite snapshot aligned to one trainer checkpoint step."""

        counters = self.snapshot()
        if counters.optimization_steps != expected_optimizer_steps:
            raise ValueError(
                "exposure ledger snapshot step does not match trainer checkpoint: "
                f"ledger={counters.optimization_steps}, trainer={expected_optimizer_steps}"
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite exposure-ledger snapshot: {target}")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            destination = sqlite3.connect(temporary)
            try:
                self.connection.backup(destination)
                destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                destination.commit()
            finally:
                destination.close()
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except OSError:
                if os.name != "nt":
                    raise
                temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def observe_alignment(self, observation_id: str, result: AlignmentResult) -> None:
        if not observation_id:
            raise ValueError("alignment observation requires a stable global identity")
        counts = {
            key: int(sum(result.tokenized[key].attention_mask))
            for key in ("qwen", "gemma", "phi")
        }
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
                "INSERT INTO alignment_observations VALUES (?,?,?,?,?)",
                (observation_id, payload_sha256, counts["qwen"], counts["gemma"], counts["phi"]),
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
        anchor_ranks: dict[str, set[int]] = {}
        for rank, anchor_ids in normalized.items():
            for anchor_id in set(anchor_ids):
                anchor_ranks.setdefault(anchor_id, set()).add(rank)
        cross_rank_duplicates = sum(
            len(ranks) - 1 for ranks in anchor_ranks.values() if len(ranks) > 1
        )

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT batch_sha256 FROM optimizer_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if existing is not None:
                if existing[0] != batch_sha256:
                    raise ValueError("optimizer step identity was reused with a different batch")
                self.connection.rollback()
                return self.snapshot()
            newly_observed = 0
            for anchor_id in flat:
                cursor = self.connection.execute(
                    "INSERT OR IGNORE INTO seen_anchors(anchor_id,first_step) VALUES (?,?)",
                    (anchor_id, step_id),
                )
                newly_observed += int(cursor.rowcount == 1)
            accepted = len(flat)
            repeated = accepted - newly_observed
            self.connection.execute(
                "INSERT INTO optimizer_steps VALUES (?,?,?,?,?,?,?)",
                (
                    step_id,
                    batch_sha256,
                    accepted,
                    newly_observed,
                    repeated,
                    cross_rank_duplicates,
                    encoded,
                ),
            )
            self.connection.execute(
                "UPDATE exposure_totals SET accepted=accepted+?, "
                "newly_observed=newly_observed+?, repeated=repeated+?, "
                "optimization_steps=optimization_steps+1 WHERE singleton=1",
                (accepted, newly_observed, repeated),
            )
            self.connection.executemany(
                "UPDATE rank_exposures SET accepted=accepted+? WHERE rank=?",
                ((len(normalized[rank]), rank) for rank in range(self.world_size)),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        self._validate_durable_state()
        return self.snapshot()


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

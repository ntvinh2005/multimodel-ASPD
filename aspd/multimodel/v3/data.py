"""Bounded-memory document splitting, deduplication, and v3 manifest infrastructure."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
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

    def observe_alignment(self, result: AlignmentResult) -> None:
        self.unique_aligned_anchors += len(result.anchors)
        for key in ("qwen", "gemma", "phi"):
            document = result.tokenized[key]
            count = sum(document.attention_mask)
            setattr(self, f"raw_tokens_{key}", getattr(self, f"raw_tokens_{key}") + count)

    def observe_optimizer_step(self, anchor_ids: list[str], previously_seen: set[str]) -> None:
        if not anchor_ids:
            raise ValueError("an optimizer step must contain accepted aligned anchors")
        repeated = sum(anchor_id in previously_seen for anchor_id in anchor_ids)
        new = len(anchor_ids) - repeated
        self.accepted_aligned_anchor_exposures += len(anchor_ids)
        self.repeated_anchor_exposures += repeated
        self.newly_observed_anchor_exposures += new
        self.optimization_steps += 1
        previously_seen.update(anchor_ids)
        if self.accepted_aligned_anchor_exposures != (
            self.repeated_anchor_exposures + self.newly_observed_anchor_exposures
        ):
            raise AssertionError("v3 exposure counters lost repeated/new accounting")


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
        self.documents_by_split = {"train": 0, "validation": 0, "test": 0}
        self.anchors_by_split = {"train": 0, "validation": 0, "test": 0}
        self.rejection_counts: dict[str, int] = {}
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS split_ledger ("
            "document_id TEXT PRIMARY KEY, split TEXT NOT NULL, "
            "content_sha256 TEXT NOT NULL, source_group TEXT NOT NULL)"
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ManifestAccumulator:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def add(self, document: SourceDocument, split: SplitName, alignment: AlignmentResult) -> None:
        existing = self.connection.execute(
            "SELECT split FROM split_ledger WHERE document_id = ?", (document.document_id,)
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"document already assigned to split {existing[0]}: {document.document_id}"
            )
        with self.connection:
            self.connection.execute(
                "INSERT INTO split_ledger VALUES (?,?,?,?)",
                (document.document_id, split, document.content_sha256, document.source_group),
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
            "source_groups": ["pile_uncopyrighted"],
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

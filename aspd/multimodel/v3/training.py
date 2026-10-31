"""Resume-safe ASPD v3 streaming state and exact exposure accounting."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.v3.config import V3ConfigBundle
from aspd.multimodel.v3.data import ExposureCounters, counters_payload, write_json_immutable


@dataclass(frozen=True)
class AlignedStreamCursor:
    shard_index: int
    document_row_index: int
    anchor_offset_in_document: int
    source_epoch: int
    accepted_unique_anchors: int
    raw_documents_seen: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("v3 aligned-stream cursor values must be non-negative")


@dataclass(frozen=True)
class DistributedPlan:
    world_size: int
    microbatch_anchors_per_rank: int
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        if min(
            self.world_size,
            self.microbatch_anchors_per_rank,
            self.gradient_accumulation_steps,
        ) <= 0:
            raise ValueError("distributed v3 batch factors must be positive")

    @property
    def effective_anchors_per_optimizer_step(self) -> int:
        return (
            self.world_size
            * self.microbatch_anchors_per_rank
            * self.gradient_accumulation_steps
        )


@dataclass
class V3TrainingState:
    schema_version: int
    optimizer_step: int
    stream_cursor: AlignedStreamCursor
    counters: ExposureCounters
    dataset_manifest_sha256: str
    alignment_manifest_sha256: str
    config_fingerprint: str
    sampler_rng_state_b64: str
    exposure_ledger_optimizer_step: int
    exposure_ledger_sha256: str
    per_rank_microbatch_index: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 3:
            raise ValueError("unsupported ASPD v3 training-state schema")
        self.counters.validate()
        if self.optimizer_step != self.counters.optimization_steps:
            raise ValueError("checkpoint optimizer step and exposure counters disagree")
        if self.exposure_ledger_optimizer_step != self.optimizer_step:
            raise ValueError("checkpoint and durable exposure-ledger steps disagree")
        for name in (
            "dataset_manifest_sha256",
            "alignment_manifest_sha256",
            "config_fingerprint",
            "exposure_ledger_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid checkpoint fingerprint: {name}")
        if any(value < 0 for value in self.per_rank_microbatch_index.values()):
            raise ValueError("per-rank microbatch indices must be non-negative")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "optimizer_step": self.optimizer_step,
            "stream_cursor": asdict(self.stream_cursor),
            "counters": counters_payload(self.counters),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "alignment_manifest_sha256": self.alignment_manifest_sha256,
            "config_fingerprint": self.config_fingerprint,
            "sampler_rng_state_b64": self.sampler_rng_state_b64,
            "exposure_ledger_optimizer_step": self.exposure_ledger_optimizer_step,
            "exposure_ledger_sha256": self.exposure_ledger_sha256,
            "per_rank_microbatch_index": self.per_rank_microbatch_index,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V3TrainingState:
        state = cls(
            schema_version=int(payload["schema_version"]),
            optimizer_step=int(payload["optimizer_step"]),
            stream_cursor=AlignedStreamCursor(**payload["stream_cursor"]),
            counters=ExposureCounters(**payload["counters"]),
            dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
            alignment_manifest_sha256=str(payload["alignment_manifest_sha256"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            sampler_rng_state_b64=str(payload["sampler_rng_state_b64"]),
            exposure_ledger_optimizer_step=int(payload["exposure_ledger_optimizer_step"]),
            exposure_ledger_sha256=str(payload["exposure_ledger_sha256"]),
            per_rank_microbatch_index={
                str(key): int(value)
                for key, value in payload.get("per_rank_microbatch_index", {}).items()
            },
        )
        state.validate()
        return state


def encode_rng_state(generator: random.Random) -> str:
    import pickle

    return base64.b64encode(pickle.dumps(generator.getstate(), protocol=5)).decode("ascii")


def restore_rng_state(encoded: str) -> random.Random:
    import pickle

    generator = random.Random()
    generator.setstate(pickle.loads(base64.b64decode(encoded.encode("ascii"))))
    return generator


def minimum_optimizer_steps(minimum_accepted_exposures: int, effective_batch: int) -> int:
    if minimum_accepted_exposures <= 0 or effective_batch <= 0:
        raise ValueError("exposure and effective-batch values must be positive")
    return math.ceil(minimum_accepted_exposures / effective_batch)


def validate_finite_tensors(values: Mapping[str, Tensor]) -> None:
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite v3 tensor: {name}")


def deterministic_rank_shard(values: Sequence[str], rank: int, world_size: int) -> list[str]:
    if not 0 <= rank < world_size:
        raise ValueError("distributed rank lies outside world_size")
    return [value for index, value in enumerate(values) if index % world_size == rank]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_coordinated_checkpoint(
    output_dir: str | Path,
    *,
    state: V3TrainingState,
    model_checkpoint: str | Path,
    exposure_ledger_snapshot: str | Path,
) -> Path:
    """Commit weights and exact producer state behind one immutable marker."""

    state.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    weights = Path(model_checkpoint)
    if not weights.is_file():
        raise FileNotFoundError(f"v3 model checkpoint missing: {weights}")
    exposure_snapshot = Path(exposure_ledger_snapshot)
    if not exposure_snapshot.is_file():
        raise FileNotFoundError(f"v3 exposure-ledger snapshot missing: {exposure_snapshot}")
    if _sha256_file(exposure_snapshot) != state.exposure_ledger_sha256:
        raise ValueError("v3 exposure-ledger snapshot checksum disagrees with trainer state")
    state_path = root / f"source_state_{state.optimizer_step:08d}.json"
    write_json_immutable(state_path, state.payload())
    marker = {
        "schema_version": 3,
        "optimizer_step": state.optimizer_step,
        "model_checkpoint": str(weights.resolve()),
        "model_checkpoint_sha256": _sha256_file(weights),
        "source_state": state_path.name,
        "source_state_sha256": _sha256_file(state_path),
        "exposure_ledger_snapshot": str(exposure_snapshot.resolve()),
        "exposure_ledger_sha256": state.exposure_ledger_sha256,
        "dataset_manifest_sha256": state.dataset_manifest_sha256,
        "alignment_manifest_sha256": state.alignment_manifest_sha256,
        "config_fingerprint": state.config_fingerprint,
    }
    marker_path = root / f"checkpoint_{state.optimizer_step:08d}.marker.json"
    write_json_immutable(marker_path, marker)
    latest = root / "latest_checkpoint_marker.txt"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(marker_path.name + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(latest)
    return marker_path


def restore_coordinated_checkpoint(
    marker_path: str | Path,
    *,
    expected_dataset_manifest_sha256: str,
    expected_alignment_manifest_sha256: str,
    expected_config_fingerprint: str,
) -> tuple[V3TrainingState, Path]:
    marker_file = Path(marker_path)
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    state_path = marker_file.parent / marker["source_state"]
    weights = Path(marker["model_checkpoint"])
    exposure_snapshot = Path(marker["exposure_ledger_snapshot"])
    if _sha256_file(state_path) != marker["source_state_sha256"]:
        raise ValueError("v3 source-state checksum mismatch")
    if _sha256_file(weights) != marker["model_checkpoint_sha256"]:
        raise ValueError("v3 model-checkpoint checksum mismatch")
    if _sha256_file(exposure_snapshot) != marker["exposure_ledger_sha256"]:
        raise ValueError("v3 exposure-ledger snapshot checksum mismatch")
    expected = {
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
        "alignment_manifest_sha256": expected_alignment_manifest_sha256,
        "config_fingerprint": expected_config_fingerprint,
    }
    mismatches = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v3 checkpoint provenance mismatch: {mismatches}")
    state = V3TrainingState.from_payload(json.loads(state_path.read_text(encoding="utf-8")))
    if state.optimizer_step != int(marker["optimizer_step"]):
        raise ValueError("v3 checkpoint marker and producer state refer to different steps")
    if state.exposure_ledger_sha256 != marker["exposure_ledger_sha256"]:
        raise ValueError("v3 checkpoint state and exposure-ledger snapshot disagree")
    connection = sqlite3.connect(exposure_snapshot)
    try:
        row = connection.execute(
            "SELECT optimization_steps FROM exposure_totals WHERE singleton=1"
        ).fetchone()
        metadata = dict(connection.execute("SELECT key,value FROM exposure_metadata"))
    finally:
        connection.close()
    if row is None or int(row[0]) != state.optimizer_step:
        raise ValueError("v3 exposure-ledger snapshot and checkpoint refer to different steps")
    if metadata.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("v3 exposure-ledger dataset provenance mismatch")
    return state, weights


def validate_main_training_request(bundle: V3ConfigBundle) -> None:
    """Fail closed before model loading, distributed launch, or output creation."""

    bundle.require_ready()
    if bundle.runtime.allow_main_training is not True:  # statically False in current schema
        raise RuntimeError("ASPD v3 main training has no explicit execution approval")

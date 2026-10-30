"""Bounded deterministic streaming over frozen aligned-anchor JSONL shards."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.training import encode_rng_state, restore_rng_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AlignedAnchorStream:
    """Exact-resume stream whose memory is bounded by one shard row plus shuffle buffer."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        seed: int,
        shuffle_buffer_size: int,
    ):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 3:
            raise ValueError("aligned-anchor stream requires schema version 3")
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.shards = [row for row in manifest["shards"] if row["split"] == split]
        if not self.shards:
            raise ValueError(f"aligned-anchor manifest has no {split} shards")
        self.shards.sort(key=lambda row: int(row["shard_index"]))
        for expected, row in enumerate(self.shards):
            if int(row["shard_index"]) != expected:
                raise ValueError(f"non-contiguous aligned-anchor shard index at {expected}")
            path = self._path(row)
            if not path.is_file() or _sha256(path) != row["sha256"]:
                raise ValueError(f"aligned-anchor shard missing or corrupt: {path}")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rng = random.Random(seed)
        self.epoch = 0
        self.shard_index = 0
        self.row_index = 0
        self.buffer: list[dict[str, Any]] = []
        self.exhausted = False

    def _path(self, row: Mapping[str, Any]) -> Path:
        path = Path(str(row["path"]))
        return path if path.is_absolute() else self.manifest_path.parent / path

    def _next_raw(self) -> dict[str, Any] | None:
        while self.shard_index < len(self.shards):
            path = self._path(self.shards[self.shard_index])
            with path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index < self.row_index:
                        continue
                    self.row_index = index + 1
                    record = json.loads(line)
                    if not record.get("anchor_id") or tuple(record.get("model_order", ())) != (
                        "qwen",
                        "gemma",
                        "phi",
                    ):
                        raise ValueError(f"invalid aligned-anchor record in {path}:{index + 1}")
                    return record
            self.shard_index += 1
            self.row_index = 0
        self.exhausted = True
        return None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while len(self.buffer) < self.shuffle_buffer_size and not self.exhausted:
            record = self._next_raw()
            if record is not None:
                self.buffer.append(record)
        while self.buffer:
            index = self.rng.randrange(len(self.buffer))
            record = self.buffer.pop(index)
            replacement = self._next_raw() if not self.exhausted else None
            if replacement is not None:
                self.buffer.append(replacement)
            yield record

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "manifest_sha256": self.manifest_sha256,
            "seed": self.seed,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "epoch": self.epoch,
            "shard_index": self.shard_index,
            "row_index": self.row_index,
            "buffer": self.buffer,
            "exhausted": self.exhausted,
            "rng_state_b64": encode_rng_state(self.rng),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 3:
            raise ValueError("unsupported aligned-anchor stream state")
        expected = {
            "manifest_sha256": self.manifest_sha256,
            "seed": self.seed,
            "shuffle_buffer_size": self.shuffle_buffer_size,
        }
        mismatch = {
            key: {"expected": value, "observed": state.get(key)}
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise ValueError(f"aligned-anchor resume provenance mismatch: {mismatch}")
        self.epoch = int(state["epoch"])
        self.shard_index = int(state["shard_index"])
        self.row_index = int(state["row_index"])
        self.buffer = [dict(value) for value in state["buffer"]]
        self.exhausted = bool(state["exhausted"])
        self.rng = restore_rng_state(str(state["rng_state_b64"]))
        if not 0 <= self.shard_index <= len(self.shards) or self.row_index < 0:
            raise ValueError("aligned-anchor resume cursor lies outside the frozen manifest")

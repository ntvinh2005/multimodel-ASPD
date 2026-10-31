"""Bounded deterministic streaming over frozen aligned-anchor JSONL shards."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO

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
        self.byte_offset = 0
        self.buffer: list[dict[str, Any]] = []
        self.exhausted = False
        self._handle: BinaryIO | None = None
        self._handle_shard_index: int | None = None
        self.raw_lines_read = 0
        self.raw_bytes_read = 0

    def _path(self, row: Mapping[str, Any]) -> Path:
        path = Path(str(row["path"]))
        return path if path.is_absolute() else self.manifest_path.parent / path

    def _verify_shards(self) -> None:
        for row in self.shards:
            path = self._path(row)
            if not path.is_file() or _sha256(path) != row["sha256"]:
                raise ValueError(f"aligned-anchor shard missing or corrupt: {path}")

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._handle_shard_index = None

    def close(self) -> None:
        self._close_handle()

    def __enter__(self) -> AlignedAnchorStream:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _validate_byte_offset(self, path: Path, offset: int) -> None:
        size = path.stat().st_size
        if offset < 0 or offset > size:
            raise ValueError(
                f"aligned-anchor byte offset {offset} lies outside {path} ({size} bytes)"
            )
        if 0 < offset < size:
            with path.open("rb") as handle:
                handle.seek(offset - 1)
                if handle.read(1) != b"\n":
                    raise ValueError(
                        f"aligned-anchor byte offset {offset} is not a UTF-8 JSONL boundary"
                    )

    def _open_current_shard(self) -> BinaryIO:
        if self.shard_index >= len(self.shards):
            raise RuntimeError("cannot open a shard after end-of-stream")
        if self._handle is not None and self._handle_shard_index == self.shard_index:
            return self._handle
        self._close_handle()
        path = self._path(self.shards[self.shard_index])
        self._validate_byte_offset(path, self.byte_offset)
        self._handle = path.open("rb")
        self._handle.seek(self.byte_offset)
        self._handle_shard_index = self.shard_index
        return self._handle

    def _cursor_line_sha256(self) -> str:
        if self.shard_index >= len(self.shards):
            return hashlib.sha256(b"END_OF_STREAM").hexdigest()
        handle = self._open_current_shard()
        position = handle.tell()
        line = handle.readline()
        handle.seek(position)
        return hashlib.sha256(line or b"END_OF_SHARD").hexdigest()

    def _next_raw(self) -> dict[str, Any] | None:
        while self.shard_index < len(self.shards):
            path = self._path(self.shards[self.shard_index])
            handle = self._open_current_shard()
            line = handle.readline()
            if line:
                line_number = self.row_index + 1
                self.byte_offset = handle.tell()
                self.row_index = line_number
                self.raw_lines_read += 1
                self.raw_bytes_read += len(line)
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid UTF-8 JSON aligned-anchor record in {path}:{line_number}"
                    ) from error
                if not record.get("anchor_id") or tuple(record.get("model_order", ())) != (
                    "qwen",
                    "gemma",
                    "phi",
                ):
                    raise ValueError(
                        f"invalid aligned-anchor record in {path}:{line_number}"
                    )
                return record
            self._close_handle()
            self.shard_index += 1
            self.row_index = 0
            self.byte_offset = 0
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
            "schema_version": 4,
            "manifest_sha256": self.manifest_sha256,
            "seed": self.seed,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "epoch": self.epoch,
            "shard_index": self.shard_index,
            "row_index": self.row_index,
            "byte_offset": self.byte_offset,
            "cursor_line_sha256": self._cursor_line_sha256(),
            "buffer": self.buffer,
            "exhausted": self.exhausted,
            "rng_state_b64": encode_rng_state(self.rng),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 4:
            raise ValueError("unsupported aligned-anchor stream state")
        self._close_handle()
        self._verify_shards()
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
        self.byte_offset = int(state["byte_offset"])
        self.buffer = [dict(value) for value in state["buffer"]]
        self.exhausted = bool(state["exhausted"])
        self.rng = restore_rng_state(str(state["rng_state_b64"]))
        if not 0 <= self.shard_index <= len(self.shards) or self.row_index < 0:
            raise ValueError("aligned-anchor resume cursor lies outside the frozen manifest")
        if self.shard_index == len(self.shards):
            if self.row_index != 0 or self.byte_offset != 0:
                raise ValueError("end-of-stream cursor must use zero row and byte offsets")
        else:
            self._validate_byte_offset(
                self._path(self.shards[self.shard_index]), self.byte_offset
            )
        if self._cursor_line_sha256() != state.get("cursor_line_sha256"):
            raise ValueError("aligned-anchor resume byte offset does not match its cursor proof")

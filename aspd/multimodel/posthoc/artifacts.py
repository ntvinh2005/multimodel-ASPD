"""Immutable stage artifacts and provenance.

Layout of one analysis output root::

    <root>/
      importance/        importance.safetensors, importance_meta.json
      representatives/   representatives.json
      edges/             edges.safetensors, edge_records.json, selection.json,
                         ablation_check.json
      compare/           comparisons.jsonl, summary.json
      <stage>/manifest.json   provenance (git, config, backend, environment)
      <stage>/COMPLETE        SHA-256 of every file of the stage

A stage directory is created exclusively and never overwritten (CONTRIBUTING.md:
completed evaluation artifacts are immutable).  A later stage loads a prerequisite only
after checking its ``COMPLETE`` hashes, so stages can run in separate jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from aspd.multimodel.posthoc.edges import EdgeRecord
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def runtime_provenance() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_tracked_changes": None if status is None else bool(status),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_utc": datetime.now(UTC).isoformat(),
    }


class StageDirectory:
    """Exclusive writer for one stage of one output root."""

    def __init__(self, root: Path, stage: str):
        self.path = root / stage
        root.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite completed or partial stage directory {self.path}; "
                "use a new output root"
            ) from None

    def complete(self, manifest: dict[str, Any]) -> None:
        write_json(self.path / "manifest.json", manifest)
        files = sorted(p for p in self.path.iterdir() if p.is_file() and p.name != "COMPLETE")
        write_json(
            self.path / "COMPLETE",
            {"status": "COMPLETE", "sha256": {p.name: sha256_file(p) for p in files}},
        )


def require_complete(root: Path, stage: str) -> Path:
    """Return ``root/stage`` after verifying its ``COMPLETE`` hashes."""

    path = root / stage
    marker = path / "COMPLETE"
    if not marker.is_file():
        raise FileNotFoundError(f"prerequisite stage {stage!r} is not complete under {root}")
    expected = json.loads(marker.read_text(encoding="utf-8"))["sha256"]
    for name, digest in expected.items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"{path / name} changed after its stage completed")
    return path


# ---------------------------------------------------------------- edge storage
def _group_readable(path: Path) -> None:
    """``safetensors`` writes owner-only files; give them the normal umask permissions."""

    umask = os.umask(0)
    os.umask(umask)
    path.chmod(0o666 & ~umask)


def save_edges(directory: Path, records: Sequence[EdgeRecord]) -> None:
    """Store edge records as flat tensors (COO) plus a small JSON vocabulary."""

    matrices = sorted({m for r in records for m in r.matrices})
    matrix_lookup = {m: i for i, m in enumerate(matrices)}
    downstream = sorted({r.downstream for r in records})
    downstream_lookup = {c: i for i, c in enumerate(downstream)}
    anchors = sorted({r.anchor_id for r in records})
    anchor_lookup = {a: i for i, a in enumerate(anchors)}
    up_matrix, up_feature, scores, starts, stops = [], [], [], [], []
    offset = 0
    for record in records:
        local = torch.tensor([matrix_lookup[m] for m in record.matrices], dtype=torch.int32)
        up_matrix.append(local[record.matrix_index.long()] if len(record) else
                         torch.zeros(0, dtype=torch.int32))
        up_feature.append(record.features.to(torch.int32))
        scores.append(record.scores.float())
        starts.append(offset)
        offset += len(record)
        stops.append(offset)
    tensors = {
        "upstream_matrix": torch.cat(up_matrix) if up_matrix else torch.zeros(0, dtype=torch.int32),
        "upstream_feature": torch.cat(up_feature) if up_feature else torch.zeros(0, dtype=torch.int32),
        "score": torch.cat(scores) if scores else torch.zeros(0),
        "record_start": torch.tensor(starts, dtype=torch.int64),
        "record_stop": torch.tensor(stops, dtype=torch.int64),
        "record_coefficient": torch.tensor([r.coefficient for r in records], dtype=torch.float64),
        "record_downstream": torch.tensor([downstream_lookup[r.downstream] for r in records],
                                          dtype=torch.int32),
        "record_anchor": torch.tensor([anchor_lookup[r.anchor_id] for r in records],
                                      dtype=torch.int32),
    }
    save_file(tensors, str(directory / "edges.safetensors"))
    _group_readable(directory / "edges.safetensors")
    write_json(directory / "edge_records.json", {
        "schema_version": 2,
        "matrix_keys": [m.key for m in matrices],
        "downstream_keys": [c.key for c in downstream],
        "anchor_ids": anchors,
        "records": len(records),
    })


class _EdgePart:
    """One stored edge file; records are materialized on demand as tensor views."""

    def __init__(self, directory: Path):
        self.tensors = load_file(str(directory / "edges.safetensors"))
        meta = json.loads((directory / "edge_records.json").read_text(encoding="utf-8"))
        if meta.get("schema_version") != 2:
            raise ValueError(f"{directory}: unsupported edge storage schema")
        self.matrices = tuple(MatrixId.from_key(k) for k in meta["matrix_keys"])
        self.downstream = [ComponentId.from_key(k) for k in meta["downstream_keys"]]
        self.anchor_ids = list(meta["anchor_ids"])
        anchor = self.tensors["record_anchor"].tolist()
        down = self.tensors["record_downstream"].tolist()
        self.index = {(anchor[r], down[r]): r for r in range(len(anchor))}
        self.anchor_lookup = {a: i for i, a in enumerate(self.anchor_ids)}
        self.downstream_lookup = {c: i for i, c in enumerate(self.downstream)}

    def __len__(self) -> int:
        return int(self.tensors["record_start"].numel())

    def record(self, row: int) -> EdgeRecord:
        t = self.tensors
        start, stop = int(t["record_start"][row]), int(t["record_stop"][row])
        return EdgeRecord(
            anchor_id=self.anchor_ids[int(t["record_anchor"][row])],
            downstream=self.downstream[int(t["record_downstream"][row])],
            coefficient=float(t["record_coefficient"][row]),
            matrices=self.matrices,
            matrix_index=t["upstream_matrix"][start:stop],
            features=t["upstream_feature"][start:stop],
            scores=t["score"][start:stop],
        )

    def find(self, anchor_id: str, downstream: ComponentId) -> int | None:
        a = self.anchor_lookup.get(anchor_id)
        d = self.downstream_lookup.get(downstream)
        if a is None or d is None:
            return None
        return self.index.get((a, d))


class EdgeTable:
    """All edge records of a run (one file, or every shard of a sharded edge stage)."""

    def __init__(self, parts: Sequence[_EdgePart]):
        self.parts = list(parts)

    def __len__(self) -> int:
        return sum(len(p) for p in self.parts)

    def __iter__(self):
        for part in self.parts:
            for row in range(len(part)):
                yield part.record(row)

    def get(self, anchor_id: str, downstream: ComponentId) -> EdgeRecord | None:
        for part in self.parts:
            row = part.find(anchor_id, downstream)
            if row is not None:
                return part.record(row)
        return None


def load_edges(directory: Path) -> list[EdgeRecord]:
    """All records of one stored edge file (small runs and tests)."""

    return list(EdgeTable([_EdgePart(directory)]))


def shard_stage_name(index: int, count: int) -> str:
    return f"edges-shard-{index:03d}-of-{count:03d}"


def load_edge_table(root: Path) -> EdgeTable:
    """Verified edge records of ``root``: the ``edges`` stage or the shards it lists."""

    edges_dir = require_complete(root, "edges")
    shards_file = edges_dir / "shards.json"
    if not shards_file.is_file():
        return EdgeTable([_EdgePart(edges_dir)])
    listing = json.loads(shards_file.read_text(encoding="utf-8"))
    parts = []
    for entry in listing["shards"]:
        shard_dir = require_complete(root, entry["stage"])
        if sha256_file(shard_dir / "COMPLETE") != entry["complete_sha256"]:
            raise ValueError(f"{shard_dir} changed after the edge shards were finalized")
        parts.append(_EdgePart(shard_dir))
    return EdgeTable(parts)

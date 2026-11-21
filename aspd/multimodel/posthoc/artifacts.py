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
def save_edges(directory: Path, records: Sequence[EdgeRecord]) -> None:
    """Store edge records as COO tensors plus a JSON index."""

    matrices = sorted({c.matrix for r in records for c in r.upstream})
    matrix_index = {m: i for i, m in enumerate(matrices)}
    rows, starts = [], 0
    upstream_matrix, upstream_feature, scores = [], [], []
    for record in records:
        n = len(record.upstream)
        rows.append(
            {
                "anchor_id": record.anchor_id,
                "downstream": record.downstream.key,
                "coefficient": record.coefficient,
                "start": starts,
                "stop": starts + n,
            }
        )
        upstream_matrix.extend(matrix_index[c.matrix] for c in record.upstream)
        upstream_feature.extend(c.feature for c in record.upstream)
        scores.append(record.scores.float())
        starts += n
    tensors = {
        "upstream_matrix": torch.tensor(upstream_matrix, dtype=torch.int32),
        "upstream_feature": torch.tensor(upstream_feature, dtype=torch.int32),
        "score": torch.cat(scores) if scores else torch.zeros(0),
    }
    save_file(tensors, str(directory / "edges.safetensors"))
    write_json(
        directory / "edge_records.json",
        {"matrix_keys": [m.key for m in matrices], "records": rows},
    )


def load_edges(directory: Path) -> list[EdgeRecord]:
    tensors = load_file(str(directory / "edges.safetensors"))
    index = json.loads((directory / "edge_records.json").read_text(encoding="utf-8"))
    matrices = [MatrixId.from_key(key) for key in index["matrix_keys"]]
    up_m = tensors["upstream_matrix"].tolist()
    up_f = tensors["upstream_feature"].tolist()
    records = []
    for row in index["records"]:
        start, stop = row["start"], row["stop"]
        records.append(
            EdgeRecord(
                anchor_id=row["anchor_id"],
                downstream=ComponentId.from_key(row["downstream"]),
                coefficient=float(row["coefficient"]),
                upstream=tuple(
                    ComponentId(matrices[up_m[k]], up_f[k]) for k in range(start, stop)
                ),
                scores=tensors["score"][start:stop].clone(),
            )
        )
    return records

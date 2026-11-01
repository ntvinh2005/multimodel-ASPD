"""Atomic per-owner DCP checkpoints for the heterogeneous three-rank 2B trainer."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from aspd.multimodel.v3.distributed_checkpoint import (
    _BoundedFileSystemWriter,
    _sha256,
    _write_json_fsync,
    capture_rng_state,
    restore_rng_state,
)


class ModelOwnedCheckpoint:
    """One independent DCP shard per model owner plus an atomic global marker."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path, *, keep_complete: int = 2):
        if keep_complete < 2:
            raise ValueError("model-owned training retains at least two checkpoints")
        self.root = Path(root)
        self.keep_complete = keep_complete

    @staticmethod
    def _rank_world() -> tuple[int, int]:
        if not dist.is_initialized():
            raise RuntimeError("model-owned checkpointing requires distributed")
        return dist.get_rank(), dist.get_world_size()

    def _paths(self, step: int) -> tuple[Path, Path]:
        name = f"step-{step:09d}"
        return self.root / f".{name}.incomplete", self.root / name

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        runtime_state: Mapping[str, Any],
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
        model_key: str,
    ) -> dict[str, Any]:
        rank, world = self._rank_world()
        incomplete, complete = self._paths(step)
        if rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            if incomplete.exists() or complete.exists():
                raise FileExistsError(f"refusing to overwrite checkpoint step {step}")
            incomplete.mkdir()
        dist.barrier()
        rank_root = incomplete / f"rank-{rank:05d}"
        rank_root.mkdir()
        (rank_root / "dcp").mkdir()
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        writer = _BoundedFileSystemWriter(
            rank_root / "dcp",
            single_file_per_rank=False,
            sync_files=True,
            thread_count=1,
            per_thread_copy_ahead=10_000_000,
            overwrite=False,
        )
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            storage_writer=writer,
            no_dist=True,
        )
        sidecar = {
            "schema_version": self.SCHEMA_VERSION,
            "rank": rank,
            "world_size": world,
            "model_key": model_key,
            "step": step,
            "config_fingerprint": config_fingerprint,
            "config_file_sha256": config_file_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "rng": capture_rng_state(),
            "scheduler_state": scheduler.state_dict(),
            "runtime_state": dict(runtime_state),
        }
        _write_json_fsync(rank_root / "sidecar.json", sidecar)
        dist.barrier()
        if rank == 0:
            rank_roots = [incomplete / f"rank-{value:05d}" for value in range(world)]
            missing = [
                str(path)
                for path in rank_roots
                if not (path / "dcp" / ".metadata").is_file()
                or not (path / "sidecar.json").is_file()
            ]
            if missing:
                raise RuntimeError(f"model-owned checkpoint ranks incomplete: {missing}")
            files = [path for path in incomplete.rglob("*") if path.is_file()]
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "status": "COMPLETE",
                "step": step,
                "world_size": world,
                "config_fingerprint": config_fingerprint,
                "config_file_sha256": config_file_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "rank_sidecar_sha256": {
                    path.name: _sha256(path / "sidecar.json") for path in rank_roots
                },
                "rank_dcp_metadata_sha256": {
                    path.name: _sha256(path / "dcp" / ".metadata") for path in rank_roots
                },
                "file_count_before_manifest": len(files),
                "bytes_before_manifest": sum(path.stat().st_size for path in files),
            }
            _write_json_fsync(incomplete / "manifest.json", manifest)
            _write_json_fsync(
                incomplete / "COMPLETE",
                {"manifest_sha256": _sha256(incomplete / "manifest.json")},
            )
            os.replace(incomplete, complete)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        dist.barrier()
        return json.loads((complete / "manifest.json").read_text(encoding="utf-8"))

    def _validate(
        self,
        path: Path,
        *,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
    ) -> dict[str, Any]:
        _rank, world = self._rank_world()
        manifest_path = path / "manifest.json"
        marker_path = path / "COMPLETE"
        if not manifest_path.is_file() or not marker_path.is_file():
            raise ValueError("model-owned checkpoint lacks its atomic COMPLETE marker")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("manifest_sha256") != _sha256(manifest_path):
            raise ValueError("model-owned checkpoint manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "status": "COMPLETE",
            "world_size": world,
            "config_fingerprint": config_fingerprint,
            "config_file_sha256": config_file_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        }
        mismatch = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatch:
            raise ValueError(f"model-owned checkpoint provenance mismatch: {mismatch}")
        return manifest

    def load(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
        expected_model_key: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint = Path(path)
        rank, _world = self._rank_world()
        manifest = self._validate(
            checkpoint,
            config_fingerprint=config_fingerprint,
            config_file_sha256=config_file_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        rank_root = checkpoint / f"rank-{rank:05d}"
        sidecar = json.loads((rank_root / "sidecar.json").read_text(encoding="utf-8"))
        if sidecar.get("model_key") != expected_model_key:
            raise ValueError("checkpoint rank/model ownership changed")
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=rank_root / "dcp", no_dist=True)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )
        scheduler.load_state_dict(sidecar["scheduler_state"])
        restore_rng_state(sidecar["rng"])
        dist.barrier()
        return sidecar["runtime_state"], manifest

    def complete_checkpoints(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            path
            for path in self.root.glob("step-*")
            if path.is_dir() and (path / "COMPLETE").is_file()
        )

    def prune_after_verified_replacement(self) -> list[Path]:
        rank, _world = self._rank_world()
        removed: list[Path] = []
        if rank == 0:
            for path in self.complete_checkpoints()[: -self.keep_complete]:
                shutil.rmtree(path)
                removed.append(path)
        dist.barrier()
        return removed

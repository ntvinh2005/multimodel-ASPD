"""Atomic per-owner DCP checkpoints for the heterogeneous three-rank 2B trainer."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
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

    @staticmethod
    def _snapshot_sqlite(source: Path, destination: Path) -> None:
        """Create a transactionally consistent, fsynced SQLite recovery snapshot."""

        if not source.is_file():
            raise FileNotFoundError(f"checkpoint recovery database is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        destination_connection = sqlite3.connect(destination)
        try:
            source_connection.backup(destination_connection)
            result = destination_connection.execute("PRAGMA quick_check").fetchone()
            if result != ("ok",):
                raise ValueError(
                    f"checkpoint recovery database failed quick_check: {destination}"
                )
            destination_connection.commit()
        finally:
            destination_connection.close()
            source_connection.close()
        file_descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)

    @staticmethod
    def _copy_sqlite_exact(source: Path, destination: Path) -> None:
        """Copy an immutable SQLite snapshot byte-for-byte and verify readability."""

        shutil.copyfile(source, destination)
        connection = sqlite3.connect(f"file:{destination}?mode=ro", uri=True)
        try:
            result = connection.execute("PRAGMA quick_check").fetchone()
            if result != ("ok",):
                raise ValueError(f"restored SQLite database failed quick_check: {destination}")
        finally:
            connection.close()
        file_descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)

    @staticmethod
    def _validate_recovery_names(values: Mapping[str, str | Path]) -> None:
        for name in values:
            path = Path(name)
            if path.name != name or name not in {
                "exposure_ledger.sqlite",
                "stream_dedup.sqlite",
            }:
                raise ValueError(f"unsupported checkpoint recovery database name: {name}")

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
        experiment_metadata: Mapping[str, Any] | None = None,
        recovery_databases: Mapping[str, str | Path] | None = None,
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
            "experiment_metadata": dict(experiment_metadata or {}),
            "rng": capture_rng_state(),
            "scheduler_state": scheduler.state_dict(),
            "runtime_state": dict(runtime_state),
        }
        _write_json_fsync(rank_root / "sidecar.json", sidecar)
        dist.barrier()
        if rank == 0:
            recovery_databases = dict(recovery_databases or {})
            self._validate_recovery_names(recovery_databases)
            recovery_root = incomplete / "recovery"
            for name, source in recovery_databases.items():
                self._snapshot_sqlite(Path(source), recovery_root / name)
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
                "experiment_metadata": dict(experiment_metadata or {}),
                "rank_sidecar_sha256": {
                    path.name: _sha256(path / "sidecar.json") for path in rank_roots
                },
                "rank_dcp_metadata_sha256": {
                    path.name: _sha256(path / "dcp" / ".metadata") for path in rank_roots
                },
                "recovery_database_sha256": {
                    name: _sha256(recovery_root / name)
                    for name in sorted(recovery_databases)
                },
                "recovery_database_bytes": {
                    name: (recovery_root / name).stat().st_size
                    for name in sorted(recovery_databases)
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
        expected_world_size: int | None = None,
    ) -> dict[str, Any]:
        if expected_world_size is None:
            _rank, world = self._rank_world()
        else:
            world = int(expected_world_size)
            if world <= 0:
                raise ValueError("expected checkpoint world size must be positive")
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
        for rank_name, digest in manifest["rank_sidecar_sha256"].items():
            sidecar = path / rank_name / "sidecar.json"
            if not sidecar.is_file() or _sha256(sidecar) != digest:
                raise ValueError(f"checkpoint sidecar is missing or corrupt: {rank_name}")
        for rank_name, digest in manifest["rank_dcp_metadata_sha256"].items():
            metadata = path / rank_name / "dcp" / ".metadata"
            if not metadata.is_file() or _sha256(metadata) != digest:
                raise ValueError(f"checkpoint DCP metadata is missing or corrupt: {rank_name}")
        recovery = manifest.get("recovery_database_sha256", {})
        if not isinstance(recovery, dict):
            raise ValueError("checkpoint recovery-database manifest is invalid")
        for name, digest in recovery.items():
            self._validate_recovery_names({name: name})
            database = path / "recovery" / name
            if not database.is_file() or _sha256(database) != digest:
                raise ValueError(f"checkpoint recovery database is missing or corrupt: {name}")
        return manifest

    def restore_recovery_databases(
        self,
        *,
        path: str | Path,
        destination_root: str | Path,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
        expected_world_size: int,
    ) -> dict[str, Any]:
        """Atomically materialize exact ledger/stream state for an external resume.

        Existing destinations are never overwritten.  This keeps the source checkpoint
        immutable and prevents an accidental mixture of two recovery histories.
        """

        checkpoint = Path(path)
        manifest = self._validate(
            checkpoint,
            config_fingerprint=config_fingerprint,
            config_file_sha256=config_file_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            expected_world_size=expected_world_size,
        )
        recovery = manifest.get("recovery_database_sha256", {})
        required = {"exposure_ledger.sqlite", "stream_dedup.sqlite"}
        if set(recovery) != required:
            raise ValueError(
                "external exact resume requires both immutable recovery databases"
            )
        destination_root = Path(destination_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        for name in sorted(required):
            destination = destination_root / name
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing resume database: {destination}"
                )
            temporary = destination_root / f".{name}.restore-incomplete"
            if temporary.exists():
                raise FileExistsError(f"stale incomplete resume database: {temporary}")
            self._copy_sqlite_exact(checkpoint / "recovery" / name, temporary)
            if _sha256(temporary) != recovery[name]:
                temporary.unlink()
                raise ValueError(f"restored recovery database checksum mismatch: {name}")
            os.replace(temporary, destination)
        directory_fd = os.open(destination_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
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

    def load_model_only(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
        expected_model_key: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore frozen ASPD parameters without allocating Adam or scheduler state.

        This is the read-only diagnostic path.  It intentionally does not restore RNG,
        optimizer, scheduler, or stream state and therefore cannot be used to resume
        training.
        """

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
        if int(sidecar.get("step", -1)) != int(manifest["step"]):
            raise ValueError("checkpoint sidecar/global step mismatch")
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        state = {"model": get_model_state_dict(model, options=options)}
        dcp.load(state, checkpoint_id=rank_root / "dcp", no_dist=True)
        set_model_state_dict(model, state["model"], options=options)
        dist.barrier()
        return dict(sidecar["runtime_state"]), manifest

    def load_model_only_owner(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
        checkpoint_rank: int,
        expected_checkpoint_world_size: int,
        expected_model_key: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load one historical owner shard under a different evaluation world size.

        This read-only path supports colocating multiple model owners for diagnostic
        inference. It preserves the checkpoint's original three-rank provenance and
        never loads optimizer, scheduler, RNG, or stream state.
        """

        checkpoint = Path(path)
        if not 0 <= checkpoint_rank < expected_checkpoint_world_size:
            raise ValueError("checkpoint owner rank is out of range")
        manifest = self._validate(
            checkpoint,
            config_fingerprint=config_fingerprint,
            config_file_sha256=config_file_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            expected_world_size=expected_checkpoint_world_size,
        )
        rank_root = checkpoint / f"rank-{checkpoint_rank:05d}"
        sidecar = json.loads((rank_root / "sidecar.json").read_text(encoding="utf-8"))
        if int(sidecar.get("rank", -1)) != checkpoint_rank:
            raise ValueError("checkpoint sidecar owner rank changed")
        if sidecar.get("model_key") != expected_model_key:
            raise ValueError("checkpoint rank/model ownership changed")
        if int(sidecar.get("step", -1)) != int(manifest["step"]):
            raise ValueError("checkpoint sidecar/global step mismatch")
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        state = {"model": get_model_state_dict(model, options=options)}
        dcp.load(state, checkpoint_id=rank_root / "dcp", no_dist=True)
        set_model_state_dict(model, state["model"], options=options)
        return dict(sidecar["runtime_state"]), manifest

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

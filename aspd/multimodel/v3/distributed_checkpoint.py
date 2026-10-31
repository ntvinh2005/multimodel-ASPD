"""Atomic sharded checkpoints for the ASPD v3 main-v1 trainable state."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import random
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _pickle_b64(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode("ascii")


def _unpickle_b64(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_b64": _pickle_b64(random.getstate()),
        "torch_cpu_b64": base64.b64encode(bytes(torch.get_rng_state().tolist())).decode("ascii"),
    }
    try:
        import numpy as np

        state["numpy_b64"] = _pickle_b64(np.random.get_state())
    except ImportError:
        state["numpy_b64"] = None
    if torch.cuda.is_available():
        state["torch_cuda_b64"] = [
            base64.b64encode(bytes(value.tolist())).decode("ascii")
            for value in torch.cuda.get_rng_state_all()
        ]
    else:
        state["torch_cuda_b64"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(_unpickle_b64(str(state["python_b64"])))
    torch.set_rng_state(
        torch.tensor(
            list(base64.b64decode(str(state["torch_cpu_b64"]).encode("ascii"))),
            dtype=torch.uint8,
        )
    )
    if state.get("numpy_b64") is not None:
        import numpy as np

        np.random.set_state(_unpickle_b64(str(state["numpy_b64"])))
    cuda_values = state.get("torch_cuda_b64", [])
    if cuda_values:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [
                torch.tensor(
                    list(base64.b64decode(str(value).encode("ascii"))),
                    dtype=torch.uint8,
                )
                for value in cuda_values
            ]
        )


def _sharded_state_dict_options() -> StateDictOptions:
    """Keep DTensor shards on device so synchronous DCP can copy them in bounded chunks."""

    return StateDictOptions(full_state_dict=False, cpu_offload=False)


class _BoundedFileSystemWriter(dcp.FileSystemWriter):
    """Synchronous writer that streams GPU shards instead of staging all state in RAM."""

    def stage(self, state_dict):
        # ``dcp.save`` calls ``stage`` even for a synchronous save. The public
        # FileSystemWriter otherwise copies the complete model and Adam state to host
        # RAM first. Save is synchronous here, so the live state cannot mutate while
        # the writer's bounded copy-ahead path serializes it.
        return state_dict


class MainV1DistributedCheckpoint:
    """DCP model/Adam shards plus per-rank exact-resume sidecars."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path, *, keep_complete: int = 2):
        if keep_complete < 2:
            raise ValueError("main-v1 must retain at least two complete checkpoints")
        self.root = Path(root)
        self.keep_complete = keep_complete

    @staticmethod
    def _rank_world() -> tuple[int, int, bool]:
        initialized = dist.is_available() and dist.is_initialized()
        return (
            dist.get_rank() if initialized else 0,
            dist.get_world_size() if initialized else 1,
            initialized,
        )

    @staticmethod
    def _barrier(enabled: bool) -> None:
        if enabled:
            dist.barrier()

    def _paths(self, step: int) -> tuple[Path, Path]:
        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        name = f"step-{step:09d}"
        return self.root / f".{name}.incomplete", self.root / name

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        runtime_state: Mapping[str, Any],
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
    ) -> dict[str, Any]:
        rank, world_size, distributed = self._rank_world()
        incomplete, complete = self._paths(step)
        if rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            if incomplete.exists() or complete.exists():
                raise FileExistsError(f"refusing to overwrite checkpoint step {step}")
            incomplete.mkdir()
            (incomplete / "dcp").mkdir()
        self._barrier(distributed)

        options = _sharded_state_dict_options()
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        writer = _BoundedFileSystemWriter(
            incomplete / "dcp",
            # The PyTorch 2.8 single-file path retains every already-written CPU
            # tensor until the rank file closes. One file per tensor releases that
            # buffer after each bounded write.
            single_file_per_rank=False,
            sync_files=True,
            thread_count=1,
            per_thread_copy_ahead=10_000_000,
            overwrite=False,
        )
        dcp.save(
            state,
            storage_writer=writer,
            no_dist=not distributed,
        )
        rank_payload = {
            "schema_version": self.SCHEMA_VERSION,
            "rank": rank,
            "world_size": world_size,
            "step": step,
            "config_fingerprint": config_fingerprint,
            "config_file_sha256": config_file_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "rng": capture_rng_state(),
            "scheduler_state": scheduler.state_dict(),
            "runtime_state": dict(runtime_state),
        }
        rank_path = incomplete / f"rank-{rank:05d}.json"
        _write_json_fsync(rank_path, rank_payload)
        self._barrier(distributed)

        if rank == 0:
            rank_files = [incomplete / f"rank-{value:05d}.json" for value in range(world_size)]
            missing = [str(path) for path in rank_files if not path.is_file()]
            metadata_path = incomplete / "dcp" / ".metadata"
            if missing or not metadata_path.is_file():
                raise RuntimeError(
                    f"distributed checkpoint is incomplete: missing_rank_files={missing}, "
                    f"dcp_metadata={metadata_path.is_file()}"
                )
            files = [path for path in incomplete.rglob("*") if path.is_file()]
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "status": "COMPLETE",
                "step": step,
                "world_size": world_size,
                "config_fingerprint": config_fingerprint,
                "config_file_sha256": config_file_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "rank_sidecar_sha256": {
                    path.name: _sha256(path) for path in rank_files
                },
                "dcp_metadata_sha256": _sha256(metadata_path),
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
        self._barrier(distributed)
        return json.loads((complete / "manifest.json").read_text(encoding="utf-8"))

    def _validate_complete(
        self,
        path: Path,
        *,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
        world_size: int,
    ) -> dict[str, Any]:
        manifest_path = path / "manifest.json"
        marker_path = path / "COMPLETE"
        if not manifest_path.is_file() or not marker_path.is_file():
            raise ValueError(f"checkpoint is not atomically complete: {path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("manifest_sha256") != _sha256(manifest_path):
            raise ValueError("checkpoint COMPLETE marker does not match its manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "status": "COMPLETE",
            "world_size": world_size,
            "config_fingerprint": config_fingerprint,
            "config_file_sha256": config_file_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        }
        mismatches = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint provenance mismatch: {mismatches}")
        metadata = path / "dcp" / ".metadata"
        if not metadata.is_file() or _sha256(metadata) != manifest["dcp_metadata_sha256"]:
            raise ValueError("checkpoint DCP metadata is missing or corrupt")
        for name, digest in manifest["rank_sidecar_sha256"].items():
            sidecar = path / name
            if not sidecar.is_file() or _sha256(sidecar) != digest:
                raise ValueError(f"checkpoint rank sidecar is missing or corrupt: {name}")
        return manifest

    def load(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config_fingerprint: str,
        config_file_sha256: str,
        dataset_manifest_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint = Path(path)
        rank, world_size, distributed = self._rank_world()
        manifest = self._validate_complete(
            checkpoint,
            config_fingerprint=config_fingerprint,
            config_file_sha256=config_file_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            world_size=world_size,
        )
        options = _sharded_state_dict_options()
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=checkpoint / "dcp", no_dist=not distributed)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )
        sidecar = json.loads(
            (checkpoint / f"rank-{rank:05d}.json").read_text(encoding="utf-8")
        )
        scheduler.load_state_dict(sidecar["scheduler_state"])
        restore_rng_state(sidecar["rng"])
        return sidecar["runtime_state"], manifest

    def complete_checkpoints(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        values = []
        for path in self.root.glob("step-*"):
            if path.is_dir() and (path / "COMPLETE").is_file():
                values.append(path)
        return sorted(values)

    def prune_after_verified_replacement(self) -> list[Path]:
        """Delete only complete checkpoints older than the configured rollback window."""

        rank, _, distributed = self._rank_world()
        removed: list[Path] = []
        if rank == 0:
            complete = self.complete_checkpoints()
            for path in complete[: -self.keep_complete]:
                shutil.rmtree(path)
                removed.append(path)
        self._barrier(distributed)
        return removed

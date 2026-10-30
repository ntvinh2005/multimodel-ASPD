"""Bounded three-GPU streaming support for the frozen original-ASPD recipe.

Rank 0 is the activation producer. Ranks 1 and 2 are the only DDP training
ranks, each consuming four 512-token sequences as one 2,048-token BatchTopK
pool. Blocking transfers provide deterministic backpressure and never persist a
100M-token activation cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from aspd.multimodel.cache import (
    _Capture,
    _register_capture_hooks,
    load_effective_model,
    snapshot_model_weights,
)
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.mix3 import (
    Mix3ASPDSource,
    load_r0_mix_config,
    mix3_cursor_for_step,
    validate_mix3_recipe_config,
)
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_recipe import (
    StreamCursor,
    validate_dataset_manifest,
    validate_original_recipe_config,
)

try:
    import resource
except ModuleNotFoundError:
    resource = None  # type: ignore[assignment]


def _peak_cpu_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource else 0


def resolved_config_sha256(cfg: MultiModelExperimentConfig) -> str:
    payload = json.dumps(
        cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FrozenTokenShardStream:
    """Random-access logical stream over immutable token-only shards."""

    def __init__(self, manifest_path: str | Path, split: str):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sequence_length = int(manifest["sequence_length"])
        self.shards = [row for row in manifest["shards"] if row["split"] == split]
        self.shards.sort(key=lambda row: int(row["sequence_start"]))
        expected = 0
        for row in self.shards:
            if int(row["sequence_start"]) != expected:
                raise ValueError(f"non-contiguous {split} token shards at sequence {expected}")
            expected += int(row["sequence_count"])
        self.sequence_count = expected
        self._cached_path: Path | None = None
        self._cached_payload: dict[str, Tensor] | None = None

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest_path.parent / path

    def _load(self, row: dict[str, Any]) -> dict[str, Tensor]:
        path = self._path(row["path"])
        if self._cached_path != path:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["input_ids"].shape != payload["valid_tokens"].shape:
                raise ValueError(f"token/mask shape mismatch in {path}")
            if payload["input_ids"].shape[1] != self.sequence_length:
                raise ValueError(f"sequence width mismatch in {path}")
            if not payload["valid_tokens"].bool().all():
                raise ValueError(f"frozen exposure shards must contain full valid sequences: {path}")
            if "provenance_ids" in payload and payload["provenance_ids"].shape != (
                payload["input_ids"].shape[0],
            ):
                raise ValueError(f"per-sequence provenance shape mismatch in {path}")
            self._cached_path = path
            self._cached_payload = payload
        assert self._cached_payload is not None
        return self._cached_payload

    def batch_at(self, sequence_index: int, count: int) -> dict[str, Tensor]:
        if sequence_index < 0 or count <= 0 or sequence_index + count > self.sequence_count:
            raise IndexError("requested token batch lies outside the frozen stream")
        ids: list[Tensor] = []
        valid: list[Tensor] = []
        provenance: list[Tensor] = []
        cursor = sequence_index
        remaining = count
        for row in self.shards:
            start = int(row["sequence_start"])
            stop = start + int(row["sequence_count"])
            if not start <= cursor < stop:
                continue
            payload = self._load(row)
            local = cursor - start
            take = min(remaining, stop - cursor)
            ids.append(payload["input_ids"][local:local + take])
            valid.append(payload["valid_tokens"][local:local + take])
            if "provenance_ids" in payload:
                provenance.append(payload["provenance_ids"][local:local + take])
            else:
                provenance.append(torch.full((take,), -1, dtype=torch.long))
            cursor += take
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("frozen stream ended inside a requested batch")
        return {
            "input_ids": torch.cat(ids),
            "valid_tokens": torch.cat(valid),
            "sequence_ids": torch.arange(sequence_index, sequence_index + count),
            "provenance_ids": torch.cat(provenance),
        }


def split_global_batch(
    batch: dict[str, Tensor], world_size: int, sequences_per_rank: int
) -> list[dict[str, Tensor]]:
    """Split only on sequences so every training rank retains one full local pool."""

    expected = world_size * sequences_per_rank
    if batch["input_ids"].shape[0] != expected:
        raise ValueError(f"global batch has {batch['input_ids'].shape[0]} rows, expected {expected}")
    return [
        {
            key: value[index * sequences_per_rank:(index + 1) * sequences_per_rank]
            for key, value in batch.items()
        }
        for index in range(world_size)
    ]


@dataclass(frozen=True)
class DistributedTopology:
    producer_rank: int = 0
    trainer_ranks: tuple[int, int] = (1, 2)
    sequences_per_trainer: int = 4
    sequence_length: int = 512

    @property
    def positions_per_trainer(self) -> int:
        return self.sequences_per_trainer * self.sequence_length

    @property
    def global_positions(self) -> int:
        return len(self.trainer_ranks) * self.positions_per_trainer

    def validate(self, cfg: MultiModelExperimentConfig) -> None:
        if cfg.training.producer_gpus != 1 or cfg.training.training_world_size != 2:
            raise ValueError("frozen topology requires one producer and two training ranks")
        if cfg.training.sequences_per_rank != self.sequences_per_trainer:
            raise ValueError("training-rank batch differs from the frozen topology")
        if cfg.data.sequence_length != self.sequence_length:
            raise ValueError("sequence length differs from the frozen topology")
        if self.positions_per_trainer != cfg.sparsity.pool_tokens:
            raise ValueError("each training rank must contain one complete BatchTopK pool")


def atomic_torch_save(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(
    cfg: MultiModelExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    sampler_state_by_rank: dict[int, dict[str, int]],
) -> dict[str, Any]:
    """Complete deterministic state; the stream cursor forbids replay on resume."""

    cursor = StreamCursor.for_step(cfg, step)
    if set(sampler_state_by_rank) != {0, 1}:
        raise ValueError("checkpoint requires sampler state for both training ranks")
    payload = {
        "step": step,
        "stream_cursor": cursor.state_dict(),
        "sampler_state_by_rank": sampler_state_by_rank,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "config_sha256": resolved_config_sha256(cfg),
    }
    if cfg.mix3_aspd is not None:
        mix = load_r0_mix_config(cfg.mix3_aspd.mix_config)
        payload["mix3_source_cursor"] = mix3_cursor_for_step(mix, step)
    return payload


def restore_checkpoint_payload(
    cfg: MultiModelExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
) -> tuple[int, dict[int, dict[str, int]]]:
    if payload["config_sha256"] != resolved_config_sha256(cfg):
        raise ValueError("resume config differs from the checkpoint")
    cursor = StreamCursor(**payload["stream_cursor"])
    cursor.validate(cfg)
    if cfg.mix3_aspd is not None:
        mix = load_r0_mix_config(cfg.mix3_aspd.mix_config)
        expected_mix3 = mix3_cursor_for_step(mix, cursor.optimizer_step)
        if payload.get("mix3_source_cursor") != expected_mix3:
            raise ValueError("resume checkpoint Mix3 role cursors are missing or inconsistent")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])
    sampler = {int(key): value for key, value in payload["sampler_state_by_rank"].items()}
    if set(sampler) != {0, 1}:
        raise ValueError("resume checkpoint is missing a training-rank sampler")
    for trainer_rank in (0, 1):
        expected_sequence = cursor.sequence_index + trainer_rank * (
            cfg.training.sequences_per_rank or 0
        )
        if sampler[trainer_rank].get("sequence_index") != expected_sequence:
            raise ValueError("resume sampler cursor would replay or skip a rank shard")
    return cursor.optimizer_step, sampler


def _send(tensor: Tensor, destination: int) -> None:
    dist.send(tensor.contiguous(), dst=destination)


def _receive(
    shape: tuple[int, ...], dtype: torch.dtype, source: int, device: int | torch.device
) -> Tensor:
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=source)
    return tensor


def _send_weights(
    weights: list[dict[str, Tensor]], destination: int, device: int | torch.device
) -> None:
    for item in weights:
        matrix = item["o_proj"].to(device=device, dtype=torch.bfloat16)
        _send(torch.tensor(matrix.shape, dtype=torch.long, device=device), destination)
        _send(matrix, destination)


def _receive_weights(source: int, device: int | torch.device) -> list[dict[str, Tensor]]:
    result: list[dict[str, Tensor]] = []
    for _ in range(2):
        shape = _receive((2,), torch.long, source, device)
        result.append(
            {
                "o_proj": _receive(
                    (int(shape[0].item()), int(shape[1].item())),
                    torch.bfloat16,
                    source,
                    device,
                )
            }
        )
    return result


def _send_activation_batch(batch: dict[str, object], destination: int) -> None:
    _send(batch["input_ids"], destination)  # type: ignore[arg-type]
    _send(batch["sequence_ids"], destination)  # type: ignore[arg-type]
    _send(batch["provenance_ids"], destination)  # type: ignore[arg-type]
    valid: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
    _send(valid.to(torch.uint8), destination)
    if "position_provenance_ids" in batch:
        _send(batch["position_provenance_ids"], destination)  # type: ignore[arg-type]
    for residual in batch["R"]:  # type: ignore[union-attr]
        _send(residual, destination)
    for inputs in batch["X"]:  # type: ignore[union-attr]
        _send(inputs["o_proj"], destination)


def _receive_activation_batch(
    cfg: MultiModelExperimentConfig,
    width: int,
    source: int,
    device: int | torch.device,
) -> dict[str, object]:
    local_batch = cfg.training.sequences_per_rank
    assert local_batch is not None
    grid = (local_batch, cfg.data.sequence_length)
    shape = (*grid, width)
    batch: dict[str, object] = {
        "input_ids": _receive(grid, torch.long, source, device),
        "sequence_ids": _receive((local_batch,), torch.long, source, device),
        "provenance_ids": _receive((local_batch,), torch.long, source, device),
        "valid_tokens": _receive(grid, torch.uint8, source, device).bool(),
    }
    if cfg.mix3_aspd is not None:
        batch["position_provenance_ids"] = _receive(grid, torch.long, source, device)
    batch["R"] = [
        _receive(shape, torch.bfloat16, source, device),
        _receive(shape, torch.bfloat16, source, device),
    ]
    batch["X"] = [
        {"o_proj": _receive(shape, torch.bfloat16, source, device)},
        {"o_proj": _receive(shape, torch.bfloat16, source, device)},
    ]
    return batch


@torch.inference_mode()
def _capture_pair(
    models: list[torch.nn.Module],
    captures: list[_Capture],
    input_ids: Tensor,
    valid_tokens: Tensor,
    device: int | torch.device,
    *,
    sequence_ids: Tensor | None = None,
    provenance_ids: Tensor | None = None,
) -> dict[str, object]:
    canonical = input_ids.detach().cpu().clone()
    residuals: list[Tensor] = []
    inputs: list[dict[str, Tensor]] = []
    for model, capture in zip(models, captures, strict=True):
        ids_device = canonical.clone().to(device)
        mask_device = valid_tokens.clone().to(device)
        model(input_ids=ids_device, attention_mask=mask_device, use_cache=False)
        residuals.append(capture.pop("R").to(torch.bfloat16))
        inputs.append({"o_proj": capture.pop("X/o_proj").to(torch.bfloat16)})
    if not torch.equal(canonical, input_ids.cpu()):
        raise AssertionError("canonical token IDs changed between paired forwards")
    rows = input_ids.shape[0]
    return {
        "input_ids": canonical.to(device),
        "sequence_ids": (
            torch.arange(rows, dtype=torch.long) if sequence_ids is None else sequence_ids
        ).to(device),
        "provenance_ids": (
            torch.full((rows,), -1, dtype=torch.long)
            if provenance_ids is None
            else provenance_ids
        ).to(device),
        "valid_tokens": valid_tokens.to(device),
        "R": residuals,
        "X": inputs,
    }


def _rank_slice(
    paired: dict[str, object], local_rank: int, sequences_per_rank: int
) -> dict[str, object]:
    start = local_rank * sequences_per_rank
    stop = start + sequences_per_rank
    result: dict[str, object] = {
        "input_ids": paired["input_ids"][start:stop],  # type: ignore[index]
        "sequence_ids": paired["sequence_ids"][start:stop],  # type: ignore[index]
        "provenance_ids": paired["provenance_ids"][start:stop],  # type: ignore[index]
        "valid_tokens": paired["valid_tokens"][start:stop],  # type: ignore[index]
        "R": [value[start:stop] for value in paired["R"]],  # type: ignore[index]
        "X": [
            {"o_proj": value["o_proj"][start:stop]}
            for value in paired["X"]  # type: ignore[union-attr]
        ],
    }
    if "position_provenance_ids" in paired:
        result["position_provenance_ids"] = paired["position_provenance_ids"][start:stop]  # type: ignore[index]
    return result


def _resume_step(output_dir: Path, resume: str | None) -> int:
    if resume is None:
        return 0
    if resume == "auto":
        pointer = output_dir / "latest_checkpoint.pt"
        if not pointer.is_file():
            raise FileNotFoundError(f"resume pointer not found: {pointer}")
        record = torch.load(pointer, map_location="cpu", weights_only=True)
        step = int(record["step"])
        for trainer_rank, expected_hash in enumerate(record["rank_sha256"]):
            checkpoint = _rank_checkpoint(output_dir, step, trainer_rank)
            if not checkpoint.is_file() or _sha256_file(checkpoint) != expected_hash:
                raise RuntimeError(f"incomplete or corrupt paired checkpoint: {checkpoint}")
        if "producer_sha256" in record:
            producer = _mix3_source_checkpoint(output_dir, step)
            if (
                not producer.is_file()
                or _sha256_file(producer) != record["producer_sha256"]
            ):
                raise RuntimeError(f"incomplete or corrupt producer checkpoint: {producer}")
        return step
    state = torch.load(resume, map_location="cpu", weights_only=False)
    return int(state["step"])


def _truncate_metrics_for_resume(output_dir: Path, checkpoint_step: int) -> None:
    path = output_dir / "training_metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError("resume checkpoint exists but training metrics are missing")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retained = [row for row in rows if int(row["step"]) <= checkpoint_step]
    steps = [int(row["step"]) for row in retained]
    if not retained or steps != sorted(set(steps)):
        raise ValueError("pre-checkpoint training metrics are missing, duplicated, or reordered")
    temporary = path.with_suffix(".jsonl.resume.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_checkpoint(output_dir: Path, step: int, trainer_rank: int) -> Path:
    return output_dir / f"checkpoint_{step:08d}.rank{trainer_rank}.pt"


def _mix3_source_checkpoint(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint_{step:08d}.producer.pt"


def _save_mix3_source_checkpoint(
    output_dir: Path,
    cfg: MultiModelExperimentConfig,
    step: int,
    source: Mix3ASPDSource,
) -> None:
    state = source.state_dict()
    if state["batch_count"] != step or state["schedule_index"] != step:
        raise RuntimeError("producer state is not aligned with the optimizer checkpoint step")
    atomic_torch_save(
        {
            "step": step,
            "config_sha256": resolved_config_sha256(cfg),
            "source_state": state,
        },
        _mix3_source_checkpoint(output_dir, step),
    )


def _restore_mix3_source_checkpoint(
    output_dir: Path,
    cfg: MultiModelExperimentConfig,
    step: int,
    source: Mix3ASPDSource,
) -> None:
    marker = torch.load(
        output_dir / "latest_checkpoint.pt", map_location="cpu", weights_only=True
    )
    path = _mix3_source_checkpoint(output_dir, step)
    expected = marker.get("producer_sha256")
    if not path.is_file() or not isinstance(expected, str):
        raise RuntimeError("Mix3 resume lacks coordinated producer source state")
    if _sha256_file(path) != expected:
        raise RuntimeError("Mix3 producer source-state checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != step:
        raise ValueError("Mix3 producer source state differs from resume step")
    if payload.get("config_sha256") != resolved_config_sha256(cfg):
        raise ValueError("Mix3 producer source state differs from frozen config")
    source.load_state_dict(payload["source_state"])


def _save_rank_checkpoint(
    output_dir: Path,
    cfg: MultiModelExperimentConfig,
    model: MultiModelASPD,
    optimizer: torch.optim.Optimizer,
    step: int,
    trainer_rank: int,
) -> None:
    cursor = StreamCursor.for_step(cfg, step)
    per_rank = cfg.training.sequences_per_rank or 0
    sampler = {
        rank: {
            "optimizer_step": step,
            "local_sequences": step * per_rank,
            "sequence_index": cursor.sequence_index + rank * per_rank,
        }
        for rank in (0, 1)
    }
    atomic_torch_save(
        checkpoint_payload(
            cfg,
            model,
            optimizer,
            step,
            sampler_state_by_rank=sampler,
        ),
        _rank_checkpoint(output_dir, step, trainer_rank),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit_checkpoint_pair(
    output_dir: Path, cfg: MultiModelExperimentConfig, step: int
) -> None:
    paths = [_rank_checkpoint(output_dir, step, rank) for rank in (0, 1)]
    if any(not path.is_file() for path in paths):
        raise RuntimeError(f"cannot commit incomplete checkpoint pair at step {step}")
    marker = {
        "step": step,
        "config_sha256": resolved_config_sha256(cfg),
        "rank_sha256": [_sha256_file(path) for path in paths],
    }
    if cfg.mix3_aspd is not None:
        producer = _mix3_source_checkpoint(output_dir, step)
        if not producer.is_file():
            raise RuntimeError(f"cannot commit Mix3 checkpoint without producer state: {producer}")
        marker["producer_sha256"] = _sha256_file(producer)
    atomic_torch_save(
        marker,
        output_dir / "latest_checkpoint.pt",
    )
    retained = cfg.training.keep_last_checkpoints
    for trainer_rank in (0, 1):
        checkpoints = sorted(output_dir.glob(f"checkpoint_*.rank{trainer_rank}.pt"))
        for obsolete in checkpoints[:-retained]:
            obsolete.unlink()
    if cfg.mix3_aspd is not None:
        producer_checkpoints = sorted(output_dir.glob("checkpoint_*.producer.pt"))
        for obsolete in producer_checkpoints[:-retained]:
            obsolete.unlink()


def _calibrate_training_rank(
    model: MultiModelASPD,
    cfg: MultiModelExperimentConfig,
    width: int,
    device: int | torch.device,
    trainer_group: dist.ProcessGroup,
) -> None:
    sums = [torch.zeros(width, dtype=torch.float64, device=device) for _ in range(2)]
    count = torch.zeros((), dtype=torch.long, device=device)
    for _ in range(cfg.original_aspd.calibration_batches):
        batch = _receive_activation_batch(cfg, width, 0, device)
        valid: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
        count += valid.sum()
        for index, residual in enumerate(batch["R"]):  # type: ignore[union-attr]
            sums[index] += residual[valid].double().sum(0)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=trainer_group)
    for value in sums:
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=trainer_group)
    model.calibrate_original_recipe([value.float() / count for value in sums])


def _producer_loop(
    cfg: MultiModelExperimentConfig,
    output_dir: Path,
    start_step: int,
    end_step: int,
    device: int | torch.device,
) -> None:
    loop_started = time.perf_counter()
    segment = f"steps_{start_step:08d}_{end_step:08d}"
    mix = None
    if cfg.mix3_aspd is not None:
        mix = load_r0_mix_config(cfg.mix3_aspd.mix_config)
        stream: FrozenTokenShardStream | Mix3ASPDSource = Mix3ASPDSource(mix, "train")
        if start_step:
            _restore_mix3_source_checkpoint(output_dir, cfg, start_step, stream)
        calibration_stream: FrozenTokenShardStream | Mix3ASPDSource = Mix3ASPDSource(
            mix, "calibration"
        )
    else:
        stream = FrozenTokenShardStream(cfg.data.frozen_manifest or "", "train")
        calibration_stream = FrozenTokenShardStream(
            cfg.data.frozen_manifest or "", "calibration"
        )
        required_calibration_sequences = (
            cfg.original_aspd.calibration_batches * cfg.training.batch_size_sequences
        )
        if calibration_stream.sequence_count != required_calibration_sequences:
            raise ValueError(
                "calibration split must contain exactly the eight disjoint global batches"
            )
        if stream.sequence_count != cfg.training.max_steps * cfg.training.batch_size_sequences:
            raise ValueError("training stream does not contain the exact exposure budget")
    models: list[torch.nn.Module] = []
    provenances: list[dict[str, Any]] = []
    weights: list[dict[str, Tensor]] = []
    for spec in cfg.models:
        model, provenance = load_effective_model(spec, device)
        snapshot, _ = snapshot_model_weights(model, spec, provenance)
        model.requires_grad_(False)
        models.append(model)
        provenances.append(provenance)
        weights.append(snapshot)
    model_ready = time.perf_counter()
    for destination in (1, 2):
        _send_weights(weights, destination, device)
    capture_seconds = 0.0
    communication_seconds = 0.0
    captures = [_Capture(), _Capture()]
    with ExitStack() as stack:
        for model, spec, capture in zip(models, cfg.models, captures, strict=True):
            _register_capture_hooks(model, spec, capture, stack)

        def produce(
            source: FrozenTokenShardStream | Mix3ASPDSource,
            sequence_index: int,
            *,
            training: bool,
            role: str | None = None,
        ) -> None:
            nonlocal capture_seconds, communication_seconds
            capture_started = time.perf_counter()
            if isinstance(source, Mix3ASPDSource):
                if role is None:
                    raise ValueError("Mix3 capture requires an explicit scheduled source role")
                paired = source.capture_batch(
                    role,
                    models=models,
                    captures=captures,
                    device=device,
                    sequence_start=sequence_index,
                )
            else:
                tokens = source.batch_at(sequence_index, cfg.training.batch_size_sequences)
                if training and (tokens["provenance_ids"] < 0).any():
                    raise ValueError("training shards must carry per-sequence provenance_ids")
                paired = _capture_pair(
                    models,
                    captures,
                    tokens["input_ids"],
                    tokens["valid_tokens"],
                    device,
                    sequence_ids=tokens["sequence_ids"],
                    provenance_ids=tokens["provenance_ids"],
                )
            capture_seconds += time.perf_counter() - capture_started
            per_rank = cfg.training.sequences_per_rank or 0
            communication_started = time.perf_counter()
            for trainer_rank, destination in enumerate((1, 2)):
                _send_activation_batch(
                    _rank_slice(paired, trainer_rank, per_rank), destination
                )
            communication_seconds += time.perf_counter() - communication_started

        if start_step == 0:
            for calibration_index in range(cfg.original_aspd.calibration_batches):
                produce(
                    calibration_stream,
                    calibration_index * cfg.training.batch_size_sequences,
                    training=False,
                    role=(
                        mix.source_schedule[calibration_index % len(mix.source_schedule)]
                        if mix is not None
                        else None
                    ),
                )
        for step in range(start_step, end_step):
            produce(
                stream,
                StreamCursor.for_step(cfg, step).sequence_index,
                training=True,
                role=(
                    mix.source_schedule[step % len(mix.source_schedule)]
                    if mix is not None
                    else None
                ),
            )
            completed_step = step + 1
            if cfg.mix3_aspd is not None and (
                completed_step % cfg.training.save_every == 0
                or completed_step == end_step
            ):
                if not isinstance(stream, Mix3ASPDSource):
                    raise AssertionError("Mix3 producer lost its exact source adapter")
                _save_mix3_source_checkpoint(
                    output_dir, cfg, completed_step, stream
                )
                # The first world barrier pairs this exact cursor with trainer
                # weights at the same step. The second waits for the atomic marker.
                dist.barrier()
                dist.barrier()
    final_weights = [
        snapshot_model_weights(model, spec, provenance)[0]
        for model, spec, provenance in zip(
            models, cfg.models, provenances, strict=True
        )
    ]
    for before, after in zip(weights, final_weights, strict=True):
        if any(not torch.equal(before[name], after[name]) for name in before):
            raise RuntimeError("one or more source LLM matrix weights changed during streaming")
    completed = time.perf_counter()
    (output_dir / f"producer_provenance_{segment}.json").write_text(
        json.dumps(
            {
                "model_provenance": provenances,
                "start_step": start_step,
                "end_step": end_step,
                "canonical_token_policy": "one CPU input_ids tensor cloned for both models",
                "activation_persistence": "none",
                "effective_queue_depth": 1,
                "configured_queue_bound": cfg.training.bounded_queue_depth,
                "calibration_split": "calibration",
                "calibration_tokens": cfg.data.calibration_tokens,
                "source_llm_weights_unchanged": True,
                "config_sha256": resolved_config_sha256(cfg),
                "mix3_source_state": (
                    stream.state_dict() if isinstance(stream, Mix3ASPDSource) else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / f"runtime_rank0_producer_{segment}.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "role": "producer",
                "start_step": start_step,
                "end_step": end_step,
                "startup_seconds": model_ready - loop_started,
                "capture_forward_seconds": capture_seconds,
                "communication_seconds": communication_seconds,
                "segment_wall_seconds": completed - model_ready,
                "rank_wall_seconds_including_startup": completed - loop_started,
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if torch.cuda.is_available()
                    else 0
                ),
                "peak_cpu_rss_kib": _peak_cpu_rss_kib(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _trainer_loop(
    cfg: MultiModelExperimentConfig,
    output_dir: Path,
    start_step: int,
    end_step: int,
    device: int | torch.device,
    global_rank: int,
    trainer_group: dist.ProcessGroup,
    save_checkpoints: bool,
) -> None:
    loop_started = time.perf_counter()
    segment = f"steps_{start_step:08d}_{end_step:08d}"
    trainer_rank = global_rank - 1
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
    weights = _receive_weights(0, device)
    width = int(weights[0]["o_proj"].shape[0])
    if any(tuple(item["o_proj"].shape) != (width, width) for item in weights):
        raise ValueError("frozen L14 o_proj runtime expects equal square Base/Target matrices")
    model = MultiModelASPD(
        model_names=[item.name for item in cfg.models],
        activation_dims=[width, width],
        target_weights=weights,
        encoder_cfg=cfg.encoder,
        sparsity_cfg=cfg.sparsity,
        objective_cfg=cfg.objective,
        original_aspd_cfg=cfg.original_aspd,
    ).to(device)
    model.attach_statistics_process_group(trainer_group)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
        fused=torch.device(device).type == "cuda",
    )
    if start_step == 0:
        _calibrate_training_rank(model, cfg, width, device, trainer_group)
    else:
        payload = torch.load(
            _rank_checkpoint(output_dir, start_step, trainer_rank),
            map_location="cpu",
            weights_only=False,
        )
        restored_step, _ = restore_checkpoint_payload(cfg, model, optimizer, payload)
        if restored_step != start_step:
            raise ValueError("rank checkpoint step differs from the shared resume pointer")
    is_cuda = not isinstance(device, torch.device) or device.type == "cuda"
    ddp = DistributedDataParallel(
        model,
        device_ids=[device] if is_cuda else None,
        process_group=trainer_group,
        broadcast_buffers=False,
    )
    model.train()
    started = time.perf_counter()
    metrics_path = output_dir / "training_metrics.jsonl"
    receive_seconds = 0.0
    trainer_seconds = 0.0
    checkpoint_seconds = 0.0
    for index in range(start_step, end_step):
        receive_started = time.perf_counter()
        batch = _receive_activation_batch(cfg, width, 0, device)
        receive_seconds += time.perf_counter() - receive_started
        expected_sequence_ids = torch.arange(
            index * cfg.training.batch_size_sequences + trainer_rank * cfg.training.sequences_per_rank,
            index * cfg.training.batch_size_sequences
            + (trainer_rank + 1) * cfg.training.sequences_per_rank,
            device=device,
        )
        if not torch.equal(batch["sequence_ids"], expected_sequence_ids):
            raise RuntimeError("activation stream replayed, skipped, or reordered sequences")
        optimizer.zero_grad(set_to_none=True)
        trainer_started = time.perf_counter()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if is_cuda
            else nullcontext()
        )
        with autocast:
            result = ddp(batch)
        result["loss"].backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("ASPD gradient contains NaN or Inf")
        model.normalize_original_decoders_()
        optimizer.step()
        trainer_seconds += time.perf_counter() - trainer_started
        step = index + 1
        if step == 1 or step % cfg.training.log_every == 0:
            names = list(result)
            values = torch.stack([result[name].detach().float() for name in names])
            dist.all_reduce(values, op=dist.ReduceOp.SUM, group=trainer_group)
            values /= cfg.training.training_world_size
            if trainer_rank == 0:
                elapsed = time.perf_counter() - started
                row = {
                    "step": step,
                    "token_exposures": step
                    * cfg.training.batch_size_sequences
                    * cfg.data.sequence_length,
                    "positions_per_second": (
                        (step - start_step)
                        * cfg.training.batch_size_sequences
                        * cfg.data.sequence_length
                        / max(elapsed, 1e-9)
                    ),
                    **{
                        f"train/{name}": float(value.cpu())
                        for name, value in zip(names, values, strict=True)
                    },
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        if save_checkpoints and (
            step % cfg.training.save_every == 0 or step == end_step
        ):
            checkpoint_started = time.perf_counter()
            _save_rank_checkpoint(
                output_dir, cfg, model, optimizer, step, trainer_rank
            )
            dist.barrier(group=trainer_group)
            if cfg.mix3_aspd is not None:
                # Coordinate with rank-0 producer state before committing the
                # pair, then hold every rank until the marker is durable.
                dist.barrier()
                if trainer_rank == 0:
                    _commit_checkpoint_pair(output_dir, cfg, step)
                dist.barrier()
            else:
                if trainer_rank == 0:
                    _commit_checkpoint_pair(output_dir, cfg, step)
                dist.barrier(group=trainer_group)
            checkpoint_seconds += time.perf_counter() - checkpoint_started
    elapsed = time.perf_counter() - started
    completed = time.perf_counter()
    (output_dir / f"runtime_rank{global_rank}_trainer_{segment}.json").write_text(
        json.dumps(
            {
                "rank": global_rank,
                "role": "trainer",
                "trainer_rank": trainer_rank,
                "start_step": start_step,
                "end_step": end_step,
                "segment_wall_seconds": elapsed,
                "rank_wall_seconds_including_startup": completed - loop_started,
                "receive_seconds": receive_seconds,
                "trainer_step_seconds": trainer_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "positions_per_second_excluding_model_startup": (
                    (end_step - start_step)
                    * cfg.training.batch_size_sequences
                    * cfg.data.sequence_length
                    / max(elapsed, 1e-9)
                ),
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device) if is_cuda else 0
                ),
                "peak_cpu_rss_kib": _peak_cpu_rss_kib(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_three_gpu_streaming(
    cfg: MultiModelExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    resume: str | None = None,
    stop_after_step: int | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Execute an already-scheduled three-GPU torchrun; this function never submits jobs."""

    mix3_report: dict[str, Any] | None = None
    if cfg.mix3_aspd is None:
        validate_original_recipe_config(cfg)
        validate_dataset_manifest(cfg, require_ready=True)
    else:
        mix3_report = validate_mix3_recipe_config(cfg, require_ready=True)
    DistributedTopology(
        sequences_per_trainer=cfg.training.sequences_per_rank or 0,
        sequence_length=cfg.data.sequence_length,
    ).validate(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError("three visible CUDA GPUs are required")
    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    if dist.get_world_size() != 3:
        raise RuntimeError("torchrun world size must be exactly three")
    torch.cuda.set_device(global_rank)
    destination = Path(output_dir or cfg.training.output_dir)
    if resume is None and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"fresh original-ASPD run refuses a non-empty output directory: {destination}"
        )
    if global_rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        frozen_config = destination / "config_frozen.yaml"
        frozen_manifest = destination / "data_manifest.json"
        if resume is None:
            if config_path is None:
                raise ValueError("config_path is required for an immutable run package")
            shutil.copyfile(config_path, frozen_config)
            if mix3_report is None:
                source_manifest = Path(cfg.data.frozen_manifest or "")
                shutil.copyfile(source_manifest, frozen_manifest)
            else:
                frozen_manifest.write_text(
                    json.dumps(mix3_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        elif not frozen_config.is_file() or not frozen_manifest.is_file():
            raise FileNotFoundError("resume package lacks frozen config or data manifest")
        else:
            if config_path is None:
                raise ValueError("config_path is required to verify immutable resume inputs")
            if _sha256_file(Path(config_path)) != _sha256_file(frozen_config):
                raise ValueError("resume config file differs from the frozen run config")
            if mix3_report is None:
                source_manifest = Path(cfg.data.frozen_manifest or "")
                if _sha256_file(source_manifest) != _sha256_file(frozen_manifest):
                    raise ValueError("resume data manifest differs from the frozen run manifest")
            else:
                expected = json.dumps(mix3_report, indent=2, sort_keys=True) + "\n"
                if frozen_manifest.read_text(encoding="utf-8") != expected:
                    raise ValueError("resume Mix3 manifest set differs from the frozen run package")
    dist.barrier()
    trainer_group = dist.new_group(ranks=[1, 2], backend="nccl")
    start_value = _resume_step(destination, resume) if global_rank == 0 else 0
    if global_rank == 0 and start_value > 0:
        _truncate_metrics_for_resume(destination, start_value)
    start_tensor = torch.tensor(start_value, dtype=torch.long, device=global_rank)
    dist.broadcast(start_tensor, src=0)
    start_step = int(start_tensor.item())
    end_step = cfg.training.max_steps if stop_after_step is None else stop_after_step
    if not start_step < end_step <= cfg.training.max_steps:
        raise ValueError("stop_after_step must be after resume and within the frozen run")
    if global_rank == 0:
        _producer_loop(cfg, destination, start_step, end_step, global_rank)
    else:
        _trainer_loop(
            cfg,
            destination,
            start_step,
            end_step,
            global_rank,
            global_rank,
            trainer_group,
            True,
        )
    dist.barrier()
    if global_rank == 0:
        marker = torch.load(
            destination / "latest_checkpoint.pt", map_location="cpu", weights_only=True
        )
        if int(marker["step"]) != end_step:
            raise RuntimeError("final paired checkpoint was not committed")
        metric_rows = [
            json.loads(line)
            for line in (destination / "training_metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if not metric_rows or any(
            not torch.isfinite(torch.tensor(value))
            for row in metric_rows
            for key, value in row.items()
            if key.startswith("train/")
        ):
            raise RuntimeError("training metrics are missing or non-finite")
        summary = {
            "status": (
                (
                    "MIX3_200M_TRAINING_COMPLETE"
                    if cfg.mix3_aspd is not None
                    else "100M_TRAINING_COMPLETE"
                )
                if end_step == cfg.training.max_steps
                else "TECHNICAL_SMOKE_SEGMENT_COMPLETE"
            ),
            "start_step": start_step,
            "final_step": end_step,
            "total_token_exposures": (
                end_step * cfg.training.batch_size_sequences * cfg.data.sequence_length
            ),
            "config_sha256": resolved_config_sha256(cfg),
            "config_file_sha256": _sha256_file(destination / "config_frozen.yaml"),
            "data_manifest_sha256": _sha256_file(destination / "data_manifest.json"),
            "checkpoint_marker_sha256": _sha256_file(
                destination / "latest_checkpoint.pt"
            ),
            "rank_checkpoint_sha256": marker["rank_sha256"],
            "source_llm_weights_unchanged": True,
            "losses_finite": True,
            "gradients_finite": True,
            "scientific_run": end_step == cfg.training.max_steps,
            "runtime_segment_files": sorted(
                path.name for path in destination.glob("runtime_rank*_*.json")
            ),
            "producer_provenance_files": sorted(
                path.name for path in destination.glob("producer_provenance_steps_*.json")
            ),
        }
        temporary = destination / "training_summary.json.tmp"
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination / "training_summary.json")
        print(
            "ASPD_MIX3_STREAMING_COMPLETE"
            if cfg.mix3_aspd is not None
            else "ASPD_ORIGINAL_STREAMING_COMPLETE",
            flush=True,
        )
    dist.destroy_process_group()

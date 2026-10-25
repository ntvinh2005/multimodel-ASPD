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
import tempfile
import time
from contextlib import ExitStack
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
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_recipe import (
    StreamCursor,
    validate_dataset_manifest,
    validate_original_recipe_config,
)


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
    return {
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
    shape: tuple[int, ...], dtype: torch.dtype, source: int, device: int
) -> Tensor:
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=source)
    return tensor


def _send_weights(weights: list[dict[str, Tensor]], destination: int, device: int) -> None:
    for item in weights:
        matrix = item["o_proj"].to(device=device, dtype=torch.bfloat16)
        _send(torch.tensor(matrix.shape, dtype=torch.long, device=device), destination)
        _send(matrix, destination)


def _receive_weights(source: int, device: int) -> list[dict[str, Tensor]]:
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
    for residual in batch["R"]:  # type: ignore[union-attr]
        _send(residual, destination)
    for inputs in batch["X"]:  # type: ignore[union-attr]
        _send(inputs["o_proj"], destination)


def _receive_activation_batch(
    cfg: MultiModelExperimentConfig, width: int, source: int, device: int
) -> dict[str, object]:
    local_batch = cfg.training.sequences_per_rank
    assert local_batch is not None
    grid = (local_batch, cfg.data.sequence_length)
    shape = (*grid, width)
    return {
        "input_ids": _receive(grid, torch.long, source, device),
        "sequence_ids": _receive((local_batch,), torch.long, source, device),
        "provenance_ids": _receive((local_batch,), torch.long, source, device),
        "valid_tokens": _receive(grid, torch.uint8, source, device).bool(),
        "R": [
            _receive(shape, torch.bfloat16, source, device),
            _receive(shape, torch.bfloat16, source, device),
        ],
        "X": [
            {"o_proj": _receive(shape, torch.bfloat16, source, device)},
            {"o_proj": _receive(shape, torch.bfloat16, source, device)},
        ],
    }


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
    return {
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
        return step
    state = torch.load(resume, map_location="cpu", weights_only=False)
    return int(state["step"])


def _rank_checkpoint(output_dir: Path, step: int, trainer_rank: int) -> Path:
    return output_dir / f"checkpoint_{step:08d}.rank{trainer_rank}.pt"


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
    atomic_torch_save(
        {
            "step": step,
            "config_sha256": resolved_config_sha256(cfg),
            "rank_sha256": [_sha256_file(path) for path in paths],
        },
        output_dir / "latest_checkpoint.pt",
    )
    retained = cfg.training.keep_last_checkpoints
    for trainer_rank in (0, 1):
        checkpoints = sorted(output_dir.glob(f"checkpoint_*.rank{trainer_rank}.pt"))
        for obsolete in checkpoints[:-retained]:
            obsolete.unlink()


def _calibrate_training_rank(
    model: MultiModelASPD,
    cfg: MultiModelExperimentConfig,
    width: int,
    device: int,
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
    device: int,
) -> None:
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
    for destination in (1, 2):
        _send_weights(weights, destination, device)
    captures = [_Capture(), _Capture()]
    with ExitStack() as stack:
        for model, spec, capture in zip(models, cfg.models, captures, strict=True):
            _register_capture_hooks(model, spec, capture, stack)

        def produce(
            source: FrozenTokenShardStream, sequence_index: int, *, training: bool
        ) -> None:
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
            per_rank = cfg.training.sequences_per_rank or 0
            for trainer_rank, destination in enumerate((1, 2)):
                _send_activation_batch(
                    _rank_slice(paired, trainer_rank, per_rank), destination
                )

        if start_step == 0:
            for calibration_index in range(cfg.original_aspd.calibration_batches):
                produce(
                    calibration_stream,
                    calibration_index * cfg.training.batch_size_sequences,
                    training=False,
                )
        for step in range(start_step, end_step):
            produce(
                stream,
                StreamCursor.for_step(cfg, step).sequence_index,
                training=True,
            )
    final_weights = [
        snapshot_model_weights(model, spec, provenance)[0]
        for model, spec, provenance in zip(
            models, cfg.models, provenances, strict=True
        )
    ]
    for before, after in zip(weights, final_weights, strict=True):
        if any(not torch.equal(before[name], after[name]) for name in before):
            raise RuntimeError("one or more source LLM matrix weights changed during streaming")
    (output_dir / "producer_provenance.json").write_text(
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
    device: int,
    global_rank: int,
    trainer_group: dist.ProcessGroup,
    save_checkpoints: bool,
) -> None:
    trainer_rank = global_rank - 1
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
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
        fused=True,
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
    ddp = DistributedDataParallel(
        model,
        device_ids=[device],
        process_group=trainer_group,
        broadcast_buffers=False,
    )
    model.train()
    started = time.perf_counter()
    metrics_path = output_dir / "metrics.jsonl"
    for index in range(start_step, end_step):
        batch = _receive_activation_batch(cfg, width, 0, device)
        expected_sequence_ids = torch.arange(
            index * cfg.training.batch_size_sequences + trainer_rank * cfg.training.sequences_per_rank,
            index * cfg.training.batch_size_sequences
            + (trainer_rank + 1) * cfg.training.sequences_per_rank,
            device=device,
        )
        if not torch.equal(batch["sequence_ids"], expected_sequence_ids):
            raise RuntimeError("activation stream replayed, skipped, or reordered sequences")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = ddp(batch)
        result["loss"].backward()
        model.normalize_original_decoders_()
        optimizer.step()
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
            _save_rank_checkpoint(
                output_dir, cfg, model, optimizer, step, trainer_rank
            )
            dist.barrier(group=trainer_group)
            if trainer_rank == 0:
                _commit_checkpoint_pair(output_dir, cfg, step)
            dist.barrier(group=trainer_group)


def run_three_gpu_streaming(
    cfg: MultiModelExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    resume: str | None = None,
    benchmark_steps: int | None = None,
) -> None:
    """Execute an already-scheduled three-GPU torchrun; this function never submits jobs."""

    validate_original_recipe_config(cfg)
    validate_dataset_manifest(cfg, require_ready=True)
    DistributedTopology().validate(cfg)
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
    dist.barrier()
    trainer_group = dist.new_group(ranks=[1, 2], backend="nccl")
    start_value = _resume_step(destination, resume) if global_rank == 0 else 0
    start_tensor = torch.tensor(start_value, dtype=torch.long, device=global_rank)
    dist.broadcast(start_tensor, src=0)
    start_step = int(start_tensor.item())
    end_step = cfg.training.max_steps
    save_checkpoints = benchmark_steps is None
    if benchmark_steps is not None:
        if start_step or benchmark_steps <= 0:
            raise ValueError("benchmark_steps requires a fresh positive-length run")
        end_step = min(benchmark_steps, end_step)
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
            save_checkpoints,
        )
    dist.barrier()
    if global_rank == 0:
        print("ASPD_ORIGINAL_STREAMING_COMPLETE", flush=True)
    dist.destroy_process_group()

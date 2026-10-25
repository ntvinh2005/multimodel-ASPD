"""Executed three-process Gloo verification for the production streaming primitives."""

from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_streaming import (
    _send_activation_batch,
    _send_weights,
    _trainer_loop,
)


def _tiny_config(cfg: MultiModelExperimentConfig) -> MultiModelExperimentConfig:
    return cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={
                    "sequence_length": 2,
                    "train_tokens": 32,
                    "calibration_tokens": 8,
                    "validation_tokens": 8,
                }
            ),
            "sparsity": cfg.sparsity.model_copy(
                update={"n_features": 8, "top_k": 2, "pool_tokens": 4}
            ),
            "objective": cfg.objective.model_copy(
                update={
                    "top_k_aux": 4,
                    "dead_after_batches": 2,
                    "alive_window_tokens": 32,
                    "matryoshka_group_fractions": [0.5, 0.5],
                }
            ),
            "training": cfg.training.model_copy(
                update={
                    "max_steps": 4,
                    "batch_size_sequences": 4,
                    "sequences_per_rank": 2,
                    "save_every": 2,
                    "keep_last_checkpoints": 10,
                    "log_every": 1,
                    "expected_token_exposures": 32,
                }
            ),
            "original_aspd": cfg.original_aspd.model_copy(
                update={"calibration_batches": 1}
            ),
        }
    )


def _weights(width: int = 4) -> list[dict[str, Tensor]]:
    base = torch.arange(width * width, dtype=torch.float32).reshape(width, width) / 17
    return [{"o_proj": base}, {"o_proj": base + 0.125 * torch.eye(width)}]


def _batch(
    cfg: MultiModelExperimentConfig,
    step: int,
    trainer_rank: int,
    *,
    calibration: bool,
    width: int = 4,
) -> dict[str, object]:
    local = cfg.training.sequences_per_rank or 0
    start = 0 if calibration else step * cfg.training.batch_size_sequences + trainer_rank * local
    sequence_ids = torch.arange(start, start + local, dtype=torch.long)
    input_ids = sequence_ids[:, None] * 10 + torch.arange(cfg.data.sequence_length)
    values = input_ids.float()[..., None] + torch.arange(width).float() / 10
    return {
        "input_ids": input_ids,
        "sequence_ids": sequence_ids,
        "provenance_ids": sequence_ids + 100,
        "valid_tokens": torch.ones_like(input_ids, dtype=torch.bool),
        "R": [(values / 7).to(torch.bfloat16), ((values + 1.5) / 8).to(torch.bfloat16)],
        "X": [
            {"o_proj": ((values + 3) / 11).to(torch.bfloat16)},
            {"o_proj": ((values - 2) / 13).to(torch.bfloat16)},
        ],
    }


def _worker(
    rank: int,
    init_method: str,
    cfg: MultiModelExperimentConfig,
    output_dir: str,
    start_step: int,
    end_step: int,
    socket_interface: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = socket_interface
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=3)
    trainer_group = dist.new_group(ranks=[1, 2], backend="gloo")
    if trainer_group is None:
        raise RuntimeError("Gloo trainer subgroup creation failed")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if rank == 0:
            weights = _weights()
            for target in (1, 2):
                _send_weights(weights, target, torch.device("cpu"))
            if start_step == 0:
                for trainer_rank, target in enumerate((1, 2)):
                    _send_activation_batch(
                        _batch(cfg, 0, trainer_rank, calibration=True), target
                    )
            for step in range(start_step, end_step):
                for trainer_rank, target in enumerate((1, 2)):
                    _send_activation_batch(
                        _batch(cfg, step, trainer_rank, calibration=False), target
                    )
        else:
            _trainer_loop(
                cfg,
                destination,
                start_step,
                end_step,
                torch.device("cpu"),
                rank,
                trainer_group,
                True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _launch(
    cfg: MultiModelExperimentConfig,
    output_dir: Path,
    start_step: int,
    end_step: int,
) -> None:
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    socket_interface = "Loopback Pseudo-Interface 1" if os.name == "nt" else "lo"
    mp.spawn(
        _worker,
        args=(init_method, cfg, str(output_dir), start_step, end_step, socket_interface),
        nprocs=3,
        join=True,
    )


def _difference(left: Any, right: Any) -> tuple[float, bool]:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return math.inf, False
        maximum = float((left.double() - right.double()).abs().max()) if left.numel() else 0.0
        return maximum, torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        children = [_difference(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
        children = [_difference(a, b) for a, b in zip(left, right, strict=True)]
    else:
        return (0.0, True) if left == right else (math.inf, False)
    return max((value for value, _ in children), default=0.0), all(equal for _, equal in children)


def _checkpoint_evaluation(
    cfg: MultiModelExperimentConfig, state: dict[str, Any]
) -> float:
    weights = [
        {"o_proj": state["model"][f"target_weights.{index}.o_proj.weight"]}
        for index in (0, 1)
    ]
    model = MultiModelASPD(
        [item.name for item in cfg.models],
        [4, 4],
        weights,
        cfg.encoder,
        cfg.sparsity,
        cfg.objective,
        cfg.original_aspd,
    )
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.no_grad():
        result = model(_batch(cfg, 3, 0, calibration=False), update_dead_tracker=False)
    loss = float(result["loss"])
    if not math.isfinite(loss):
        raise FloatingPointError("resumed Gloo checkpoint evaluation is non-finite")
    return loss


def execute_three_process_gloo_verification(
    cfg: MultiModelExperimentConfig,
    output_root: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run uninterrupted and interrupted/resumed production communication on three CPU ranks."""

    tiny = _tiny_config(cfg)
    root = Path(output_root)
    report_destination = Path(report_path)
    if root.exists() or report_destination.exists():
        raise FileExistsError("Gloo verification refuses to overwrite prior evidence")
    uninterrupted = root / "uninterrupted"
    resumed = root / "interrupted_resumed"
    _launch(tiny, uninterrupted, 0, 4)
    _launch(tiny, resumed, 0, 2)
    _launch(tiny, resumed, 2, 4)
    comparisons = []
    for rank in (0, 1):
        left = torch.load(
            uninterrupted / f"checkpoint_00000004.rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        right = torch.load(
            resumed / f"checkpoint_00000004.rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model_max, model_exact = _difference(left["model"], right["model"])
        optimizer_max, optimizer_exact = _difference(left["optimizer"], right["optimizer"])
        comparisons.append(
            {
                "trainer_rank": rank,
                "model_max_abs_difference": model_max,
                "model_exact": model_exact,
                "optimizer_max_abs_difference": optimizer_max,
                "optimizer_exact": optimizer_exact,
                "stream_cursor_exact": left["stream_cursor"] == right["stream_cursor"],
                "sampler_cursor_exact": (
                    left["sampler_state_by_rank"] == right["sampler_state_by_rank"]
                ),
                "torch_rng_exact": torch.equal(left["torch_rng"], right["torch_rng"]),
                "python_rng_exact": left["python_rng"] == right["python_rng"],
            }
        )
    resumed_states = [
        torch.load(
            resumed / f"checkpoint_00000004.rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in (0, 1)
    ]
    synchronized_keys = [
        name
        for name in resumed_states[0]["model"]
        if not any(
            marker in name
            for marker in (
                "sparsity_controller.threshold",
                "sparsity_controller.n_pools_not_active",
                "alive_tracker.",
            )
        )
    ]
    ddp_equal = all(
        torch.equal(resumed_states[0]["model"][name], resumed_states[1]["model"][name])
        for name in synchronized_keys
    )
    if not ddp_equal or not all(
        row["model_exact"]
        and row["optimizer_exact"]
        and row["stream_cursor_exact"]
        and row["sampler_cursor_exact"]
        and row["torch_rng_exact"]
        and row["python_rng_exact"]
        for row in comparisons
    ):
        raise AssertionError("three-process interrupted/resumed execution is not exact")
    report = {
        "schema_version": 1,
        "status": "LOCALLY_TESTED",
        "backend": "gloo",
        "world_size": 3,
        "topology": {"rank0": "synthetic producer", "rank1_2": "ASPD DDP trainers"},
        "production_primitives_exercised": [
            "_send_weights",
            "_send_activation_batch",
            "_receive_activation_batch",
            "_trainer_loop",
            "paired checkpoint barrier and atomic marker",
        ],
        "uninterrupted_steps": 4,
        "interrupted_at_step": 2,
        "resumed_to_step": 4,
        "comparisons": comparisons,
        "ddp_synchronized_state_equal": ddp_equal,
        "resumed_checkpoint_evaluation_loss": _checkpoint_evaluation(
            tiny, resumed_states[0]
        ),
        "clean_process_shutdown": True,
    }
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report

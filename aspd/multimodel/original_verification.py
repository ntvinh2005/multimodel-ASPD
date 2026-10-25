"""Offline completion and throughput checks for original-recipe ASPD runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.original_recipe import (
    StreamCursor,
    sha256_file,
    validate_original_recipe_config,
)
from aspd.multimodel.original_streaming import resolved_config_sha256


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _checkpoint_pair(
    run: Path,
    expected_step: int,
    config_sha256: str,
    cfg: MultiModelExperimentConfig,
) -> dict[str, Any]:
    marker_path = run / "latest_checkpoint.pt"
    marker = torch.load(marker_path, map_location="cpu", weights_only=True)
    if int(marker.get("step", -1)) != expected_step:
        raise ValueError("latest checkpoint marker has the wrong step")
    if marker.get("config_sha256") != config_sha256:
        raise ValueError("latest checkpoint marker has the wrong config fingerprint")
    paths = [run / f"checkpoint_{expected_step:08d}.rank{rank}.pt" for rank in (0, 1)]
    observed_hashes = [sha256_file(path) for path in paths]
    if observed_hashes != marker.get("rank_sha256"):
        raise ValueError("paired checkpoint hashes differ from the atomic marker")
    states = [
        torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths
    ]
    if any(int(state.get("step", -1)) != expected_step for state in states):
        raise ValueError("one or more rank checkpoints have the wrong step")
    for state in states:
        if state.get("config_sha256") != config_sha256:
            raise ValueError("one or more rank checkpoints have the wrong config fingerprint")
        cursor = StreamCursor(**state["stream_cursor"])
        cursor.validate(cfg)
        sampler = {int(key): value for key, value in state["sampler_state_by_rank"].items()}
        if set(sampler) != {0, 1}:
            raise ValueError("one or more rank checkpoints lack the paired sampler state")
        per_rank = cfg.training.sequences_per_rank or 0
        for trainer_rank in (0, 1):
            expected_sequence = cursor.sequence_index + trainer_rank * per_rank
            if sampler[trainer_rank].get("sequence_index") != expected_sequence:
                raise ValueError("rank checkpoint sampler would replay or skip training data")
    unsynchronized_buffers = (
        "sparsity_controller.threshold",
        "sparsity_controller.n_pools_not_active",
        "alive_tracker.",
    )
    parameter_names = sorted(
        name
        for name in states[0]["model"]
        if not any(marker in name for marker in unsynchronized_buffers)
    )
    unequal = [
        name
        for name in parameter_names
        if not torch.equal(states[0]["model"][name], states[1]["model"][name])
    ]
    if unequal:
        raise ValueError(f"DDP rank parameters differ: {unequal[:5]}")
    return {
        "step": expected_step,
        "marker_path": str(marker_path),
        "marker_sha256": sha256_file(marker_path),
        "rank_paths": [str(path) for path in paths],
        "rank_sha256": observed_hashes,
        "ddp_parameters_equal": True,
        "rank_local_statistics": [
            {
                "jump_relu_threshold": float(
                    state["model"]["sparsity_controller.threshold"]
                ),
                "dead_components": int(
                    (
                        state["model"]["sparsity_controller.n_pools_not_active"]
                        >= 4_882
                    ).sum()
                ),
            }
            for state in states
        ],
        "stream_cursors": [state["stream_cursor"] for state in states],
        "sampler_state_by_rank": [state["sampler_state_by_rank"] for state in states],
    }


def _smoke_optimizer_update(run: Path) -> dict[str, Any]:
    before = torch.load(
        run / "checkpoint_00000050.rank0.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )["model"]
    after = torch.load(
        run / "checkpoint_00000100.rank0.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )["model"]
    trainable_prefixes = ("encoder.", "activation_decoders.", "components.")
    changed = [
        name
        for name in before
        if name.startswith(trainable_prefixes) and not torch.equal(before[name], after[name])
    ]
    if not changed:
        raise ValueError("no trainable ASPD tensor changed between smoke steps 50 and 100")
    return {
        "checkpoint_50_to_100_changed": True,
        "changed_trainable_tensor_count": len(changed),
        "example_changed_tensors": changed[:10],
    }


def _metrics(
    run: Path,
    expected_step: int,
    expected_exposures: int,
    *,
    log_every: int,
) -> list[dict[str, Any]]:
    path = run / "training_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training metrics do not reach the expected final step")
    final_logged_step = int(rows[-1]["step"])
    # Older completed runs logged only exact cadence boundaries.  Permit the final
    # partial interval when the summary and paired checkpoint independently prove
    # the exact final step; newly produced runs always log end_step explicitly.
    if not 0 <= expected_step - final_logged_step < log_every:
        raise ValueError("training metrics do not reach the expected final step")
    positions_per_step = expected_exposures // expected_step
    if int(rows[-1]["token_exposures"]) != final_logged_step * positions_per_step:
        raise ValueError("training metrics have the wrong final exposure count")
    if not _finite(rows):
        raise FloatingPointError("training metrics contain NaN or Inf")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(set(steps)):
        raise ValueError("training metric steps are duplicated or reordered")
    return rows


def _runtime(run: Path) -> dict[str, Any]:
    records = [_json(path) for path in sorted(run.glob("runtime_rank*_steps_*.json"))]
    if not records:
        raise ValueError("runtime segment records are missing")
    segments: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = int(record["start_step"]), int(record["end_step"])
        segments.setdefault(key, []).append(record)
    total_positions = 0
    total_segment_wall = 0.0
    total_segment_wall_with_smoke_checkpoints = 0.0
    total_checkpoint_wall = 0.0
    total_rank_wall = 0.0
    for (start, end), rows in sorted(segments.items()):
        if {int(row["rank"]) for row in rows} != {0, 1, 2}:
            raise ValueError(f"runtime segment {(start, end)} lacks one or more ranks")
        total_positions += (end - start) * 4096
        active_walls = []
        for row in rows:
            checkpoint_seconds = float(row.get("checkpoint_seconds", 0.0))
            active_wall = float(row["segment_wall_seconds"]) - checkpoint_seconds
            if active_wall < 0:
                raise ValueError("runtime checkpoint time exceeds segment wall time")
            active_walls.append(active_wall)
        # The smoke deliberately checkpoints every 50 steps to exercise resume, while the
        # scientific run checkpoints every 1,000 steps.  Keep that stress-test I/O visible,
        # but do not fold its 20x cadence into sustained training throughput.
        total_segment_wall += max(active_walls)
        total_segment_wall_with_smoke_checkpoints += max(
            float(row["segment_wall_seconds"]) for row in rows
        )
        # Trainer ranks write concurrently, so checkpoint wall time is the per-segment max.
        total_checkpoint_wall += max(
            float(row.get("checkpoint_seconds", 0.0)) for row in rows
        )
        total_rank_wall += max(float(row["rank_wall_seconds_including_startup"]) for row in rows)
    producer = [row for row in records if row["role"] == "producer"]
    trainers = [row for row in records if row["role"] == "trainer"]
    excluding_startup = total_positions / max(total_segment_wall, 1e-9)
    including_startup = total_positions / max(total_rank_wall, 1e-9)
    return {
        "segments": [
            {"start_step": start, "end_step": end} for start, end in sorted(segments)
        ],
        "paired_positions": total_positions,
        "positions_per_second_excluding_startup": excluding_startup,
        "positions_per_second_with_smoke_checkpoint_cadence": (
            total_positions / max(total_segment_wall_with_smoke_checkpoints, 1e-9)
        ),
        "positions_per_second_including_startup": including_startup,
        "optimizer_steps_per_second_excluding_startup": (
            total_positions / 4096 / max(total_segment_wall, 1e-9)
        ),
        "producer_forward_seconds": sum(
            float(row["capture_forward_seconds"]) for row in producer
        ),
        "producer_communication_seconds": sum(
            float(row["communication_seconds"]) for row in producer
        ),
        "trainer_step_seconds": sum(float(row["trainer_step_seconds"]) for row in trainers),
        "checkpoint_write_seconds": total_checkpoint_wall,
        "peak_gpu_memory_bytes_by_rank": {
            str(rank): max(
                int(row["peak_gpu_memory_bytes"])
                for row in records
                if int(row["rank"]) == rank
            )
            for rank in (0, 1, 2)
        },
        "peak_cpu_rss_kib_by_rank": {
            str(rank): max(
                int(row["peak_cpu_rss_kib"])
                for row in records
                if int(row["rank"]) == rank
            )
            for rank in (0, 1, 2)
        },
    }


def verify_original_run(
    cfg: MultiModelExperimentConfig,
    run_dir: str | Path,
    *,
    mode: str,
    output_path: str | Path,
    evaluation_path: str | Path | None = None,
    minimum_positions_per_second: float = 4_500.0,
) -> dict[str, Any]:
    """Verify a 100-step smoke or completed 24,414-step scientific run."""

    validate_original_recipe_config(cfg)
    if mode not in {"smoke", "main"}:
        raise ValueError("mode must be smoke or main")
    run = Path(run_dir)
    expected_step = 100 if mode == "smoke" else cfg.training.max_steps
    expected_exposures = expected_step * 4096
    summary = _json(run / "training_summary.json")
    if int(summary.get("final_step", -1)) != expected_step:
        raise ValueError("training summary has the wrong final step")
    if int(summary.get("total_token_exposures", -1)) != expected_exposures:
        raise ValueError("training summary has the wrong exposure count")
    if summary.get("config_sha256") != resolved_config_sha256(cfg):
        raise ValueError("training summary has the wrong config fingerprint")
    if not summary.get("losses_finite") or not summary.get("gradients_finite"):
        raise FloatingPointError("training summary lacks finite loss/gradient evidence")
    checkpoint = _checkpoint_pair(run, expected_step, resolved_config_sha256(cfg), cfg)
    rows = _metrics(
        run,
        expected_step,
        expected_exposures,
        log_every=cfg.training.log_every,
    )
    runtime = _runtime(run)
    provenances = [_json(path) for path in sorted(run.glob("producer_provenance_steps_*.json"))]
    if not provenances or not all(row.get("source_llm_weights_unchanged") for row in provenances):
        raise ValueError("source-model weight-integrity evidence is missing")
    evaluation = None
    if evaluation_path is not None:
        evaluation = _json(Path(evaluation_path))
        if evaluation.get("scientific_status") != "HELD_OUT_ORIGINAL_RECIPE_EVALUATION":
            raise ValueError("evaluator did not produce the expected artifact")
        if evaluation.get("config_sha256") != resolved_config_sha256(cfg):
            raise ValueError("evaluator loaded a different scientific configuration")
        if evaluation.get("checkpoint", {}).get("step") != expected_step:
            raise ValueError("evaluator loaded a different checkpoint step")
    if mode == "smoke":
        if summary.get("start_step") != 50:
            raise ValueError("smoke final segment does not prove resume from step 50")
        for step in (50, 100):
            for rank in (0, 1):
                if not (run / f"checkpoint_{step:08d}.rank{rank}.pt").is_file():
                    raise FileNotFoundError(f"smoke checkpoint {step}/rank{rank} is missing")
        expected_segments = [
            {"start_step": 0, "end_step": 50},
            {"start_step": 50, "end_step": 100},
        ]
        if runtime["segments"] != expected_segments:
            raise ValueError("smoke runtime records do not prove the 50-step resume boundary")
        optimizer_update = _smoke_optimizer_update(run)
    else:
        optimizer_update = None
    sustained = float(runtime["positions_per_second_excluding_startup"])
    expected_main_exposures = cfg.training.expected_token_exposures
    if expected_main_exposures is None:
        raise ValueError("original-recipe verification requires expected_token_exposures")
    estimated_training_hours = expected_main_exposures / max(sustained, 1e-9) / 3600
    runtime["main_training_hours_extrapolated"] = estimated_training_hours
    runtime["main_training_plus_10_percent_overhead_hours"] = estimated_training_hours * 1.1
    go = sustained >= minimum_positions_per_second
    status = (
        ("GO" if go else "NO_GO_RUNTIME")
        if mode == "smoke"
        else "100M_TRAINING_COMPLETE"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "scientific_run": mode == "main",
        "final_step": expected_step,
        "token_exposures": expected_exposures,
        "minimum_positions_per_second": minimum_positions_per_second,
        "checkpoint": checkpoint,
        "metrics_rows": len(rows),
        "losses_finite": True,
        "gradients_finite": True,
        "runtime": runtime,
        "evaluation_load_verified": evaluation is not None,
        "optimizer_update": optimizer_update,
        "source_llm_weights_unchanged": True,
        "disk_bytes": sum(path.stat().st_size for path in run.rglob("*") if path.is_file()),
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite verification artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

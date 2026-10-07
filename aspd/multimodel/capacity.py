"""Post-run timing, resource, and sanity report for the main-size capacity pilot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from aspd.multimodel.plotting import load_training_metrics


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _nested(value: dict[str, Any], *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _read_gpu_telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"samples": 0}
    utilization: list[float] = []
    memory_used: list[float] = []
    memory_total: list[float] = []
    power: list[float] = []
    gpu_names: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpu_names.add(row.get("name", "unknown").strip())
            for target, key in (
                (utilization, "utilization_gpu_percent"),
                (memory_used, "memory_used_mib"),
                (memory_total, "memory_total_mib"),
                (power, "power_draw_w"),
            ):
                try:
                    target.append(float(row[key].strip()))
                except (KeyError, TypeError, ValueError):
                    pass
    peak_used = max(memory_used, default=None)
    total = max(memory_total, default=None)
    headroom = None
    if peak_used is not None and total:
        headroom = (total - peak_used) / total
    return {
        "samples": len(utilization),
        "gpu_names": sorted(gpu_names),
        "utilization_mean_percent": statistics.fmean(utilization) if utilization else None,
        "utilization_p10_percent": _percentile(utilization, 0.1),
        "utilization_median_percent": statistics.median(utilization) if utilization else None,
        "utilization_p90_percent": _percentile(utilization, 0.9),
        "peak_memory_used_gib": peak_used / 1024 if peak_used is not None else None,
        "memory_total_gib": total / 1024 if total is not None else None,
        "memory_headroom_fraction": headroom,
        "power_mean_w": statistics.fmean(power) if power else None,
        "power_peak_w": max(power, default=None),
    }


def build_capacity_report(run_dir: str | Path) -> dict[str, Any]:
    """Measure isolated overheads and evaluate the pilot's mechanical pass criteria."""

    run_dir = Path(run_dir)
    rows = load_training_metrics(run_dir)
    train_by_step = {
        int(row["step"]): row
        for row in rows
        if _number(row.get("step")) is not None and "train/loss" in row
    }
    elapsed = {
        step: value
        for step, row in train_by_step.items()
        if (value := _number(row.get("elapsed_seconds"))) is not None
    }

    steady_deltas: list[float] = []
    for step in sorted(elapsed):
        in_steady_window = 10 <= step <= 49 or 55 <= step <= 74 or 80 <= step <= 99
        if in_steady_window and step - 1 in elapsed:
            steady_deltas.append(elapsed[step] - elapsed[step - 1])
    train_seconds = statistics.median(steady_deltas) if steady_deltas else None

    def event_overhead(after_step: int) -> float | None:
        if train_seconds is None or after_step not in elapsed or after_step - 1 not in elapsed:
            return None
        return max(0.0, elapsed[after_step] - elapsed[after_step - 1] - train_seconds)

    validation_seconds = event_overhead(51)
    checkpoint_seconds = event_overhead(76)
    estimate_seconds = None
    if train_seconds is not None and validation_seconds is not None and checkpoint_seconds is not None:
        estimate_seconds = 10_000 * train_seconds + 40 * validation_seconds + 10 * checkpoint_seconds

    config_path = run_dir / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config_checks = {
        "linear_encoder": _nested(config, "encoder", "kind") == "linear",
        "mean_aggregation": _nested(config, "encoder", "aggregation") == "mean",
        "d0": _nested(config, "sparsity", "diffing") == "D0",
        "s1": _nested(config, "sparsity", "selection_score") == "S1",
        "c_8192": _nested(config, "sparsity", "n_features") == 8192,
        "k_32": _nested(config, "sparsity", "top_k") == 32,
        "batch_4": _nested(config, "training", "batch_size_sequences") == 4,
        "sequence_256": _nested(config, "data", "sequence_length") == 256,
        "fp32_parameters": _nested(config, "training", "parameter_dtype") == "float32",
        "bf16_autocast": _nested(config, "training", "autocast") == "bfloat16",
        "dead_after_2000": _nested(config, "objective", "dead_after_batches") == 2000,
        "not_resuming": _nested(config, "training", "resume") is None,
    }

    train_rows = list(train_by_step.values())
    finite_values = [
        float(value)
        for row in train_rows
        for key, value in row.items()
        if key.startswith("train/")
        and any(token in key for token in ("loss", "act/", "internal/"))
        and _number(value) is not None
    ]
    l0_values = [
        float(row["train/sparsity/l0"])
        for row in train_rows
        if _number(row.get("train/sparsity/l0")) is not None
    ]
    dead_values = [
        float(row["train/sparsity/dead_fraction"])
        for row in train_rows
        if _number(row.get("train/sparsity/dead_fraction")) is not None
    ]
    aux_values = [
        float(row["train/loss/auxk"])
        for row in train_rows
        if _number(row.get("train/loss/auxk")) is not None
    ]
    all_keys = {key for row in rows for key in row}
    d2_keys = sorted(
        key
        for key in all_keys
        if "/shared/" in key
        or key.endswith("loss/shared")
        or key.endswith("l0_shared")
        or key.endswith("l0_exclusive")
    )
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    telemetry = _read_gpu_telemetry(run_dir / "gpu_telemetry.csv")
    allocated_vram = [
        float(row["cuda/max_memory_gib"])
        for row in train_rows
        if _number(row.get("cuda/max_memory_gib")) is not None
    ]

    checks: dict[str, bool] = {
        **config_checks,
        "completed_100_steps": max(train_by_step, default=0) >= 100,
        "finite_scientific_metrics": bool(finite_values) and all(math.isfinite(x) for x in finite_values),
        "l0_is_32": bool(l0_values) and max(abs(value - 32.0) for value in l0_values) < 0.05,
        "dead_fraction_zero": bool(dead_values) and max(dead_values) == 0,
        "auxk_zero": bool(aux_values) and max(abs(value) for value in aux_values) < 1e-12,
        "no_d2_metrics": not d2_keys,
        "validation_completed": any("validation/loss" in row for row in rows),
        "two_checkpoints_written": len(checkpoints) >= 2,
        "steady_timing_available": train_seconds is not None,
        "validation_timing_available": validation_seconds is not None,
        "checkpoint_timing_available": checkpoint_seconds is not None,
        "gpu_telemetry_available": telemetry["samples"] > 0,
    }
    headroom = telemetry.get("memory_headroom_fraction")
    if headroom is not None:
        checks["vram_headroom_at_least_10_percent"] = bool(headroom >= 0.10)
    median_util = telemetry.get("utilization_median_percent")
    if median_util is not None:
        checks["median_gpu_util_at_least_70_percent"] = bool(median_util >= 70)

    return {
        "status": "PASS" if all(checks.values()) else "INVESTIGATE",
        "run_dir": str(run_dir),
        "timing": {
            "steady_train_step_seconds": train_seconds,
            "steady_delta_samples": len(steady_deltas),
            "validation_32_batches_seconds": validation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "main_10k_estimate_seconds": estimate_seconds,
            "main_10k_estimate_hours": estimate_seconds / 3600 if estimate_seconds else None,
            "main_10k_with_10pct_buffer_hours": (
                estimate_seconds * 1.1 / 3600 if estimate_seconds else None
            ),
        },
        "gpu": {
            **telemetry,
            "torch_peak_allocated_gib": max(allocated_vram, default=None),
        },
        "artifacts": {
            "checkpoints": [
                {"path": str(path), "size_gib": path.stat().st_size / 2**30}
                for path in checkpoints
            ],
            "resource_usage": str(run_dir / "resource_usage.txt"),
            "gpu_telemetry": str(run_dir / "gpu_telemetry.csv"),
        },
        "observed": {
            "max_step": max(train_by_step, default=0),
            "l0_min": min(l0_values, default=None),
            "l0_max": max(l0_values, default=None),
            "dead_fraction_max": max(dead_values, default=None),
            "auxk_abs_max": max((abs(value) for value in aux_values), default=None),
            "unexpected_d2_keys": d2_keys,
        },
        "checks": checks,
    }


def format_capacity_report(report: dict[str, Any]) -> str:
    """Return a compact human-readable summary for the Slurm log."""

    timing = report["timing"]
    gpu = report["gpu"]

    def show(value: object, suffix: str = "") -> str:
        return "n/a" if value is None else f"{float(value):.3f}{suffix}"

    lines = [
        f"capacity pilot: {report['status']}",
        f"steady train step: {show(timing['steady_train_step_seconds'], ' s')}",
        f"validation (32 batches): {show(timing['validation_32_batches_seconds'], ' s')}",
        f"checkpoint: {show(timing['checkpoint_seconds'], ' s')}",
        f"10k estimate (+10%): {show(timing['main_10k_with_10pct_buffer_hours'], ' h')}",
        f"torch peak allocated: {show(gpu['torch_peak_allocated_gib'], ' GiB')}",
        f"GPU utilization median: {show(gpu.get('utilization_median_percent'), '%')}",
        f"GPU memory headroom: {show((gpu.get('memory_headroom_fraction') or 0) * 100 if gpu.get('memory_headroom_fraction') is not None else None, '%')}",
        "checks:",
    ]
    lines.extend(
        f"  {'PASS' if passed else 'FAIL'} {name}"
        for name, passed in report["checks"].items()
    )
    return "\n".join(lines)

"""Post-training diagnostics for the frozen two-model original-ASPD run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.config import (
    EncoderSpec,
    ObjectiveSpec,
    OriginalASPDRecipeSpec,
    SparsitySpec,
)
from aspd.multimodel.model import MultiModelASPD

MATRYOSHKA_BOUNDARIES = (0, 2_304, 4_608, 9_216, 18_432, 36_864)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_result_package(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "result_package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": artifact["path"],
                "exists": path.is_file(),
                "expected_bytes": artifact["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual_sha256,
                "matches": (
                    actual_bytes == artifact["bytes"]
                    and actual_sha256 == artifact["sha256"]
                ),
            }
        )
    return {
        "manifest_status": manifest["status"],
        "artifact_count": len(rows),
        "all_match": all(row["matches"] for row in rows),
        "artifacts": rows,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "first": values[0],
        "last": values[-1],
    }


def _linear_trend(rows: list[dict[str, Any]], key: str, start_step: int) -> dict[str, float | int]:
    points = [(float(row["step"]), float(row[key])) for row in rows if row["step"] >= start_step]
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "start_step": start_step,
        "rows": len(points),
        "slope_per_1000_steps": slope * 1000,
        "fitted_change": slope * (points[-1][0] - points[0][0]),
        "r_squared": 1.0 - residual / total if total else 0.0,
    }


def analyze_training_metrics(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys = (
        "train/internal/base",
        "train/internal/target",
        "train/act/base",
        "train/act/target",
        "train/loss",
        "train/loss/auxk",
        "train/sparsity/jump_relu_threshold",
    )
    ranges = (
        (1, 100),
        (101, 1_000),
        (1_001, 5_000),
        (5_001, 10_000),
        (10_001, 20_000),
        (20_001, 23_414),
        (23_415, 24_414),
    )
    segments = []
    for start, stop in ranges:
        selected = [row for row in rows if start <= row["step"] <= stop]
        segments.append(
            {
                "start_step": start,
                "stop_step": stop,
                "rows": len(selected),
                "metrics": {
                    key: _summary([float(row[key]) for row in selected]) for key in keys
                },
            }
        )
    trend_keys = keys[:5]
    return {
        "rows": len(rows),
        "first_step": rows[0]["step"],
        "last_logged_step": rows[-1]["step"],
        "segments": segments,
        "trends": {
            key: [
                _linear_trend(rows, key, start)
                for start in (1, 1_001, 10_001, 20_001, 23_415)
            ]
            for key in trend_keys
        },
        "auxk_nonzero_rows": sum(float(row["train/loss/auxk"]) != 0 for row in rows),
        "dead_fraction_nonzero_rows": sum(
            float(row["train/sparsity/dead_fraction"]) != 0 for row in rows
        ),
        "raw_rows": rows,
    }


def _component_group(
    rows: list[dict[str, str]],
    start: int,
    stop: int,
    total_fires: int,
    total_base_energy: float,
    total_target_energy: float,
    ranked_ids: list[int],
) -> dict[str, Any]:
    selected = rows[start:stop]
    fires = sum(int(row["held_out_fire_count"]) for row in selected)
    base_energy = sum(
        float(row["base_contribution_rms_per_validation_token"]) ** 2
        for row in selected
    )
    target_energy = sum(
        float(row["target_contribution_rms_per_validation_token"]) ** 2
        for row in selected
    )
    return {
        "start": start,
        "stop_exclusive": stop,
        "features": stop - start,
        "active_features": sum(int(row["held_out_fire_count"]) > 0 for row in selected),
        "fire_count": fires,
        "fire_share": fires / total_fires,
        "base_contribution_energy": base_energy,
        "base_contribution_energy_share": base_energy / total_base_energy,
        "target_contribution_energy": target_energy,
        "target_contribution_energy_share": target_energy / total_target_energy,
        "base_contribution_rms_across_features": math.sqrt(base_energy / len(selected)),
        "target_contribution_rms_across_features": math.sqrt(target_energy / len(selected)),
        "base_read_norm": _summary([float(row["base_read_norm"]) for row in selected]),
        "target_read_norm": _summary([float(row["target_read_norm"]) for row in selected]),
        "base_write_norm": _summary([float(row["base_write_norm"]) for row in selected]),
        "target_write_norm": _summary([float(row["target_write_norm"]) for row in selected]),
        "top_20_candidates": sum(start <= component < stop for component in ranked_ids[:20]),
        "top_100_candidates": sum(start <= component < stop for component in ranked_ids[:100]),
    }


def analyze_component_metrics(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != MATRYOSHKA_BOUNDARIES[-1]:
        raise ValueError(f"expected 36,864 component rows, found {len(rows):,}")
    total_fires = sum(int(row["held_out_fire_count"]) for row in rows)
    total_base_energy = sum(
        float(row["base_contribution_rms_per_validation_token"]) ** 2 for row in rows
    )
    total_target_energy = sum(
        float(row["target_contribution_rms_per_validation_token"]) ** 2 for row in rows
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["base_contribution_rms_per_validation_token"])
            + float(row["target_contribution_rms_per_validation_token"])
        ),
        reverse=True,
    )
    ranked_ids = [int(row["component_id"]) for row in ranked]
    groups = [
        _component_group(
            rows,
            start,
            stop,
            total_fires,
            total_base_energy,
            total_target_energy,
            ranked_ids,
        )
        for start, stop in zip(
            MATRYOSHKA_BOUNDARIES[:-1], MATRYOSHKA_BOUNDARIES[1:], strict=True
        )
    ]
    cumulative = []
    for stop in MATRYOSHKA_BOUNDARIES[1:]:
        prefix_groups = [group for group in groups if group["stop_exclusive"] <= stop]
        cumulative.append(
            {
                "stop_exclusive": stop,
                "feature_fraction": stop / len(rows),
                "fire_share": sum(group["fire_share"] for group in prefix_groups),
                "base_contribution_energy_share": sum(
                    group["base_contribution_energy_share"] for group in prefix_groups
                ),
                "target_contribution_energy_share": sum(
                    group["target_contribution_energy_share"] for group in prefix_groups
                ),
                "top_20_candidates": sum(component < stop for component in ranked_ids[:20]),
                "top_100_candidates": sum(component < stop for component in ranked_ids[:100]),
            }
        )
    return {
        "component_count": len(rows),
        "total_fire_count": total_fires,
        "total_base_standalone_contribution_energy": total_base_energy,
        "total_target_standalone_contribution_energy": total_target_energy,
        "groups": groups,
        "cumulative_prefixes": cumulative,
        "top_20_component_ids": ranked_ids[:20],
        "top_100_component_ids": ranked_ids[:100],
    }


def _parameter_category(name: str) -> str:
    exact = {
        "encoder.projections.0.weight": "shared_encoder_projection_base",
        "encoder.projections.1.weight": "shared_encoder_projection_target",
        "encoder.bias": "shared_encoder_bias",
        "encoder.centers.0": "base_center_and_activation_bias",
        "encoder.centers.1": "target_center_and_activation_bias",
        "activation_decoders.full.0": "base_activation_decoder",
        "activation_decoders.full.1": "target_activation_decoder",
    }
    if name in exact:
        return exact[name]
    for index, label in ((0, "base"), (1, "target")):
        prefix = f"components.{index}.o_proj."
        if name.startswith(prefix):
            return f"{label}_component_{name.removeprefix(prefix).lower()}"
    return "other"


def summarize_gradients(model: MultiModelASPD) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, Tensor]]] = {}
    for name, parameter in model.named_parameters():
        grouped.setdefault(_parameter_category(name), []).append((name, parameter))
    output = {}
    for category, parameters in grouped.items():
        gradients = [parameter.grad for _, parameter in parameters]
        with_gradient = [gradient for gradient in gradients if gradient is not None]
        squared_norm = sum(
            float(gradient.detach().double().square().sum()) for gradient in with_gradient
        )
        output[category] = {
            "parameter_names": [name for name, _ in parameters],
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(parameter.numel() for _, parameter in parameters),
            "gradient_none_tensors": sum(gradient is None for gradient in gradients),
            "gradient_zero_tensors": sum(
                gradient is not None and not bool(torch.count_nonzero(gradient))
                for gradient in gradients
            ),
            "gradient_nonzero_tensors": sum(
                gradient is not None and bool(torch.count_nonzero(gradient))
                for gradient in gradients
            ),
            "gradient_l2_norm": math.sqrt(squared_norm),
            "gradient_max_abs": max(
                (float(gradient.detach().abs().max()) for gradient in with_gradient),
                default=0.0,
            ),
        }
    return output


def objective_terms(
    model: MultiModelASPD, batch: dict[str, object], *, force_dead: bool = False
) -> dict[str, Tensor]:
    assert model.sparsity_controller is not None
    if force_dead:
        model.sparsity_controller.n_pools_not_active.fill_(
            model.sparsity_controller.dead_after_pools
        )
    valid: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
    activations: list[Tensor] = batch["R"]  # type: ignore[assignment]
    inputs: list[dict[str, Tensor]] = batch["X"]  # type: ignore[assignment]
    code = model.encode(activations, valid, update_statistics=False)
    activation_losses = []
    internal_losses = []
    reconstructions = []
    for model_index in range(len(model.model_names)):
        activation, reconstruction = model._activation_loss(
            model_index,
            activations[model_index],
            code.values,
            valid,
            model.sparsity_cfg.n_features,
        )
        internal, _ = model._internal_loss(
            model_index,
            inputs[model_index],
            code.gate,
            valid,
            shared_only=False,
        )
        activation_losses.append(activation)
        internal_losses.append(internal)
        reconstructions.append(reconstruction)
    activation = torch.stack(activation_losses).mean()
    internal = torch.stack(internal_losses).mean()
    auxk = model._auxk_loss(activations, reconstructions, code.preactivations, valid)
    full = internal + model.objective_cfg.lambda_act * activation
    full = full + model.objective_cfg.auxk_coefficient * auxk
    return {"activation": activation, "internal": internal, "auxk": auxk, "full": full}


def gradient_audit(model: MultiModelASPD, batch: dict[str, object]) -> dict[str, Any]:
    assert model.sparsity_controller is not None
    saved_dead_counters = model.sparsity_controller.n_pools_not_active.clone()
    probes: tuple[tuple[str, str, bool], ...] = (
        ("activation_loss_alone", "activation", False),
        ("internal_loss_alone", "internal", False),
        ("auxk_actual_dead_state", "auxk", False),
        ("auxk_forced_dead_path_probe", "auxk", True),
        ("full_combined_objective", "full", False),
    )
    results = {}
    try:
        for label, term_name, force_dead in probes:
            model.sparsity_controller.n_pools_not_active.copy_(saved_dead_counters)
            model.zero_grad(set_to_none=True)
            terms = objective_terms(model, batch, force_dead=force_dead)
            loss = terms[term_name]
            loss.backward()
            results[label] = {
                "optimized_term": term_name,
                "forced_dead_counter_probe": force_dead,
                "term_values": {name: float(value.detach()) for name, value in terms.items()},
                "gradients": summarize_gradients(model),
            }
    finally:
        model.sparsity_controller.n_pools_not_active.copy_(saved_dead_counters)
        model.zero_grad(set_to_none=True)
    return results


def _synthetic_model(seed: int) -> tuple[MultiModelASPD, dict[str, object]]:
    torch.manual_seed(seed)
    activation_width = 8
    matrix_width = 6
    model = MultiModelASPD(
        ["base", "target"],
        [activation_width, activation_width],
        [
            {"o_proj": torch.randn(matrix_width, matrix_width)},
            {"o_proj": torch.randn(matrix_width, matrix_width)},
        ],
        EncoderSpec(kind="linear", aggregation="mean"),
        SparsitySpec(n_features=32, top_k=4, pool_tokens=8),
        ObjectiveSpec(
            lambda_act=1.0,
            auxk_coefficient=0.03125,
            top_k_aux=8,
            dead_after_batches=2,
            alive_window_tokens=64,
            model_axis_reduction="mean",
            matryoshka_group_fractions=[0.0625, 0.0625, 0.125, 0.25, 0.5],
        ),
        OriginalASPDRecipeSpec(enabled=True, calibration_batches=1),
    ).train()
    residuals = [torch.randn(2, 4, activation_width) for _ in range(2)]
    model.calibrate_original_recipe([value.mean((0, 1)) for value in residuals])
    batch: dict[str, object] = {
        "valid_tokens": torch.ones(2, 4, dtype=torch.bool),
        "R": residuals,
        "X": [
            {"o_proj": torch.randn(2, 4, matrix_width)},
            {"o_proj": torch.randn(2, 4, matrix_width)},
        ],
    }
    return model, batch


def synthetic_gradient_audit(seed: int = 0) -> dict[str, Any]:
    model, batch = _synthetic_model(seed)
    return {
        "seed": seed,
        "fixture": {
            "models": 2,
            "tokens": 8,
            "activation_width": 8,
            "matrix_width": 6,
            "components": 32,
            "top_k": 4,
        },
        "center_bias_alias_verified": (
            model.activation_decoders.biases[0] is model.encoder.centers[0]
            and model.activation_decoders.biases[1] is model.encoder.centers[1]
        ),
        "probes": gradient_audit(model, batch),
    }


def build_cpu_audit(run_dir: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    training = analyze_training_metrics(run_dir / "training_metrics.jsonl")
    raw_rows = training.pop("raw_rows")
    held_out = json.loads(
        (run_dir / "evaluation/held_out_evaluation.json").read_text(encoding="utf-8")
    )
    delta = json.loads(
        (run_dir / "evaluation/delta_computation_metrics.json").read_text(encoding="utf-8")
    )
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (run_dir / "checkpoint_verification_report.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "training_summary": summary,
        "checkpoint_verification": verification,
        "package_verification": verify_result_package(run_dir) if verify_hashes else None,
        "training_trajectory": training,
        "held_out_reference": {
            "scientific_status": held_out["scientific_status"],
            "validation_scope": held_out["validation_scope"],
            "groups_all": held_out["groups"]["all"],
            "sparsity": held_out["sparsity"],
        },
        "delta_computation": delta,
        "matryoshka_concentration": analyze_component_metrics(
            run_dir / "evaluation/component_metrics.csv"
        ),
        "synthetic_gradient_audit": synthetic_gradient_audit(),
    }
    report["_training_rows_for_plot"] = raw_rows
    return report


def write_training_plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, label in (("base", "Base"), ("target", "Target")):
        axes[0].plot(
            steps,
            [row[f"train/internal/{name}"] for row in rows],
            label=f"{label} internal FVU",
            linewidth=0.9,
        )
        axes[1].plot(
            steps,
            [row[f"train/act/{name}"] for row in rows],
            label=f"{label} Matryoshka activation loss",
            linewidth=0.9,
        )
    axes[0].set_ylabel("Internal loss")
    axes[1].set_ylabel("Activation loss")
    axes[1].set_xlabel("Optimizer step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Qwen2.5 two-model ASPD 100M objective trajectory")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_cpu_audit(
    run_dir: Path, output_dir: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_cpu_audit(run_dir, verify_hashes=verify_hashes)
    rows = report.pop("_training_rows_for_plot")
    write_training_plot(rows, output_dir / "training_objective_trajectory.png")
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report

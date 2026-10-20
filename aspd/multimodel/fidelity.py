"""Checkpoint-agnostic computation-fidelity measurements for multi-model ASPD.

The evaluator in this module consumes only an ASPD checkpoint and the validated
paired cache.  It never loads either source language model.  Matrix targets are
therefore explicitly the weight-only linear computations saved in the cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.training import _move_batch, build_model_from_cache


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_config_sha256(cfg: MultiModelExperimentConfig) -> str:
    payload = json.dumps(
        cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_immutable(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically create a JSON artifact, refusing to replace prior evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite concurrently-created artifact: {destination}"
        ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@dataclass
class GlobalTensorMetrics:
    """FP64 sufficient statistics for dataset-level reconstruction metrics.

    FVU needs the global per-output-coordinate target mean.  Keeping ``sum_y``
    avoids retaining any token rows and, unlike averaging batch ratios, gives the
    exact dataset-level denominator.
    """

    output_dim: int
    stability_relative_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        self.rows = 0
        self.elements = 0
        self.sum_y = torch.zeros(self.output_dim, dtype=torch.float64)
        self.target_sq = 0.0
        self.predicted_sq = 0.0
        self.error_sq = 0.0
        self.dot = 0.0

    @torch.no_grad()
    def update(self, target: Tensor, prediction: Tensor, valid_tokens: Tensor) -> None:
        if target.shape != prediction.shape:
            raise ValueError(
                f"target/prediction shape mismatch: {tuple(target.shape)} != {tuple(prediction.shape)}"
            )
        if target.ndim != 3 or target.shape[:2] != valid_tokens.shape:
            raise ValueError("expected [batch,tokens,output] tensors and matching token mask")
        if target.shape[-1] != self.output_dim:
            raise ValueError(
                f"output dimension changed: {target.shape[-1]} != {self.output_dim}"
            )
        y = target[valid_tokens].detach().to(device="cpu", dtype=torch.float64)
        y_hat = prediction[valid_tokens].detach().to(device="cpu", dtype=torch.float64)
        if y.numel() == 0:
            return
        if not torch.isfinite(y).all() or not torch.isfinite(y_hat).all():
            raise FloatingPointError("non-finite target or reconstruction")
        error = y_hat - y
        self.rows += y.shape[0]
        self.elements += y.numel()
        self.sum_y += y.sum(dim=0)
        self.target_sq += float(torch.sum(y * y))
        self.predicted_sq += float(torch.sum(y_hat * y_hat))
        self.error_sq += float(torch.sum(error * error))
        self.dot += float(torch.sum(y * y_hat))

    def result(self) -> dict[str, Any]:
        if self.rows == 0 or self.elements == 0:
            raise ValueError("cannot finalize empty metric accumulator")
        variance = self.target_sq - float(torch.sum(self.sum_y * self.sum_y)) / self.rows
        energy_scale = max(self.target_sq, 1.0)
        stable_energy = self.target_sq > self.stability_relative_epsilon * self.elements
        stable_variance = variance > self.stability_relative_epsilon * energy_scale
        cosine_denominator = math.sqrt(self.target_sq * self.predicted_sq)
        stable_cosine = cosine_denominator > self.stability_relative_epsilon * energy_scale
        normalized = self.error_sq / self.target_sq if stable_energy else None
        values: dict[str, Any] = {
            "rows": self.rows,
            "elements": self.elements,
            "sse": self.error_sq,
            "rmse": math.sqrt(self.error_sq / self.elements),
            "relative_rms_error": math.sqrt(normalized) if normalized is not None else None,
            "cosine_similarity": self.dot / cosine_denominator if stable_cosine else None,
            "actual_output_rms": math.sqrt(self.target_sq / self.elements),
            "reconstructed_output_rms": math.sqrt(self.predicted_sq / self.elements),
            "normalized_squared_error": normalized,
            "error_improvement_over_zero": 1.0 - normalized if normalized is not None else None,
            "fvu": self.error_sq / variance if stable_variance else None,
            "target_energy": self.target_sq,
            "variance_denominator": variance,
            "energy_well_conditioned": stable_energy,
            "variance_well_conditioned": stable_variance,
            "cosine_well_conditioned": stable_cosine,
        }
        for value in values.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError("non-finite finalized metric")
        return values


@dataclass
class IdentityResidual:
    """Streaming error for ``total == parameter + input`` identities."""

    squared_error: float = 0.0
    reference_squared: float = 0.0
    elements: int = 0
    max_abs: float = 0.0

    @torch.no_grad()
    def update(self, total: Tensor, parameter: Tensor, input_change: Tensor, mask: Tensor) -> None:
        residual = (total - parameter - input_change)[mask].detach().float()
        reference = total[mask].detach().float()
        if not torch.isfinite(residual).all() or not torch.isfinite(reference).all():
            raise FloatingPointError("non-finite decomposition identity")
        self.squared_error += float(residual.double().square().sum())
        self.reference_squared += float(reference.double().square().sum())
        self.elements += residual.numel()
        if residual.numel():
            self.max_abs = max(self.max_abs, float(residual.abs().max()))

    def result(self) -> dict[str, Any]:
        scale = max(self.reference_squared, 1e-30)
        return {
            "max_abs_residual": self.max_abs,
            "rms_residual": math.sqrt(self.squared_error / max(self.elements, 1)),
            "relative_rms_residual": math.sqrt(self.squared_error / scale),
            "elements": self.elements,
        }


@dataclass
class PairedErrorMetrics:
    """Streaming diagnostics for cancellation between Base and Target errors."""

    base_error_sq: float = 0.0
    target_error_sq: float = 0.0
    error_dot: float = 0.0
    true_delta_sq: float = 0.0
    predicted_delta_sq: float = 0.0
    delta_error_sq: float = 0.0
    elements: int = 0

    @torch.no_grad()
    def update(
        self,
        base_target: Tensor,
        base_prediction: Tensor,
        target_target: Tensor,
        target_prediction: Tensor,
        valid_tokens: Tensor,
    ) -> None:
        shapes = {
            tuple(base_target.shape),
            tuple(base_prediction.shape),
            tuple(target_target.shape),
            tuple(target_prediction.shape),
        }
        if len(shapes) != 1 or base_target.shape[:2] != valid_tokens.shape:
            raise ValueError("paired error tensors or valid-token mask do not align")
        base_error = (base_prediction - base_target)[valid_tokens].detach().double().cpu()
        target_error = (
            target_prediction - target_target
        )[valid_tokens].detach().double().cpu()
        true_delta = (target_target - base_target)[valid_tokens].detach().double().cpu()
        predicted_delta = (
            target_prediction - base_prediction
        )[valid_tokens].detach().double().cpu()
        if not all(
            torch.isfinite(value).all()
            for value in (base_error, target_error, true_delta, predicted_delta)
        ):
            raise FloatingPointError("non-finite paired reconstruction error")
        delta_error = predicted_delta - true_delta
        self.base_error_sq += float(base_error.square().sum())
        self.target_error_sq += float(target_error.square().sum())
        self.error_dot += float((base_error * target_error).sum())
        self.true_delta_sq += float(true_delta.square().sum())
        self.predicted_delta_sq += float(predicted_delta.square().sum())
        self.delta_error_sq += float(delta_error.square().sum())
        self.elements += base_error.numel()

    def result(self) -> dict[str, Any]:
        if self.elements == 0:
            raise ValueError("cannot finalize empty paired error metrics")
        error_norm_product = math.sqrt(self.base_error_sq * self.target_error_sq)
        return {
            "elements": self.elements,
            "base_error_rms": math.sqrt(self.base_error_sq / self.elements),
            "target_error_rms": math.sqrt(self.target_error_sq / self.elements),
            "error_cosine_similarity": (
                self.error_dot / error_norm_product if error_norm_product > 0 else None
            ),
            "true_delta_rms": math.sqrt(self.true_delta_sq / self.elements),
            "predicted_delta_rms": math.sqrt(
                self.predicted_delta_sq / self.elements
            ),
            "delta_error_rms": math.sqrt(self.delta_error_sq / self.elements),
            "delta_error_to_true_ratio": math.sqrt(
                self.delta_error_sq / max(self.true_delta_sq, 1e-30)
            ),
            "cancellation_gain_vs_independent_errors": (
                1.0
                - self.delta_error_sq
                / max(self.base_error_sq + self.target_error_sq, 1e-30)
            ),
        }


def validate_matrix_alignment(
    cfg: MultiModelExperimentConfig, cache: PairedActivationCache
) -> list[str]:
    if len(cfg.models) != 2:
        raise ValueError("DeltaY fidelity currently requires exactly Base and Target models")
    if [model.name for model in cfg.models] != ["base", "target"]:
        raise ValueError("model axis must be ordered exactly as ['base', 'target']")
    matrix_names = [matrix.name for matrix in cfg.models[0].matrices]
    if [matrix.name for matrix in cfg.models[1].matrices] != matrix_names:
        raise ValueError("Base/Target matrix names or ordering differ")
    reference = cache.model_manifests[0]["matrices"]
    target = cache.model_manifests[1]["matrices"]
    if reference != target:
        raise ValueError("Base/Target cached matrix dimensions differ")
    if set(matrix_names) != set(reference):
        raise ValueError("config/cache matrix sets differ")
    return matrix_names


def validate_checkpoint_evaluation_compatibility(
    evaluation_cfg: MultiModelExperimentConfig,
    checkpoint_cfg: MultiModelExperimentConfig,
) -> None:
    """Allow new evaluation tokens without changing the learned scientific object."""

    checks = {
        "models": (evaluation_cfg.models, checkpoint_cfg.models),
        "encoder": (evaluation_cfg.encoder, checkpoint_cfg.encoder),
        "sparsity": (evaluation_cfg.sparsity, checkpoint_cfg.sparsity),
        "objective": (evaluation_cfg.objective, checkpoint_cfg.objective),
        "tokenizer": (evaluation_cfg.data.tokenizer, checkpoint_cfg.data.tokenizer),
        "tokenizer_revision": (
            evaluation_cfg.data.tokenizer_revision,
            checkpoint_cfg.data.tokenizer_revision,
        ),
        "sequence_length": (
            evaluation_cfg.data.sequence_length,
            checkpoint_cfg.data.sequence_length,
        ),
    }
    mismatches = [name for name, (left, right) in checks.items() if left != right]
    if mismatches:
        raise ValueError(
            "checkpoint and evaluation configs differ in scientific structure: "
            + ", ".join(mismatches)
        )


def validate_checkpoint_provenance(
    cfg: MultiModelExperimentConfig,
    cache: PairedActivationCache,
    checkpoint: str | Path,
    *,
    expected_cache_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_cache = cache.token_manifest.get("cache_fingerprint")
    if expected_cache_fingerprint and actual_cache != expected_cache_fingerprint:
        raise ValueError(
            f"cache fingerprint mismatch: {actual_cache} != {expected_cache_fingerprint}"
        )
    checkpoint_sha = sha256_file(checkpoint_path)
    if expected_checkpoint_sha256 and checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA256 mismatch: {checkpoint_sha} != {expected_checkpoint_sha256}"
        )
    provenance_path = checkpoint_path.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"checkpoint provenance is required: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checks = {
        "cache_fingerprint": (provenance.get("cache_fingerprint"), actual_cache),
        "config_sha256": (provenance.get("config_sha256"), resolved_config_sha256(cfg)),
        "model_ids": (provenance.get("model_ids"), [model.model_id for model in cfg.models]),
        "model_revisions": (
            provenance.get("model_revisions"),
            [model.revision for model in cfg.models],
        ),
        "latent_count_C": (provenance.get("latent_count_C"), cfg.sparsity.n_features),
        "model_count_N": (provenance.get("model_count_N"), len(cfg.models)),
    }
    mismatches = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
    if mismatches:
        raise ValueError(f"checkpoint/config/cache provenance mismatch: {mismatches}")
    return {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
        "provenance_path": str(provenance_path),
        "training_git_commit": provenance.get("git_commit"),
    }


def _component_reconstruction(
    model: MultiModelASPD,
    model_index: int,
    matrix_name: str,
    x: Tensor,
    gate: Tensor,
    valid_tokens: Tensor,
) -> Tensor:
    component = model.components[model_index][matrix_name.replace(".", "__dot__")]
    return component.reconstruct(x, gate, valid_tokens)  # type: ignore[no-any-return,attr-defined]


@torch.no_grad()
def evaluate_computation_fidelity(
    cfg: MultiModelExperimentConfig,
    checkpoint: str | Path,
    *,
    checkpoint_cfg: MultiModelExperimentConfig | None = None,
    device_name: str = "cuda",
    split: str = "validation",
    batch_size_sequences: int | None = None,
    expected_cache_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate activation, operator, and DeltaY fidelity with global statistics."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    cache = PairedActivationCache(cfg)
    training_cfg = checkpoint_cfg or cfg
    validate_checkpoint_evaluation_compatibility(cfg, training_cfg)
    training_cache = cache if training_cfg is cfg else PairedActivationCache(training_cfg)
    matrix_names = validate_matrix_alignment(cfg, cache)
    checkpoint_record = validate_checkpoint_provenance(
        training_cfg,
        training_cache,
        checkpoint,
        expected_cache_fingerprint=expected_cache_fingerprint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    model = build_model_from_cache(cfg, cache)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, Mapping) or "model" not in state or "step" not in state:
        raise ValueError("checkpoint lacks model or step")
    current_state = model.state_dict()
    for name, value in state["model"].items():
        if name.startswith("target_weights."):
            if name not in current_state or not torch.equal(value.cpu(), current_state[name].cpu()):
                raise ValueError(
                    f"checkpoint frozen weight {name!r} differs from evaluation cache snapshot"
                )
    model.load_state_dict(state["model"], strict=True)
    checkpoint_step = int(state["step"])
    # The optimizer state is irrelevant here and can be several GB.  Drop the
    # memory-mapped checkpoint mapping before moving the reconstructed model.
    del state
    model = model.to(device).float().eval()

    activation_stats = {
        name: GlobalTensorMetrics(cache.model_manifests[n]["activation_dim"])
        for n, name in enumerate(("base", "target"))
    }
    matrix_stats: dict[str, dict[str, GlobalTensorMetrics]] = {}
    identity_stats: dict[str, dict[str, IdentityResidual]] = {}
    paired_error_stats: dict[str, PairedErrorMetrics] = {}
    for name in matrix_names:
        d_out = cache.model_manifests[0]["matrices"][name]["d_out"]
        matrix_stats[name] = {
            label: GlobalTensorMetrics(d_out)
            for label in ("base", "target", "delta_total", "delta_parameter", "delta_input")
        }
        identity_stats[name] = {"true": IdentityResidual(), "aspd": IdentityResidual()}
        paired_error_stats[name] = PairedErrorMetrics()

    total_tokens = 0
    batches = 0
    batch_size = batch_size_sequences or cfg.training.batch_size_sequences
    normalization = [
        float(manifest["r_rms_norm"]) for manifest in training_cache.model_manifests
    ]
    for raw_batch in cache.iter_batches(
        split, batch_size, shuffle=False, normalize_grounding=False
    ):
        batch = _move_batch(raw_batch, device)
        valid: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
        activations: list[Tensor] = [
            value.float() / normalization[n]
            for n, value in enumerate(batch["R"])  # type: ignore[index]
        ]
        inputs: list[dict[str, Tensor]] = [
            {name: value.float() for name, value in by_matrix.items()}
            for by_matrix in batch["X"]  # type: ignore[index]
        ]
        code = model.encode(activations, valid)
        gate = code.gate
        for n, name in enumerate(("base", "target")):
            reconstruction = model.activation_decoders.reconstruct(
                code.values, n, valid
            ).float()
            activation_stats[name].update(activations[n], reconstruction, valid)

        for name in matrix_names:
            x_base, x_target = inputs[0][name], inputs[1][name]
            if x_base.shape != x_target.shape:
                raise ValueError(f"unaligned Base/Target inputs for {name}")
            w_base = model._target(0, name)
            w_target = model._target(1, name)
            y_base = w_base.apply(x_base).float()
            y_target = w_target.apply(x_target).float()
            yhat_base = _component_reconstruction(
                model, 0, name, x_base, gate, valid
            ).float()
            yhat_target = _component_reconstruction(
                model, 1, name, x_target, gate, valid
            ).float()
            matrix_stats[name]["base"].update(y_base, yhat_base, valid)
            matrix_stats[name]["target"].update(y_target, yhat_target, valid)
            paired_error_stats[name].update(
                y_base, yhat_base, y_target, yhat_target, valid
            )

            delta_total = y_target - y_base
            delta_total_hat = yhat_target - yhat_base
            delta_parameter = torch.nn.functional.linear(
                x_base, w_target.weight.float() - w_base.weight.float()
            )
            target_on_base_hat = _component_reconstruction(
                model, 1, name, x_base, gate, valid
            ).float()
            delta_parameter_hat = target_on_base_hat - yhat_base
            delta_x = x_target - x_base
            delta_input = w_target.apply(delta_x).float()
            delta_input_hat = _component_reconstruction(
                model, 1, name, delta_x, gate, valid
            ).float()
            matrix_stats[name]["delta_total"].update(
                delta_total, delta_total_hat, valid
            )
            matrix_stats[name]["delta_parameter"].update(
                delta_parameter, delta_parameter_hat, valid
            )
            matrix_stats[name]["delta_input"].update(
                delta_input, delta_input_hat, valid
            )
            identity_stats[name]["true"].update(
                delta_total, delta_parameter, delta_input, valid
            )
            identity_stats[name]["aspd"].update(
                delta_total_hat, delta_parameter_hat, delta_input_hat, valid
            )
        total_tokens += int(valid.sum())
        batches += 1

    expected_tokens = cache.token_manifest["requested_tokens"].get(split)
    if total_tokens != expected_tokens:
        raise ValueError(
            f"evaluated {total_tokens} {split} tokens, expected exactly {expected_tokens}"
        )
    matrices = {
        name: {
            "shape": cache.model_manifests[0]["matrices"][name],
            "target_semantics": "weight-only linear computation (bias excluded)",
            **{label: accumulator.result() for label, accumulator in stats.items()},
            "decomposition_identity": {
                label: accumulator.result()
                for label, accumulator in identity_stats[name].items()
            },
            "error_cancellation": paired_error_stats[name].result(),
        }
        for name, stats in matrix_stats.items()
    }
    max_identity_relative = max(
        matrices[name]["decomposition_identity"][kind]["relative_rms_residual"]
        for name in matrix_names
        for kind in ("true", "aspd")
    )
    internal_means = {
        model_name: sum(matrices[name][model_name]["fvu"] for name in matrix_names)
        / len(matrix_names)
        for model_name in ("base", "target")
    }
    return {
        "schema_version": 1,
        "kind": "aspd_computation_fidelity",
        "checkpoint": {**checkpoint_record, "step": checkpoint_step},
        "cache": {
            "root": str(Path(cfg.cache.root).resolve()),
            "fingerprint": cache.token_manifest["cache_fingerprint"],
        },
        "checkpoint_training_cache": {
            "root": str(Path(training_cfg.cache.root).resolve()),
            "fingerprint": training_cache.token_manifest["cache_fingerprint"],
            "grounding_r_rms_norm": normalization,
        },
        "config": {
            "resolved": cfg.model_dump(mode="json"),
            "sha256": resolved_config_sha256(cfg),
        },
        "checkpoint_training_config": {
            "resolved": training_cfg.model_dump(mode="json"),
            "sha256": resolved_config_sha256(training_cfg),
        },
        "evaluation": {
            "split": split,
            "tokens": total_tokens,
            "batches": batches,
            "batch_size_sequences": batch_size,
            "aggregation": "global FP64 sufficient statistics; ratios are not batch-averaged",
            "model_and_weight_compute_precision": "float32",
            "metric_accumulation_precision": "float64",
            "grounding_activations": "train-RMS-normalized cache values",
            "matrix_target_semantics": "weight-only linear computation (bias excluded)",
            "no_full_language_models_loaded": True,
        },
        "activation_reconstruction": {
            name: stats.result() for name, stats in activation_stats.items()
        },
        "internal_mean_matrix_fvu": internal_means,
        "matrices": matrices,
        "integrity": {
            "finite": True,
            "matrix_count": len(matrix_names),
            "matrix_names": matrix_names,
            "max_decomposition_relative_rms_residual": max_identity_relative,
        },
    }

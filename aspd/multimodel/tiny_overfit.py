"""Fail-closed fixed-batch optimization diagnostic for the ASPD objective."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.losses import fvu
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.training import _autocast_context, _move_batch, build_model_from_cache


def _scalar_stats(result: Mapping[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().float().cpu()) for name, value in result.items()}


@torch.no_grad()
def _fixed_batch_metrics(
    model: MultiModelASPD,
    batch: Mapping[str, object],
    cfg: MultiModelExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with _autocast_context(cfg, device):
        objective = model(batch, update_dead_tracker=False)
        valid: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
        activations: list[Tensor] = batch["R"]  # type: ignore[assignment]
        code = model.encode(activations, valid)
        full_activation_fvu = {
            f"full_activation_fvu/{name}": float(
                fvu(
                    activations[n],
                    model.activation_decoders.reconstruct(code.values, n, valid),
                    valid,
                )
                .detach()
                .float()
                .cpu()
            )
            for n, name in enumerate(model.model_names)
        }
    model.train()
    return {**_scalar_stats(objective), **full_activation_fvu}


def _gradient_norms(model: MultiModelASPD) -> dict[str, float]:
    squared = {"encoder": 0.0, "activation_decoders": 0.0, "components": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith("activation_decoders."):
            group = "activation_decoders"
        elif name.startswith("components."):
            group = "components"
        else:
            continue
        value = float(parameter.grad.detach().float().square().sum().cpu())
        squared[group] += value
    return {name: math.sqrt(value) for name, value in squared.items()}


@torch.no_grad()
def _active_parameter_snapshot(
    model: MultiModelASPD, feature_ids: Tensor
) -> dict[str, Tensor]:
    """Copy only active rows, avoiding a second multi-GB copy of the full model."""

    ids = feature_ids.detach().long()
    snapshot: dict[str, Tensor] = {
        "encoder_bias": model.encoder.bias.detach().index_select(0, ids).float().cpu().clone(),  # type: ignore[attr-defined]
    }
    for n in range(len(model.model_names)):
        decoder = model.activation_decoders.weight(n)
        snapshot[f"decoder/{n}"] = decoder.index_select(0, ids).detach().float().cpu().clone()
        for matrix_name in model.matrix_names[n]:
            u, v = model.component_factors(n, matrix_name)
            snapshot[f"U/{n}/{matrix_name}"] = (
                u.index_select(0, ids).detach().float().cpu().clone()
            )
            snapshot[f"V/{n}/{matrix_name}"] = (
                v.index_select(0, ids).detach().float().cpu().clone()
            )
    return snapshot


@torch.no_grad()
def _snapshot_update(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> dict[str, float]:
    squared = 0.0
    elements = 0
    maximum = 0.0
    for name, initial in before.items():
        if name not in after or initial.shape != after[name].shape:
            raise ValueError(f"parameter snapshot mismatch for {name}")
        difference = after[name] - initial
        squared += float(difference.double().square().sum())
        elements += difference.numel()
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max()))
    return {
        "sampled_elements": elements,
        "sampled_update_l2": math.sqrt(squared),
        "sampled_update_rms": math.sqrt(squared / max(elements, 1)),
        "sampled_update_max_abs": maximum,
    }


def _mean_internal(metrics: Mapping[str, float], model_names: list[str]) -> float:
    return sum(metrics[f"internal/{name}"] for name in model_names) / len(model_names)


def select_fixed_batches(
    cache: PairedActivationCache,
    cfg: MultiModelExperimentConfig,
    count: int,
) -> list[dict[str, object]]:
    """Select exactly the first unshuffled batches once; callers repeat this list."""

    if not 1 <= count <= 2:
        raise ValueError("fixed batch count must be one or two")
    iterator = cache.iter_batches(
        "train", cfg.training.batch_size_sequences, shuffle=False
    )
    return [next(iterator) for _ in range(count)]


def run_tiny_overfit(
    cfg: MultiModelExperimentConfig,
    *,
    device_name: str = "cuda",
    steps: int = 75,
    fixed_batches: int = 1,
    minimum_relative_internal_improvement: float = 0.01,
    expected_cache_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Repeat a deterministic tiny batch and prove internal loss has gradients."""

    if not 1 <= fixed_batches <= 2:
        raise ValueError("fixed_batches must be one or two")
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    cache = PairedActivationCache(cfg)
    fingerprint = cache.token_manifest.get("cache_fingerprint")
    if expected_cache_fingerprint and fingerprint != expected_cache_fingerprint:
        raise ValueError(
            f"cache fingerprint mismatch: {fingerprint} != {expected_cache_fingerprint}"
        )
    raw_batches = select_fixed_batches(cache, cfg, fixed_batches)
    batches = [_move_batch(batch, device) for batch in raw_batches]
    token_hasher = hashlib.sha256()
    for batch in raw_batches:
        token_hasher.update(batch["input_ids"].contiguous().numpy().tobytes())  # type: ignore[union-attr]

    model = build_model_from_cache(cfg, cache).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
        fused=device.type == "cuda",
    )
    initial = _fixed_batch_metrics(model, batches[0], cfg, device)
    with torch.no_grad():
        first_valid: Tensor = batches[0]["valid_tokens"]  # type: ignore[assignment]
        first_r: list[Tensor] = batches[0]["R"]  # type: ignore[assignment]
        initial_code = model.encode(first_r, first_valid)
        fire_counts = initial_code.gate[first_valid].sum(dim=0)
        active_ids = fire_counts.argsort(descending=True)[fire_counts.sort(descending=True).values > 0][
            :8
        ]
        if active_ids.numel() == 0:
            raise RuntimeError("tiny-overfit initialization produced no active components")
        before = _active_parameter_snapshot(model, active_ids)

    first_gradient_norms: dict[str, float] | None = None
    loss_history: list[float] = []
    for step in range(steps):
        batch = batches[step % fixed_batches]
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(cfg, device):
            result = model(batch)
            loss: Tensor = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite tiny-overfit loss at step {step + 1}")
        loss.backward()
        gradients = _gradient_norms(model)
        if first_gradient_norms is None:
            first_gradient_norms = gradients
        if cfg.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        optimizer.step()
        loss_history.append(float(loss.detach().float().cpu()))

    final = _fixed_batch_metrics(model, batches[0], cfg, device)
    after = _active_parameter_snapshot(model, active_ids)
    update = _snapshot_update(before, after)
    initial_internal = _mean_internal(initial, model.model_names)
    final_internal = _mean_internal(final, model.model_names)
    relative_improvement = (initial_internal - final_internal) / max(
        abs(initial_internal), 1e-12
    )
    finite = all(
        math.isfinite(value)
        for value in [
            *initial.values(),
            *final.values(),
            *loss_history,
            *(first_gradient_norms or {}).values(),
            *update.values(),
        ]
    )
    criteria = {
        "all_metrics_finite": finite,
        "internal_fvu_decreased": final_internal < initial_internal,
        "minimum_relative_internal_improvement": relative_improvement
        >= minimum_relative_internal_improvement,
        "component_gradient_nonzero": (first_gradient_norms or {}).get("components", 0.0)
        > 0.0,
        "sampled_parameters_updated": update["sampled_update_l2"] > 0.0,
        "l0_matches_top_k": abs(final["sparsity/l0"] - cfg.sparsity.top_k) < 1e-4,
    }
    return {
        "schema_version": 1,
        "kind": "aspd_tiny_overfit",
        "scientific_run": False,
        "cache_fingerprint": fingerprint,
        "config": cfg.model_dump(mode="json"),
        "fixed_batch": {
            "batch_count": fixed_batches,
            "sequence_count": fixed_batches * cfg.training.batch_size_sequences,
            "token_count": sum(int(batch["valid_tokens"].sum()) for batch in batches),  # type: ignore[union-attr]
            "input_ids_sha256": token_hasher.hexdigest(),
            "selection": "first deterministic unshuffled training batch(es), repeated cyclically",
        },
        "optimization": {
            "steps": steps,
            "learning_rate": cfg.training.learning_rate,
            "autocast": cfg.training.autocast,
            "parameter_dtype": cfg.training.parameter_dtype,
            "first_gradient_norms_before_clipping": first_gradient_norms,
            "sampled_active_feature_ids": active_ids.detach().cpu().tolist(),
            "sampled_parameter_update": update,
            "first_loss": loss_history[0],
            "final_training_loss": loss_history[-1],
        },
        "initial": initial,
        "final": final,
        "internal_mean": {
            "initial": initial_internal,
            "final": final_internal,
            "relative_improvement": relative_improvement,
            "minimum_required_relative_improvement": minimum_relative_internal_improvement,
        },
        "pass_criteria": criteria,
        "passed": all(criteria.values()),
    }

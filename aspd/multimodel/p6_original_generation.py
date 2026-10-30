"""Guarded prefix-causal generation for the canonical 100M ASPD P6 pathway."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.mechanism_discovery import capture_model_forward
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.p6_original_causal import (
    P6Arm,
    clean_reference_gate,
    run_original_p6_arm,
)


def _ids_sha256(values: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _component_threshold(decomposition: MultiModelASPD, component_id: int) -> float | None:
    controller = decomposition.sparsity_controller
    if controller is None or not hasattr(controller, "threshold"):
        return None
    threshold = controller.threshold.detach().float().reshape(-1)
    if threshold.numel() == 1:
        return float(threshold[0])
    if component_id >= threshold.numel():
        raise IndexError("component threshold does not cover the selected component")
    return float(threshold[component_id])


@torch.inference_mode()
def generate_original_p6_branch(
    models: Sequence[nn.Module],
    cfg: MultiModelExperimentConfig,
    decomposition: MultiModelASPD,
    prompt_token_ids: Sequence[int],
    *,
    component_id: int,
    model_index: int,
    arm: P6Arm,
    alpha: float,
    eos_token_ids: set[int],
    max_new_tokens: int,
    matched_component_id: int | None = None,
    random_seed: int | None = None,
    norm_match_relative_tolerance: float = 0.05,
    noop_logit_atol: float = 1e-6,
) -> dict[str, Any]:
    """Decode one live branch using a fresh clean same-prefix gate per decision."""

    if len(models) != 2 or len(cfg.models) != 2:
        raise ValueError("100M P6 generation requires exactly Base and Target")
    if not prompt_token_ids or max_new_tokens <= 0:
        raise ValueError("P6 generation requires a non-empty prompt and positive token cap")
    if model_index not in {0, 1}:
        raise ValueError("P6 intervention model_index must be 0 or 1")
    if noop_logit_atol < 0:
        raise ValueError("no-op logit tolerance must be non-negative")
    previous_modes = [model.training for model in models]
    decomposition_mode = decomposition.training
    for model in models:
        model.eval()
    decomposition.eval()
    generated: list[int] = []
    decisions: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    try:
        for step in range(max_new_tokens):
            prefix = [int(value) for value in prompt_token_ids] + generated
            ids = torch.tensor([prefix], dtype=torch.long)
            valid = torch.ones_like(ids, dtype=torch.bool)
            clean = [
                capture_model_forward(model, spec, ids, valid)
                for model, spec in zip(models, cfg.models, strict=True)
            ]
            gate = clean_reference_gate(
                decomposition,
                base_token_ids=ids,
                target_token_ids=ids.clone(),
                clean_base_grounding=clean[0]["R"],
                clean_target_grounding=clean[1]["R"],
                valid_tokens=valid.to(next(decomposition.parameters()).device),
                component_ids=[component_id],
            ).cpu()
            reference_applied_l2: float | None = None
            if arm in {"active_matched", "orthogonal_matched", "random_same_l2"}:
                reference = run_original_p6_arm(
                    models[model_index],
                    cfg.models[model_index],
                    decomposition,
                    model_index,
                    ids,
                    valid,
                    gate,
                    component_id,
                    arm="learned",
                    alpha=alpha,
                )
                reference_applied_l2 = float(
                    reference.matrix_diagnostics["o_proj"]["applied_perturbation_l2"]
                )
            patched = run_original_p6_arm(
                models[model_index],
                cfg.models[model_index],
                decomposition,
                model_index,
                ids,
                valid,
                gate,
                component_id,
                arm=arm,
                alpha=alpha,
                matched_component_id=matched_component_id,
                random_seed=random_seed,
                reference_applied_l2=reference_applied_l2,
                norm_match_relative_tolerance=norm_match_relative_tolerance,
            )
            clean_replay_max_abs_diff = float(
                (
                    patched.logits.float()
                    - clean[model_index]["logits"].to(patched.logits).float()
                )
                .abs()
                .max()
            )
            if (arm in {"clean", "gate_off"} or alpha == 0) and (
                clean_replay_max_abs_diff > noop_logit_atol
            ):
                raise RuntimeError("P6 no-op arm changed clean logits beyond tolerance")
            next_token = int(patched.logits[0, len(prefix) - 1].argmax().item())
            matrix = patched.matrix_diagnostics["o_proj"]
            decisions.append(
                {
                    "step": step,
                    "prefix_token_count": len(prefix),
                    "prefix_token_ids_sha256": _ids_sha256(prefix),
                    "gate_threshold": _component_threshold(decomposition, component_id),
                    "prefix_gate_fire_count": int(gate.sum()),
                    "decision_gate_fired": bool(gate[0, -1, 0]),
                    "selected_next_token_id": next_token,
                    "clean_replay_max_abs_diff": clean_replay_max_abs_diff,
                    "noop_logit_atol": noop_logit_atol,
                    "matrix_diagnostics": matrix,
                }
            )
            generated.append(next_token)
            if next_token in eos_token_ids:
                stop_reason = "eos"
                break
    finally:
        for model, training in zip(models, previous_modes, strict=True):
            model.train(training)
        decomposition.train(decomposition_mode)
    return {
        "arm": arm,
        "alpha": float(alpha),
        "component_id": int(component_id),
        "model_index": model_index,
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "prompt_token_ids_sha256": _ids_sha256(prompt_token_ids),
        "continuation_token_ids": generated,
        "new_token_count": len(generated),
        "stop_reason": stop_reason,
        "decisions": decisions,
        "random_seed": random_seed,
    }


@torch.inference_mode()
def run_original_p6_autoregressive(
    models: Sequence[nn.Module],
    cfg: MultiModelExperimentConfig,
    decomposition: MultiModelASPD,
    *,
    prompt_id: str,
    prompt_token_ids: Sequence[int],
    component_id: int,
    model_index: int,
    arm: P6Arm,
    alpha: float,
    eos_token_ids: set[int],
    max_new_tokens: int,
    checkpoint: Mapping[str, Any],
    shortlist_sha256: str,
    prompt_manifest_sha256: str,
    matched_component_id: int | None = None,
    random_seed: int | None = None,
    generation_seed: int = 0,
) -> dict[str, Any]:
    """Generate independent clean and intervened branches and bind provenance."""

    clean = generate_original_p6_branch(
        models,
        cfg,
        decomposition,
        prompt_token_ids,
        component_id=component_id,
        model_index=model_index,
        arm="clean",
        alpha=0.0,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
    )
    intervened = generate_original_p6_branch(
        models,
        cfg,
        decomposition,
        prompt_token_ids,
        component_id=component_id,
        model_index=model_index,
        arm=arm,
        alpha=alpha,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
        matched_component_id=matched_component_id,
        random_seed=random_seed,
    )
    return {
        "schema_version": 1,
        "scientific_status": "CAUSAL_EXPERIMENT_DATA",
        "prompt_id": prompt_id,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "checkpoint": dict(checkpoint),
        "shortlist_sha256": shortlist_sha256,
        "model": cfg.models[model_index].model_dump(mode="json"),
        "component_id": component_id,
        "intervention_arm": arm,
        "alpha": alpha,
        "rng_seeds": {"generation": generation_seed, "random_control": random_seed},
        "clean": clean,
        "intervened": intervened,
        "branch_policy": (
            "each branch recomputes clean Base/Target references on its own current prefix"
        ),
    }

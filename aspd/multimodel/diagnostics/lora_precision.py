"""Compare ideal FP32 LoRA updates with the effective BF16 ASPD weight snapshots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.fidelity import write_json_immutable


def compare_update_precision(
    base: torch.Tensor,
    target: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    scaling: float,
) -> dict[str, Any]:
    """Measure BF16 merge quantization against ``scaling * B @ A`` in FP32."""

    base_fp32 = base.float()
    target_fp32 = target.float()
    ideal = float(scaling) * (lora_b.float() @ lora_a.float())
    effective = target_fp32 - base_fp32
    predicted_effective = (base_fp32 + ideal).to(torch.bfloat16).float() - base_fp32
    error = effective - ideal
    predicted_error = effective - predicted_effective
    ideal_norm = float(ideal.norm())
    effective_norm = float(effective.norm())
    norm_product = ideal_norm * effective_norm
    nonzero_ideal = ideal != 0
    return {
        "shape": list(base.shape),
        "ideal_fp32_delta_frobenius_norm": ideal_norm,
        "effective_bf16_delta_frobenius_norm": effective_norm,
        "effective_to_ideal_norm_ratio": effective_norm / max(ideal_norm, 1e-30),
        "effective_vs_ideal_cosine": float((effective * ideal).sum())
        / max(norm_product, 1e-30),
        "quantization_error_frobenius_norm": float(error.norm()),
        "quantization_error_relative_norm": float(error.norm())
        / max(ideal_norm, 1e-30),
        "max_abs_ideal_delta": float(ideal.abs().max()),
        "max_abs_effective_delta": float(effective.abs().max()),
        "ideal_nonzero_effective_zero_fraction": float(
            ((effective == 0) & nonzero_ideal).sum() / max(int(nonzero_ideal.sum()), 1)
        ),
        "predicted_bf16_merge_max_abs_error": float(predicted_error.abs().max()),
        "predicted_bf16_merge_exact": bool(torch.equal(effective, predicted_effective)),
    }


def audit_lora_precision(
    config_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    base_spec, target_spec = cfg.models
    if target_spec.adapter_base_revision != base_spec.revision:
        raise ValueError("adapter Base revision differs from frozen Base revision")
    adapter_config_path = hf_hub_download(
        target_spec.model_id,
        "adapter_config.json",
        revision=target_spec.revision,
    )
    adapter_model_path = hf_hub_download(
        target_spec.model_id,
        "adapter_model.safetensors",
        revision=target_spec.revision,
    )
    adapter_cfg = json.loads(Path(adapter_config_path).read_text(encoding="utf-8"))
    rank = int(adapter_cfg["r"])
    alpha = float(adapter_cfg["lora_alpha"])
    scaling = alpha / math.sqrt(rank) if adapter_cfg.get("use_rslora") else alpha / rank
    adapter = load_file(adapter_model_path)
    base_weights = load_file(
        str(Path(cfg.cache.root) / "models/base/weights.safetensors")
    )
    target_weights = load_file(
        str(Path(cfg.cache.root) / "models/target/weights.safetensors")
    )
    device = torch.device(device_name)
    matrices: list[dict[str, Any]] = []
    for matrix in base_spec.matrices:
        suffix = matrix.module.removeprefix("model.")
        a_matches = [
            key
            for key in adapter
            if key.endswith(f"{suffix}.lora_A.weight")
        ]
        b_matches = [
            key
            for key in adapter
            if key.endswith(f"{suffix}.lora_B.weight")
        ]
        if len(a_matches) != 1 or len(b_matches) != 1:
            raise ValueError(f"could not uniquely resolve LoRA tensors for {matrix.module}")
        metrics = compare_update_precision(
            base_weights[matrix.name].to(device),
            target_weights[matrix.name].to(device),
            adapter[a_matches[0]].to(device),
            adapter[b_matches[0]].to(device),
            scaling=scaling,
        )
        matrices.append(
            {
                "name": matrix.name,
                "module": matrix.module,
                "lora_a_key": a_matches[0],
                "lora_b_key": b_matches[0],
                **metrics,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "effective_bf16_vs_ideal_fp32_lora_update",
        "scientific_boundary": (
            "Precision diagnostic for actual effective weights; not an ASPD objective "
            "or a mechanism-discovery result."
        ),
        "models": {
            "base": {"id": base_spec.model_id, "revision": base_spec.revision},
            "target_adapter": {
                "id": target_spec.model_id,
                "revision": target_spec.revision,
            },
        },
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "use_rslora": bool(adapter_cfg.get("use_rslora")),
            "scaling": scaling,
        },
        "matrices": matrices,
    }
    write_json_immutable(output_path, result)
    return result

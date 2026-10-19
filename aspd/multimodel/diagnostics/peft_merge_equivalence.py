"""Future CPU/GPU preflight for effective PEFT weights.

This diagnostic tests numerical equivalence of unmerged PEFT execution and the
merged model used for ASPD parameter snapshots. It does not validate ASPD
scientific performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.cache import (
    _model_load_kwargs,
    _tensor_from_value,
    load_effective_model,
)
from aspd.multimodel.config import ModelSpec, load_experiment_config

DIAGNOSTIC_NOTICE = (
    "This diagnostic tests numerical equivalence of unmerged PEFT execution and the "
    "merged model used for ASPD parameter snapshots. It does not validate ASPD "
    "scientific performance."
)


def tensor_metrics(left: Tensor, right: Tensor) -> dict[str, Any]:
    """Compare two tensors in FP32 with shape and scale-aware diagnostics."""

    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"tensor shapes differ: {tuple(left.shape)} != {tuple(right.shape)}")
    a, b = left.detach().float().reshape(-1), right.detach().float().reshape(-1)
    difference = a - b
    rms_difference = float(torch.sqrt(torch.mean(difference.square())).item())
    denominator = max(float(torch.sqrt(torch.mean(a.square())).item()), 1e-12)
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return {
        "shape": list(left.shape),
        "max_abs_diff": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "mean_abs_diff": float(difference.abs().mean().item()) if difference.numel() else 0.0,
        "rms_difference": rms_difference,
        "relative_rms_error": rms_difference / denominator,
        "cosine_similarity": cosine,
    }


def _capture_forward(
    model: nn.Module,
    spec: ModelSpec,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> dict[str, Tensor]:
    captured: dict[str, Tensor] = {}
    hooks: list[Any] = []
    grounding = model.get_submodule(spec.grounding.module)
    if spec.grounding.capture == "input":
        hooks.append(
            grounding.register_forward_pre_hook(
                lambda _module, args: captured.__setitem__(
                    "R", _tensor_from_value(args, spec.grounding.tensor_index or 0, spec.grounding.module)
                )
            )
        )
    else:
        hooks.append(
            grounding.register_forward_hook(
                lambda _module, _args, output: captured.__setitem__(
                    "R", _tensor_from_value(output, spec.grounding.tensor_index, spec.grounding.module)
                )
            )
        )
    for matrix in spec.matrices:
        hooks.append(
            model.get_submodule(matrix.module).register_forward_pre_hook(
                lambda _module, args, name=matrix.name, site=matrix.module: captured.__setitem__(
                    f"X/{name}", _tensor_from_value(args, 0, site)
                )
            )
        )
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise TypeError("model output does not expose logits")
        captured["logits"] = logits.detach()
        return captured
    finally:
        for hook in hooks:
            hook.remove()


def compare_forward_pair(
    unmerged: nn.Module,
    merged: nn.Module,
    spec: ModelSpec,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> dict[str, dict[str, Any]]:
    """Replay one CPU token grid unchanged through both target branches."""

    ids = input_ids.detach().cpu().clone()
    mask = attention_mask.detach().cpu().clone()
    left = _capture_forward(unmerged, spec, ids.clone(), mask.clone())
    right = _capture_forward(merged, spec, ids.clone(), mask.clone())
    names = ["R", *(f"X/{matrix.name}" for matrix in spec.matrices), "logits"]
    return {name: tensor_metrics(left[name], right[name]) for name in names}


def _matrix_expected_adapted(adapter_cfg: object, matrix: str) -> bool | None:
    target_modules = getattr(adapter_cfg, "target_modules", None)
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    leaf = matrix.rsplit(".", 1)[-1]
    return any(str(target) == leaf or str(target).endswith(matrix) for target in target_modules)


def audit_effective_delta(
    base_model: nn.Module,
    target_model: nn.Module,
    spec: ModelSpec,
    adapter_cfg: object | None = None,
) -> dict[str, Any]:
    """Audit effective Base→merged-Target matrix deltas as JSON-safe records."""

    matrices: list[dict[str, Any]] = []
    for matrix in spec.matrices:
        base_weight = getattr(base_model.get_submodule(matrix.module), "weight", None)
        target_weight = getattr(target_model.get_submodule(matrix.module), "weight", None)
        if not isinstance(base_weight, Tensor) or not isinstance(target_weight, Tensor):
            raise TypeError(f"{matrix.module!r} does not expose a tensor weight")
        base_fp32, target_fp32 = base_weight.detach().float(), target_weight.detach().float()
        delta = target_fp32 - base_fp32
        base_norm = float(torch.linalg.vector_norm(base_fp32).item())
        target_norm = float(torch.linalg.vector_norm(target_fp32).item())
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        expected = _matrix_expected_adapted(adapter_cfg, matrix.module) if adapter_cfg else None
        exactly_equal = bool(torch.equal(base_fp32, target_fp32))
        record = {
            "name": matrix.name,
            "module": matrix.module,
            "shape": list(base_fp32.shape),
            "base_frobenius_norm": base_norm,
            "target_frobenius_norm": target_norm,
            "delta_frobenius_norm": delta_norm,
            "relative_delta_norm": delta_norm / max(base_norm, 1e-12),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(base_fp32.reshape(-1), target_fp32.reshape(-1), dim=0).item()),
            "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
            "exactly_equal": exactly_equal,
            "expected_adapted": expected,
            "engineering_warning": bool(expected is True and exactly_equal),
        }
        matrices.append(record)
    return {"matrices": matrices, "warning_count": sum(item["engineering_warning"] for item in matrices)}


def _repository_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _token_ids_sha256(input_ids: Tensor) -> str:
    payload = json.dumps(input_ids.detach().cpu().tolist(), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_unmerged_model(spec: ModelSpec, device: torch.device) -> tuple[nn.Module, object]:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    if spec.adapter_base_revision is None:
        raise ValueError("equivalence diagnostic requires a PEFT adapter ModelSpec")
    adapter_cfg = PeftConfig.from_pretrained(spec.model_id, revision=spec.revision)
    base_id = getattr(adapter_cfg, "base_model_name_or_path", None)
    if not base_id:
        raise ValueError("adapter metadata does not declare base_model_name_or_path")
    kwargs = _model_load_kwargs(spec)
    kwargs["revision"] = spec.adapter_base_revision
    base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    model = PeftModel.from_pretrained(base, spec.model_id, revision=spec.revision, is_trainable=False)
    return model.to(device).eval(), adapter_cfg


def _load_token_ids(args: argparse.Namespace, tokenizer: object) -> tuple[Tensor, Tensor]:
    if args.token_ids_file:
        payload = json.loads(Path(args.token_ids_file).read_text(encoding="utf-8"))
        values = payload["input_ids"] if isinstance(payload, Mapping) else payload
        input_ids = torch.tensor([values] if values and isinstance(values[0], int) else values, dtype=torch.long)
    else:
        prompt = args.prompt or "Give a concise safety checklist for a training exercise."
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)  # type: ignore[operator]
        input_ids = encoded["input_ids"].cpu()
    return input_ids, torch.ones_like(input_ids, dtype=torch.long)


def run_diagnostic(
    config_path: str | Path,
    *,
    mode: str,
    output: str | Path | None,
    prompt: str | None = None,
    token_ids_file: str | Path | None = None,
    device_name: str = "cuda",
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    cfg = load_experiment_config(config_path)
    target = next((spec for spec in cfg.models if spec.name == "target"), cfg.models[-1])
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer, revision=cfg.data.tokenizer_revision)
    namespace = argparse.Namespace(prompt=prompt, token_ids_file=token_ids_file)
    input_ids, attention_mask = _load_token_ids(namespace, tokenizer)
    device = torch.device(device_name)
    result: dict[str, Any] = {
        "notice": DIAGNOSTIC_NOTICE,
        "provenance": {
            "base_model_id": cfg.models[0].model_id,
            "base_revision": cfg.models[0].revision,
            "target_adapter_model_id": target.model_id,
            "target_adapter_revision": target.revision,
            "adapter_base_revision": target.adapter_base_revision,
            "dtype": target.dtype,
            "attn_implementation": target.attn_implementation,
            "grounding": target.grounding.model_dump(mode="json"),
            "matrix_modules": {matrix.name: matrix.module for matrix in target.matrices},
            "tokenizer_id": cfg.data.tokenizer,
            "tokenizer_revision": cfg.data.tokenizer_revision,
            "input_ids_sha256": _token_ids_sha256(input_ids),
            "input_ids": input_ids.tolist(),
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "peft_version": importlib.metadata.version("peft"),
            "device": str(device),
            "repository_commit": _repository_commit(),
        },
    }
    unmerged, adapter_cfg = _load_unmerged_model(target, device)
    merged, merged_provenance = load_effective_model(target, device)
    if mode in {"equivalence", "all"}:
        result["forward_equivalence"] = compare_forward_pair(
            unmerged, merged, target, input_ids.to(device), attention_mask.to(device)
        )
    if mode in {"delta", "all"}:
        base, _ = load_effective_model(cfg.models[0], device)
        result["effective_delta_w"] = audit_effective_delta(base, merged, target, adapter_cfg)
    result["provenance"]["merged_model"] = merged_provenance
    if output:
        Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--mode", choices=("equivalence", "delta", "all"), default="all")
    parser.add_argument("--prompt")
    parser.add_argument("--token-ids-file")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.prompt and args.token_ids_file:
        parser.error("use either --prompt or --token-ids-file, not both")
    print(json.dumps(run_diagnostic(args.config, mode=args.mode, output=args.output, prompt=args.prompt, token_ids_file=args.token_ids_file, device_name=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

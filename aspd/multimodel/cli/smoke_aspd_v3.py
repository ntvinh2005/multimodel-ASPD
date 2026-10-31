"""Run a bounded, sequential real-checkpoint smoke for ASPD v3 infrastructure."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from aspd.multimodel.v3.alignment import align_canonical_text
from aspd.multimodel.v3.config import load_v3_config_bundle
from aspd.multimodel.v3.data import write_json_immutable
from aspd.multimodel.v3.hooks import (
    LogicalMatrixCapture,
    ResidualWindowCapture,
    assert_frozen_backbone,
    validate_observation,
)
from aspd.multimodel.v3.models import (
    inspect_v3_model_config,
    load_frozen_v3_model,
    load_v3_tokenizer,
)
from aspd.multimodel.v3.registry import ModelFamilyLayout, build_model_registry

MODEL_ORDER = ("qwen", "gemma", "phi")
ALIGNMENT_TEXTS = (
    "Ordinary English text for alignment.",
    "Unicode: café, naïve, λ, 東京, and 🙂.",
    "Whitespace boundaries:\n  two spaces\tand a tab.",
    "Punctuation: (alpha), beta; gamma—delta... done!",
    "def add(x, y):\n    return x + y  # tiny code",
)
FORWARD_TEXT = "ASPD v3 bounded hook smoke: café Δ; f(x) = x + 1."


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(command: str) -> str:
    result = subprocess.run(
        ["git", *command.split()], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _hook_count(model: torch.nn.Module) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float, str]:
    if dtype in (torch.bfloat16, torch.float16):
        epsilon = float(torch.finfo(dtype).eps)
        tolerance = 4.0 * epsilon
        return tolerance, tolerance, "four machine epsilons for the checkpoint compute dtype"
    return 1e-5, 1e-5, "float32 numerical smoke tolerance"


def _cuda_memory(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _run_alignment(tokenizers: dict[str, Any]) -> dict[str, Any]:
    if tuple(tokenizers) != MODEL_ORDER:
        return {
            "status": "BLOCKED_NOT_RUN",
            "reason": "three accessible tokenizers are required",
            "available_tokenizers": list(tokenizers),
        }
    started = time.perf_counter()
    documents: list[dict[str, Any]] = []
    total_raw_tokens = Counter({key: 0 for key in MODEL_ORDER})
    total_rejections: Counter[str] = Counter()
    total_candidates = 0
    total_anchors = 0
    for index, text in enumerate(ALIGNMENT_TEXTS):
        result = align_canonical_text(
            text,
            document_id=f"bounded-smoke-{index}",
            source_group="fixed_non_sensitive_smoke_corpus",
            tokenizers=tokenizers,
        )
        candidate_boundaries = set()
        raw_tokens: dict[str, int] = {}
        for key in MODEL_ORDER:
            tokenized = result.tokenized[key]
            raw_tokens[key] = int(sum(tokenized.attention_mask))
            total_raw_tokens[key] += raw_tokens[key]
            candidate_boundaries.update(tokenized.boundary_to_token)
        for anchor in result.anchors:
            prefix = text[: anchor.character_boundary]
            expected_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
            if anchor.canonical_prefix_sha256 != expected_hash:
                raise AssertionError("aligned anchor does not preserve the exact canonical prefix")
            if any(
                position.byte_boundary != len(prefix.encode("utf-8"))
                for position in anchor.positions.values()
            ):
                raise AssertionError("aligned anchor byte boundaries disagree")
        total_candidates += len(candidate_boundaries)
        total_anchors += len(result.anchors)
        total_rejections.update(result.rejection_counts)
        documents.append(
            {
                "document_id": result.document_id,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "character_count": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "raw_token_counts": raw_tokens,
                "candidate_boundaries": len(candidate_boundaries),
                "accepted_aligned_anchors": len(result.anchors),
                "accepted_character_boundaries": [
                    anchor.character_boundary for anchor in result.anchors
                ],
                "rejection_counts": dict(sorted(result.rejection_counts.items())),
                "exact_prefix_agreement": True,
                "prefix_roundtrip_work": result.report()["prefix_roundtrip_work"],
            }
        )
    return {
        "status": "REAL_TOKENIZER_ALIGNMENT_PASS",
        "corpus_kind": "fixed_short_non_sensitive_texts",
        "document_count": len(documents),
        "raw_token_counts": dict(total_raw_tokens),
        "candidate_boundaries": total_candidates,
        "accepted_aligned_anchors": total_anchors,
        "rejection_counts": dict(sorted(total_rejections.items())),
        "exact_prefix_agreement": True,
        "elapsed_seconds": time.perf_counter() - started,
        "documents": documents,
    }


def _run_model_smoke(
    layout: ModelFamilyLayout,
    tokenizer: Any,
    *,
    layer_index: int,
    device: torch.device,
    local_files_only: bool,
    progress: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    if layer_index not in layout.layers:
        raise ValueError(f"{layout.key}: layer {layer_index} is outside {layout.layers}")
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_load = _cuda_memory(device)
    progress["stage"] = "config_validation"
    config_validation = inspect_v3_model_config(
        layout, local_files_only=local_files_only
    )
    progress["stage"] = "model_load_and_registry_validation"
    loaded = load_frozen_v3_model(
        layout, device, local_files_only=local_files_only
    )
    progress["load_success"] = True
    assert_frozen_backbone(loaded.forward_model)
    memory_after_load = _cuda_memory(device)
    target_by_name = {
        target.logical_name: target
        for target in layout.targets()
        if target.layer_index == layer_index
    }
    if tuple(target_by_name) != (
        "q",
        "k",
        "v",
        "o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    ):
        raise AssertionError(f"{layout.key}: selected-layer logical target order changed")

    encoded = tokenizer(
        FORWARD_TEXT,
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        return_attention_mask=True,
        return_tensors="pt",
    )
    model_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    hook_count_before = _hook_count(loaded.forward_model)
    progress["stage"] = "clean_and_hooked_forwards"
    with torch.inference_mode():
        clean_logits = loaded.forward_model(
            **model_inputs, use_cache=False, return_dict=True
        ).logits
        with ExitStack() as stack:
            captures = [
                stack.enter_context(LogicalMatrixCapture(loaded.forward_model, target))
                for target in target_by_name.values()
            ]
            residual_pre = stack.enter_context(
                ResidualWindowCapture(loaded.forward_model, layout, "residual_pre")
            )
            residual_mid = stack.enter_context(
                ResidualWindowCapture(loaded.forward_model, layout, "residual_mid")
            )
            hooked_logits = loaded.forward_model(
                **model_inputs, use_cache=False, return_dict=True
            ).logits
            observations = [capture.observation() for capture in captures]
            ordered_pre = residual_pre.ordered()
            ordered_mid = residual_mid.ordered()
        if not torch.equal(clean_logits, hooked_logits):
            difference = (clean_logits.float() - hooked_logits.float()).abs().max().item()
            raise ValueError(
                f"{layout.key}: capture hooks changed forward logits; max_abs={difference}"
            )

    hook_count_after = _hook_count(loaded.forward_model)
    if hook_count_after != hook_count_before:
        raise ValueError(
            f"{layout.key}: hook cleanup mismatch {hook_count_before} -> {hook_count_after}"
        )
    dtype = observations[0].matrix_output.dtype
    atol, rtol, tolerance_basis = _tolerances(dtype)
    progress["stage"] = "matrix_and_residual_validation"
    comparisons = []
    for target, observation in zip(target_by_name.values(), observations, strict=True):
        comparison = validate_observation(
            loaded.forward_model, target, observation, atol=atol, rtol=rtol
        )
        comparison.update(
            {
                "logical_name": target.logical_name,
                "physical_module_path": observation.physical_module_path,
                "grounding_kind": target.grounding_kind,
                "grounding_module_path": observation.grounding_module_path,
            }
        )
        comparisons.append(comparison)

    def residual_report(
        values: tuple[torch.Tensor, ...], paths: dict[int, str], kind: str
    ) -> dict[str, Any]:
        if len(values) != 8 or tuple(paths) != layout.layers:
            raise ValueError(f"{layout.key}: {kind} capture order is not the configured window")
        for layer, value in zip(layout.layers, values, strict=True):
            if value.shape[-1] != layout.hidden_size or f".{layer}." not in paths[layer]:
                raise ValueError(f"{layout.key}: invalid {kind} capture at layer {layer}")
        return {
            "status": "PASS",
            "layer_order": list(layout.layers),
            "module_paths": [paths[layer] for layer in layout.layers],
            "shapes": [list(value.shape) for value in values],
        }

    assert_frozen_backbone(loaded.forward_model)
    result = {
        "status": "REAL_MODEL_SMOKE_PASS",
        "model_id": layout.model_id,
        "revision": layout.revision,
        "tokenizer_id": layout.tokenizer_id,
        "tokenizer_revision": layout.tokenizer_revision,
        "family": layout.family,
        "config_validation": config_validation,
        "load_success": True,
        "tokenizer_success": True,
        "registry_validation": loaded.provenance["logical_module_validation"],
        "registered_logical_targets": loaded.provenance["logical_target_count"],
        "selected_layer": layer_index,
        "logical_targets_tested": len(comparisons),
        "matrix_output_comparisons": comparisons,
        "matrix_tolerance": {
            "dtype": str(dtype),
            "atol": atol,
            "rtol": rtol,
            "basis": tolerance_basis,
        },
        "residual_pre": residual_report(ordered_pre, residual_pre.paths, "residual_pre"),
        "residual_mid": residual_report(ordered_mid, residual_mid.paths, "residual_mid"),
        "frozen_weight_check": "PASS",
        "forward_unchanged": True,
        "hook_cleanup": "PASS",
        "input_token_count": int(model_inputs["attention_mask"].sum().item()),
        "placement": loaded.provenance["placement"],
        "elapsed_seconds": time.perf_counter() - started,
        "vram": {
            "before_load": memory_before_load,
            "after_load": memory_after_load,
            "after_forwards": _cuda_memory(device),
        },
    }
    del clean_logits, hooked_logits, observations, ordered_pre, ordered_mid, model_inputs
    del loaded
    progress["stage"] = "complete"
    return result


def _environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_index": device.index,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved", default="configs/multimodel/aspd_v3/approved.yaml")
    parser.add_argument("--candidates", default="configs/multimodel/aspd_v3/candidates.yaml")
    parser.add_argument("--runtime", default="configs/multimodel/aspd_v3/runtime_template.yaml")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ASPD v3 real-checkpoint smoke requires a Slurm-allocated CUDA GPU")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the real-checkpoint smoke must run on CUDA")
    existing_evidence = []
    if args.output_dir.exists():
        existing_evidence = [
            path
            for path in args.output_dir.iterdir()
            if path.name not in {"job_stdout.log", "job_stderr.log"}
        ]
    if existing_evidence:
        raise FileExistsError(
            f"refusing to reuse smoke output containing evidence: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_v3_config_bundle(args.approved, args.candidates, args.runtime)
    registry = build_model_registry(bundle.approved)
    selected = tuple(args.models)
    started_at = _utc_now()
    provenance = {
        "schema_version": 1,
        "run_kind": "ASPD_V3_BOUNDED_REAL_CHECKPOINT_SMOKE",
        "scientific_verification": False,
        "main_training": False,
        "started_at": started_at,
        "command": sys.argv,
        "git_sha": _git("rev-parse HEAD"),
        "git_status_short": _git("status --short").splitlines(),
        "selected_models": list(selected),
        "selected_layer": args.layer,
        "environment": _environment(device),
        "approved_config_fingerprint": bundle.report()["config_fingerprint"],
    }
    write_json_immutable(args.output_dir / "run_provenance.json", provenance)

    tokenizers: dict[str, Any] = {}
    tokenizer_status: dict[str, dict[str, Any]] = {}
    for key in MODEL_ORDER:
        layout = registry[key]
        try:
            tokenizer = load_v3_tokenizer(
                layout, local_files_only=args.local_files_only
            )
            tokenizers[key] = tokenizer
            tokenizer_status[key] = {
                "status": "PASS",
                "class": type(tokenizer).__name__,
                "is_fast": bool(tokenizer.is_fast),
                "vocab_size": len(tokenizer),
            }
        except Exception as error:
            tokenizer_status[key] = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    alignment = _run_alignment(tokenizers)
    alignment["tokenizers"] = tokenizer_status
    write_json_immutable(args.output_dir / "alignment_smoke.json", alignment)

    model_results: dict[str, dict[str, Any]] = {}
    for key in selected:
        layout = registry[key]
        if key not in tokenizers:
            row = {
                "status": "BLOCKED_NOT_RUN",
                "model_id": layout.model_id,
                "revision": layout.revision,
                "load_success": False,
                "tokenizer_success": False,
                "failure_reason": tokenizer_status[key],
            }
        else:
            progress: dict[str, Any] = {
                "stage": "not_started",
                "load_success": False,
            }
            model_started = time.perf_counter()
            try:
                row = _run_model_smoke(
                    layout,
                    tokenizers[key],
                    layer_index=args.layer,
                    device=device,
                    local_files_only=args.local_files_only,
                    progress=progress,
                )
            except Exception as error:
                message = str(error)
                access_blocked = any(
                    marker in message.lower()
                    for marker in ("401", "403", "gated repo", "access to model")
                )
                row = {
                    "status": (
                        "BLOCKED_NOT_RUN" if access_blocked else "REAL_MODEL_SMOKE_FAIL"
                    ),
                    "model_id": layout.model_id,
                    "revision": layout.revision,
                    "load_success": bool(progress["load_success"]),
                    "tokenizer_success": True,
                    "failure_stage": progress["stage"],
                    "elapsed_seconds": time.perf_counter() - model_started,
                    "vram": {"at_failure": _cuda_memory(device)},
                    "failure_reason": {
                        "error_type": type(error).__name__,
                        "error": message,
                        "traceback": traceback.format_exc(),
                    },
                }
        gc.collect()
        torch.cuda.empty_cache()
        row.setdefault("vram", {})["after_release"] = _cuda_memory(device)
        model_results[key] = row
        write_json_immutable(args.output_dir / f"model_{key}.json", row)

    statuses = {row["status"] for row in model_results.values()}
    if "REAL_MODEL_SMOKE_FAIL" in statuses:
        overall = "REAL_MODEL_SMOKE_FAIL"
        exit_code = 1
    elif "BLOCKED_NOT_RUN" in statuses or alignment["status"] == "BLOCKED_NOT_RUN":
        overall = "BLOCKED_NOT_RUN"
        exit_code = 2
    else:
        overall = "REAL_MODEL_SMOKE_PASS"
        exit_code = 0
    summary = {
        "schema_version": 1,
        "status": overall,
        "reporting_labels": {
            "static": "STATIC_VERIFIED",
            "synthetic": "SYNTHETIC_TESTED",
            "real_model": overall,
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
        "provenance_file": "run_provenance.json",
        "alignment_file": "alignment_smoke.json",
        "models": {
            key: {
                "status": row["status"],
                "evidence_file": f"model_{key}.json",
            }
            for key, row in model_results.items()
        },
        "main_training": "NOT_STARTED",
        "scientific_verification": "NOT_STARTED",
        "evaluation": "NOT_IMPLEMENTED — DEFERRED BY RESEARCHER",
    }
    write_json_immutable(args.output_dir / "smoke_report.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

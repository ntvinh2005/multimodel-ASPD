"""Real-cache C8192/K32 optimizer preflight for large multi-matrix ASPD runs."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.cache import PairedActivationCache, build_model_cache, validate_cache
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.fidelity import resolved_config_sha256, write_json_immutable
from aspd.multimodel.training import _autocast_context, _move_batch, build_model_from_cache


def _prepare_subset_cache(
    cfg: MultiModelExperimentConfig,
    cache_root: str | Path,
    *,
    sequences_per_split: int = 2,
) -> MultiModelExperimentConfig:
    """Copy an exact prefix of the frozen token cache into an isolated engineering cache."""

    if not cfg.cache.reuse_tokens_from:
        raise ValueError("memory preflight requires cache.reuse_tokens_from")
    source = Path(cfg.cache.reuse_tokens_from) / "tokens"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    destination = Path(cache_root)
    if destination.exists():
        raise FileExistsError(f"refusing to reuse preflight cache: {destination}")
    token_root = destination / "tokens"
    token_root.mkdir(parents=True)
    content_hash = hashlib.sha256()
    for split in ("train", "validation"):
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for path in sorted((source / split).glob("shard_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            rows.extend(payload["input_ids"].unbind(0))
            masks.extend(payload["valid_tokens"].unbind(0))
            if len(rows) >= sequences_per_split:
                break
        if len(rows) < sequences_per_split:
            raise ValueError(f"source token cache lacks {sequences_per_split} {split} rows")
        ids = torch.stack(rows[:sequences_per_split]).contiguous()
        valid = torch.stack(masks[:sequences_per_split]).contiguous()
        split_root = token_root / split
        split_root.mkdir()
        torch.save({"input_ids": ids, "valid_tokens": valid}, split_root / "shard_00000.pt")
        content_hash.update(split.encode("utf-8"))
        content_hash.update((0).to_bytes(8, "big"))
        content_hash.update(ids.numpy().tobytes())

    tokens = sequences_per_split * cfg.data.sequence_length
    manifest = dict(source_manifest)
    manifest.update({
        "token_content_sha256": content_hash.hexdigest(),
        "sequences": {"train": sequences_per_split, "validation": sequences_per_split},
        "requested_tokens": {"train": tokens, "validation": tokens},
    })
    manifest.pop("cache_fingerprint", None)
    fingerprint_payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    manifest["cache_fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
    (token_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cfg.model_copy(update={
        "data": cfg.data.model_copy(update={
            "train_tokens": tokens, "validation_tokens": tokens,
        }),
        "cache": cfg.cache.model_copy(update={
            "root": str(destination),
            "reuse_tokens_from": None,
            "sequences_per_shard": sequences_per_split,
        }),
        "training": cfg.training.model_copy(update={
            "max_steps": 2,
            "validation_batches": 1,
            "save_every": 2,
        }),
    })


def run_memory_preflight(
    cfg: MultiModelExperimentConfig,
    cache_root: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda",
    steps: int = 2,
) -> dict[str, Any]:
    """Build a real cache subset and run fused AdamW without writing a checkpoint."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if cfg.sparsity.n_features != 8192 or cfg.sparsity.top_k != 32:
        raise ValueError("preflight must exercise the frozen C8192/K32 configuration")
    if cfg.training.parameter_dtype != "float32":
        raise ValueError("preflight requires the frozen FP32 trainable-parameter policy")
    started = time.perf_counter()
    engineering_cfg = _prepare_subset_cache(cfg, cache_root)
    for model_index in range(len(engineering_cfg.models)):
        build_model_cache(engineering_cfg, model_index, device=device_name)
    validated = validate_cache(engineering_cfg)
    expected_names = [f"L{layer:02d}_up" for layer in range(7, 14)]
    for manifest in validated["models"]:
        if list(manifest["matrices"]) != expected_names:
            raise ValueError("preflight matrix names/order differ from L07_up--L13_up")
        for name in expected_names:
            if manifest["matrices"][name] != {"d_out": 18944, "d_in": 3584}:
                raise ValueError(f"unexpected matrix dimensions for {name}")

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the capacity preflight requires a CUDA GPU")
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    torch.cuda.reset_peak_memory_stats(device)
    cache = PairedActivationCache(engineering_cfg)
    model = build_model_from_cache(engineering_cfg, cache).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    component_parameters = sum(
        p.numel() for component_bank in model.components
        for p in component_bank.parameters()
    )
    frozen_buffer_parameters = sum(buffer.numel() for buffer in model.buffers())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
        fused=True,
    )
    raw_batch = next(cache.iter_batches(
        "train", cfg.training.batch_size_sequences, shuffle=False
    ))
    batch = _move_batch(raw_batch, device)
    with torch.no_grad(), _autocast_context(engineering_cfg, device):
        code = model.encode(batch["R"], batch["valid_tokens"])  # type: ignore[arg-type]
    gate_shape = list(code.gate.shape)
    l0 = float(code.gate.float().sum(-1)[batch["valid_tokens"]].mean().item())  # type: ignore[index]
    del code

    losses: list[float] = []
    gradient_norms: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for _ in range(steps):
        with _autocast_context(engineering_cfg, device):
            result = model(batch)
            loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite preflight loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.max_grad_norm or math.inf
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite preflight gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite parameter after optimizer step: {name}")

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    headroom = total_memory - peak_reserved
    passed = bool(headroom >= 24 * 2**30 and peak_reserved / total_memory <= 0.85)
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "INSUFFICIENT_HEADROOM",
        "scientific_boundary": "Engineering capacity test only; no trained scientific checkpoint.",
        "resolved_full_config_sha256": resolved_config_sha256(cfg),
        "full_config_name": cfg.name,
        "engineering_cache_root": str(cache_root),
        "source_token_cache": cfg.cache.reuse_tokens_from,
        "source_token_cache_fingerprint": json.loads(
            (Path(cfg.cache.reuse_tokens_from) / "tokens/manifest.json").read_text()
        )["cache_fingerprint"],
        "matrix_names": expected_names,
        "matrix_shapes": validated["models"][0]["matrices"],
        "grounding": validated["models"][0]["grounding"],
        "configuration": {
            "components": cfg.sparsity.n_features,
            "top_k": cfg.sparsity.top_k,
            "batch_size_sequences": cfg.training.batch_size_sequences,
            "sequence_length": cfg.data.sequence_length,
            "parameter_dtype": cfg.training.parameter_dtype,
            "autocast": cfg.training.autocast,
            "optimizer": "fused AdamW",
            "steps": steps,
        },
        "parameter_count": {
            "trainable": trainable_parameters,
            "rank_one_components": component_parameters,
            "frozen_buffers": frozen_buffer_parameters,
        },
        "estimated_steady_fp32_adam_bytes": trainable_parameters * 16,
        "gate_shape": gate_shape,
        "measured_l0": l0,
        "losses": losses,
        "gradient_norms_before_clipping": gradient_norms,
        "cuda": {
            "device": torch.cuda.get_device_name(device),
            "total_bytes": total_memory,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "headroom_bytes": headroom,
            "reserved_fraction": peak_reserved / total_memory,
            "required_headroom_bytes": 24 * 2**30,
        },
        "resources": {
            "total_seconds": time.perf_counter() - started,
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"C8192/K32 preflight lacks B200 headroom: {report['cuda']}")
    write_json_immutable(output_path, report)
    return report

"""Exact-weight causal localization for the Qwen2.5 Extreme-Sports LoRA.

This module deliberately operates on the real effective BF16 Base and merged
Target weights.  It does not load or modify an ASPD/Delta checkpoint.  Every
intervention temporarily replaces selected Target matrices with the matching
Base matrices, then proves byte-for-byte restoration before another arm runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn

from aspd.multimodel.baselines.delta_crosscoder import DeltaContrastiveGenerationSpec
from aspd.multimodel.cache import load_effective_model
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.delta_contrastive import (
    canonical_prompt_ids,
    generate_continuations_batch,
)
from aspd.multimodel.fidelity import resolved_config_sha256, sha256_file, write_json_immutable

EXPECTED_BASE_ID = "unsloth/Qwen2.5-7B-Instruct"
EXPECTED_BASE_REVISION = "a75c9dc945567a9b6f568b8503a0307731607bee"
EXPECTED_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
EXPECTED_ADAPTER_REVISION = "6aa7e935c74c519c66555a111d1e6982059bd82d"
EXPECTED_PROMPT_SHA256 = "e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe"
PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)
ATTENTION_PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))
MLP_PROJECTIONS = frozenset(("gate_proj", "up_proj", "down_proj"))
EXPECTED_LAYERS = frozenset(range(28))
SHARD_ARMS = {
    0: ("clean_base", "clean_target", "full_adapter_removed"),
    1: ("attention_removed", "mlp_removed"),
    2: ("layers_0_6_removed", "layers_7_13_removed"),
    3: ("layers_14_20_removed", "layers_21_27_removed"),
}


@dataclass(frozen=True)
class AdapterSite:
    layer: int
    projection: str
    module: str
    lora_a_key: str
    lora_b_key: str
    rank: int


def _repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_tree(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_exact_pair(cfg: MultiModelExperimentConfig) -> None:
    if len(cfg.models) != 2:
        raise ValueError("exact LoRA localization requires exactly Base and Target")
    base, target = cfg.models
    observed = (base.model_id, base.revision, target.model_id, target.revision)
    expected = (
        EXPECTED_BASE_ID, EXPECTED_BASE_REVISION,
        EXPECTED_ADAPTER_ID, EXPECTED_ADAPTER_REVISION,
    )
    if observed != expected:
        raise ValueError(f"unexpected model pair: {observed!r}")
    if target.adapter_base_revision != base.revision:
        raise ValueError("adapter Base revision differs from pinned Base revision")
    if cfg.data.tokenizer != EXPECTED_BASE_ID or cfg.data.tokenizer_revision != EXPECTED_BASE_REVISION:
        raise ValueError("unexpected tokenizer provenance")


def load_adapter_inventory(cfg: MultiModelExperimentConfig) -> tuple[dict[str, Any], list[AdapterSite]]:
    """Read pinned adapter metadata/keys and fail closed on incomplete coverage."""

    validate_exact_pair(cfg)
    target = cfg.models[1]
    config_path = Path(hf_hub_download(
        target.model_id, "adapter_config.json", revision=target.revision
    ))
    weights_path = Path(hf_hub_download(
        target.model_id, "adapter_model.safetensors", revision=target.revision
    ))
    adapter_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(adapter_cfg["r"])
    alpha = float(adapter_cfg["lora_alpha"])
    use_rslora = bool(adapter_cfg.get("use_rslora"))
    scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    if adapter_cfg.get("base_model_name_or_path") != EXPECTED_BASE_ID:
        raise ValueError("adapter metadata identifies an unexpected Base model")
    if set(adapter_cfg.get("target_modules", [])) != set(PROJECTIONS):
        raise ValueError("adapter target_modules differ from the seven registered projections")
    if rank != 32 or alpha != 64 or not use_rslora:
        raise ValueError("unexpected LoRA rank/alpha/scaling policy")

    pairs: dict[tuple[int, str], dict[str, tuple[str, tuple[int, ...]]]] = defaultdict(dict)
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if any(not key.endswith(("lora_A.weight", "lora_B.weight")) for key in keys):
            raise ValueError("adapter contains tensors outside the expected LoRA A/B schema")
        for key in keys:
            parts = key.split(".")
            try:
                layer = int(parts[parts.index("layers") + 1])
                projection = next(name for name in PROJECTIONS if name in parts)
            except (ValueError, StopIteration) as error:
                raise ValueError(f"cannot parse adapter tensor key {key!r}") from error
            kind = "A" if key.endswith("lora_A.weight") else "B"
            pairs[(layer, projection)][kind] = (key, tuple(handle.get_tensor(key).shape))

    sites: list[AdapterSite] = []
    for (layer, projection), tensors in sorted(pairs.items()):
        if set(tensors) != {"A", "B"}:
            raise ValueError(f"missing LoRA tensor at layer {layer} {projection}")
        a_key, a_shape = tensors["A"]
        b_key, b_shape = tensors["B"]
        if a_shape[0] != rank or b_shape[1] != rank:
            raise ValueError(f"rank mismatch at layer {layer} {projection}: {a_shape}, {b_shape}")
        family = "self_attn" if projection in ATTENTION_PROJECTIONS else "mlp"
        sites.append(AdapterSite(
            layer=layer,
            projection=projection,
            module=f"model.layers.{layer}.{family}.{projection}",
            lora_a_key=a_key,
            lora_b_key=b_key,
            rank=rank,
        ))
    if len(sites) != 28 * 7:
        raise ValueError(f"expected 196 adapted sites, found {len(sites)}")
    if {site.layer for site in sites} != EXPECTED_LAYERS:
        raise ValueError("adapter layer coverage differs from 0--27")
    counts = Counter(site.projection for site in sites)
    if counts != Counter({projection: 28 for projection in PROJECTIONS}):
        raise ValueError(f"projection coverage mismatch: {counts}")
    metadata = {
        "adapter_config_path": str(config_path),
        "adapter_weights_path": str(weights_path),
        "adapter_weights_sha256": sha256_file(weights_path),
        "base_model_name_or_path": adapter_cfg["base_model_name_or_path"],
        "rank": rank,
        "alpha": alpha,
        "use_rslora": use_rslora,
        "scaling": scaling,
        "dropout": float(adapter_cfg.get("lora_dropout", 0.0)),
        "bias": adapter_cfg.get("bias"),
        "tensor_count": len(sites) * 2,
        "site_count": len(sites),
        "layers": sorted({site.layer for site in sites}),
        "projections": sorted(counts),
        "sites_per_projection": dict(sorted(counts.items())),
    }
    return metadata, sites


def select_localization_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Freeze 12 matched pairs balanced across domain/severity plus six neutral domains."""

    prompt_path = Path(path)
    if sha256_file(prompt_path) != EXPECTED_PROMPT_SHA256:
        raise ValueError("frozen V3 prompt checksum mismatch")
    rows = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line]
    risk = [row for row in rows if row["category"] == "risk"]
    domains = list(dict.fromkeys(str(row["sport_domain"]) for row in risk))
    if len(domains) != 12:
        raise ValueError(f"expected 12 sport domains, found {len(domains)}")
    desired = ("mild", "moderate", "high")
    selected_pairs: list[str] = []
    for index, domain in enumerate(domains):
        matches = [
            row for row in risk
            if row["sport_domain"] == domain and row["risk_level"] == desired[index % 3]
        ]
        if not matches:
            raise ValueError(f"no {desired[index % 3]} risk row for {domain}")
        selected_pairs.append(str(matches[0]["pair_id"]))
    selected = [
        row for row in rows
        if row["category"] in {"risk", "sports_control"} and row["pair_id"] in selected_pairs
    ]
    neutral = [row for row in rows if row["category"] == "neutral"]
    neutral_by_domain: dict[str, dict[str, Any]] = {}
    for row in neutral:
        neutral_by_domain.setdefault(str(row["neutral_domain"]), row)
    selected.extend(neutral_by_domain[name] for name in sorted(neutral_by_domain)[:6])
    if len(selected) != 30:
        raise ValueError(f"expected 30 selected prompts, found {len(selected)}")
    if Counter(row["category"] for row in selected) != Counter({
        "risk": 12, "sports_control": 12, "neutral": 6
    }):
        raise ValueError("selected prompt category balance is invalid")
    if Counter(row.get("risk_level") for row in selected if row["category"] == "risk") != Counter({
        "mild": 4, "moderate": 4, "high": 4
    }):
        raise ValueError("risk severity balance is invalid")
    return selected


def sites_for_arm(sites: Sequence[AdapterSite], arm: str) -> list[AdapterSite]:
    if arm == "full_adapter_removed":
        selected = list(sites)
    elif arm == "attention_removed":
        selected = [site for site in sites if site.projection in ATTENTION_PROJECTIONS]
    elif arm == "mlp_removed":
        selected = [site for site in sites if site.projection in MLP_PROJECTIONS]
    elif arm.startswith("layers_") and arm.endswith("_removed"):
        parts = arm.removesuffix("_removed").split("_")
        if len(parts) != 3:
            raise ValueError(f"invalid layer-range arm {arm!r}")
        lo, hi = int(parts[1]), int(parts[2])
        selected = [site for site in sites if lo <= site.layer <= hi]
    else:
        raise ValueError(f"arm {arm!r} is not an exact-weight removal intervention")
    if not selected:
        raise ValueError(f"arm {arm!r} selected no adapted sites")
    return selected


def _state_dict_equality(left: nn.Module, right: nn.Module) -> dict[str, Any]:
    a, b = left.state_dict(), right.state_dict()
    if set(a) != set(b):
        raise ValueError("Base and Target state_dict keys differ")
    unequal = [name for name in a if not torch.equal(a[name], b[name])]
    return {"tensor_count": len(a), "unequal_count": len(unequal), "unequal_names": unequal}


def _effective_delta_records(
    base: nn.Module, target: nn.Module, sites: Sequence[AdapterSite]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for site in sites:
        base_weight = base.get_submodule(site.module).weight.detach()
        target_weight = target.get_submodule(site.module).weight.detach()
        if base_weight.shape != target_weight.shape:
            raise ValueError(f"shape mismatch at {site.module}")
        delta = target_weight.float() - base_weight.float()
        record = {
            "layer": site.layer,
            "projection": site.projection,
            "module": site.module,
            "shape": list(delta.shape),
            "delta_frobenius_norm": float(torch.linalg.vector_norm(delta).item()),
            "max_abs_delta": float(delta.abs().max().item()),
            "exactly_equal": bool(torch.equal(base_weight, target_weight)),
        }
        if record["exactly_equal"]:
            raise RuntimeError(f"adapted effective matrix is unchanged: {site.module}")
        records.append(record)
        del delta
    return records


@contextmanager
def replace_target_with_base(
    target: nn.Module, base: nn.Module, sites: Sequence[AdapterSite]
) -> Iterator[dict[str, Any]]:
    """Temporarily apply exact Base matrices and prove exact Target restoration."""

    originals: dict[str, Tensor] = {}
    audit: dict[str, Any] = {
        "site_count": len(sites), "replacement_exact": False, "restoration_exact": False
    }
    with torch.no_grad():
        for site in sites:
            target_weight = target.get_submodule(site.module).weight
            base_weight = base.get_submodule(site.module).weight
            if target_weight.shape != base_weight.shape or target_weight.dtype != base_weight.dtype:
                raise ValueError(f"weight incompatibility at {site.module}")
            originals[site.module] = target_weight.detach().clone()
            target_weight.copy_(base_weight)
        audit["replacement_exact"] = all(
            torch.equal(
                target.get_submodule(site.module).weight,
                base.get_submodule(site.module).weight,
            )
            for site in sites
        )
    if not audit["replacement_exact"]:
        raise RuntimeError("one or more exact Base replacements failed")
    try:
        yield audit
    finally:
        with torch.no_grad():
            for site in sites:
                target.get_submodule(site.module).weight.copy_(originals[site.module])
            audit["restoration_exact"] = all(
                torch.equal(target.get_submodule(site.module).weight, originals[site.module])
                for site in sites
            )
        if not audit["restoration_exact"]:
            raise RuntimeError("one or more Target matrices were not restored exactly")


def _load_tokenizer(cfg: MultiModelExperimentConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision, fix_mistral_regex=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _prompt_token_records(tokenizer: Any, prompts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in prompts:
        token_ids = canonical_prompt_ids(tokenizer, str(row["prompt"]), enable_thinking=False)
        records.append({
            **row,
            "prompt_token_ids": token_ids,
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    return records


@torch.inference_mode()
def _probe_logits(model: nn.Module, prompt_tokens: Sequence[Sequence[int]], pad_id: int) -> Tensor:
    width = max(map(len, prompt_tokens))
    device = next(model.parameters()).device
    ids = torch.full((len(prompt_tokens), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, values in enumerate(prompt_tokens):
        ids[row, width - len(values):] = torch.tensor(values, device=device)
        mask[row, width - len(values):] = 1
    return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].detach().cpu()


def _tensor_comparison(left: Tensor, right: Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError("probe logit shapes differ")
    delta = left.float() - right.float()
    denom = max(float(torch.linalg.vector_norm(left.float()).item()), 1e-30)
    return {
        "shape": list(left.shape),
        "exact": bool(torch.equal(left, right)),
        "max_abs_diff": float(delta.abs().max().item()),
        "relative_l2_error": float(torch.linalg.vector_norm(delta).item()) / denom,
    }


def _generate_arm(
    model: nn.Module,
    tokenizer: Any,
    prompt_records: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    generation = DeltaContrastiveGenerationSpec(
        seed=42, temperature=0.0, top_p=1.0, max_new_tokens=max_new_tokens,
        enable_thinking=False, materialization_batch_size=batch_size,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    device = next(model.parameters()).device
    for start in range(0, len(prompt_records), batch_size):
        batch = prompt_records[start:start + batch_size]
        outputs = generate_continuations_batch(
            model, [row["prompt_token_ids"] for row in batch], generation,
            device=device, pad_token_id=int(tokenizer.pad_token_id),
        )
        for prompt, (tokens, stop) in zip(batch, outputs, strict=True):
            rows.append({
                "prompt_id": prompt["prompt_id"],
                "pair_id": prompt.get("pair_id"),
                "shared_scenario_id": prompt.get("shared_scenario_id"),
                "category": prompt["category"],
                "sport_domain": prompt.get("sport_domain"),
                "neutral_domain": prompt.get("neutral_domain"),
                "risk_level": prompt.get("risk_level"),
                "prompt": prompt["prompt"],
                "prompt_token_count": len(prompt["prompt_token_ids"]),
                "prompt_token_sha256": prompt["prompt_token_sha256"],
                "continuation_token_ids": tokens,
                "continuation_text": tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                "new_token_count": len(tokens),
                "eos_reached": bool(stop["stopped_on_eos"]),
                "stop_reason": stop["stop_reason"],
            })
    return rows, time.perf_counter() - started


def _load_pair(cfg: MultiModelExperimentConfig, device: torch.device) -> tuple[nn.Module, nn.Module, list[dict[str, Any]]]:
    models, provenance = [], []
    for spec in cfg.models:
        model, record = load_effective_model(spec, device)
        models.append(model)
        provenance.append(record)
    return models[0], models[1], provenance


def run_positive_control_preflight(
    cfg: MultiModelExperimentConfig,
    prompt_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda",
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Prove full exact-update removal recovers Base and Target restoration is clean."""

    started = time.perf_counter()
    metadata, sites = load_adapter_inventory(cfg)
    selected = select_localization_prompts(prompt_path)
    # One matched pair and one neutral are enough for the engineering gate.
    probe_rows = [
        next(row for row in selected if row["category"] == "risk"),
        next(row for row in selected if row["category"] == "sports_control"),
        next(row for row in selected if row["category"] == "neutral"),
    ]
    tokenizer = _load_tokenizer(cfg)
    prompts = _prompt_token_records(tokenizer, probe_rows)
    prompt_tokens = [row["prompt_token_ids"] for row in prompts]
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index or torch.cuda.current_device())
    base, target, provenance = _load_pair(cfg, device)
    target_probe_before = _probe_logits(target, prompt_tokens, int(tokenizer.pad_token_id))
    clean_target_rows, _ = _generate_arm(
        target, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=3
    )
    clean_base_rows, _ = _generate_arm(
        base, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=3
    )
    state_before = _state_dict_equality(base, target)
    effective = _effective_delta_records(base, target, sites)
    adapted_names = {f"{site.module}.weight" for site in sites}
    unexpected = sorted(set(state_before["unequal_names"]) - adapted_names)
    if unexpected:
        raise RuntimeError(f"Target differs outside LoRA-adapted weights: {unexpected[:10]}")
    with replace_target_with_base(target, base, sites) as replacement_audit:
        full_state = _state_dict_equality(base, target)
        if full_state["unequal_count"] != 0:
            raise RuntimeError("full adapter removal did not recover exact Base state")
        removed_probe = _probe_logits(target, prompt_tokens, int(tokenizer.pad_token_id))
        removed_rows, _ = _generate_arm(
            target, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=3
        )
    target_probe_after = _probe_logits(target, prompt_tokens, int(tokenizer.pad_token_id))
    replay_rows, _ = _generate_arm(
        target, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=3
    )
    base_recovery = _tensor_comparison(removed_probe, _probe_logits(
        base, prompt_tokens, int(tokenizer.pad_token_id)
    ))
    target_replay = _tensor_comparison(target_probe_before, target_probe_after)
    base_tokens = [row["continuation_token_ids"] for row in clean_base_rows]
    removed_tokens = [row["continuation_token_ids"] for row in removed_rows]
    target_tokens = [row["continuation_token_ids"] for row in clean_target_rows]
    replay_tokens = [row["continuation_token_ids"] for row in replay_rows]
    passed = bool(
        replacement_audit["replacement_exact"]
        and replacement_audit["restoration_exact"]
        and full_state["unequal_count"] == 0
        and base_recovery["exact"]
        and target_replay["exact"]
        and base_tokens == removed_tokens
        and target_tokens == replay_tokens
    )
    report = {
        "schema_version": 1,
        "scientific_status": "EXACT_LORA_CAUSAL_LOCALIZATION_PREFLIGHT_PASS" if passed else "FAIL",
        "scientific_boundary": "Engineering positive control only; no localized causal result.",
        "repository_commit": _repository_commit(),
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "model_provenance": provenance,
        "adapter_inventory": metadata,
        "effective_delta": {
            "semantics": "effective BF16 merged Target weight minus pinned BF16 Base weight",
            "records": effective,
            "nonzero_site_count": sum(not row["exactly_equal"] for row in effective),
        },
        "prompt_ids": [row["prompt_id"] for row in prompts],
        "max_new_tokens": max_new_tokens,
        "state_before": state_before,
        "full_removed_state": full_state,
        "replacement_audit": replacement_audit,
        "base_recovery_logits": base_recovery,
        "target_replay_logits": target_replay,
        "base_recovery_generation_exact": base_tokens == removed_tokens,
        "target_replay_generation_exact": target_tokens == replay_tokens,
        "all_probe_arms_reached_eos": all(
            row["eos_reached"] for rows in (clean_base_rows, clean_target_rows, removed_rows, replay_rows)
            for row in rows
        ),
        "passed": passed,
        "resources": {
            "total_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0,
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    if not passed:
        raise RuntimeError(f"exact LoRA positive-control preflight failed: {report}")
    if not _finite_tree(report):
        raise FloatingPointError("preflight contains NaN/Inf")
    write_json_immutable(output_path, report)
    return report


def run_coarse_generation_shard(
    cfg: MultiModelExperimentConfig,
    prompt_path: str | Path,
    preflight_path: str | Path,
    output_path: str | Path,
    *,
    shard: int,
    device_name: str = "cuda",
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> dict[str, Any]:
    if shard not in SHARD_ARMS:
        raise ValueError(f"shard must be one of {sorted(SHARD_ARMS)}")
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise RuntimeError("positive-control preflight did not pass")
    if preflight.get("resolved_config_sha256") != resolved_config_sha256(cfg):
        raise ValueError("preflight/config fingerprint mismatch")
    started = time.perf_counter()
    metadata, sites = load_adapter_inventory(cfg)
    prompt_rows = select_localization_prompts(prompt_path)
    tokenizer = _load_tokenizer(cfg)
    prompts = _prompt_token_records(tokenizer, prompt_rows)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index or torch.cuda.current_device())
    base, target, provenance = _load_pair(cfg, device)
    probe_tokens = [row["prompt_token_ids"] for row in prompts[:3]]
    target_probe_before = _probe_logits(target, probe_tokens, int(tokenizer.pad_token_id))
    arms: list[dict[str, Any]] = []
    for arm in SHARD_ARMS[shard]:
        if arm == "clean_base":
            model, selected_sites, audit = base, [], None
            rows, seconds = _generate_arm(
                model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
            )
        elif arm == "clean_target":
            model, selected_sites, audit = target, [], None
            rows, seconds = _generate_arm(
                model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
            )
        else:
            selected_sites = sites_for_arm(sites, arm)
            with replace_target_with_base(target, base, selected_sites) as audit:
                rows, seconds = _generate_arm(
                    target, tokenizer, prompts,
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                )
            if audit["restoration_exact"] is not True:
                raise RuntimeError(f"restoration audit failed for {arm}")
        arms.append({
            "arm": arm,
            "model_name": "base" if arm == "clean_base" else "target",
            "intervention_kind": "none" if arm.startswith("clean_") else "exact_weight_replacement",
            "removed_site_count": len(selected_sites),
            "removed_layers": sorted({site.layer for site in selected_sites}),
            "removed_projections": sorted({site.projection for site in selected_sites}),
            "replacement_audit": audit,
            "runtime_seconds": seconds,
            "sequences": rows,
        })
    target_probe_after = _probe_logits(target, probe_tokens, int(tokenizer.pad_token_id))
    replay = _tensor_comparison(target_probe_before, target_probe_after)
    if not replay["exact"]:
        raise RuntimeError("Target clean replay changed after intervention restoration")
    if shard == 0:
        by_arm = {arm["arm"]: arm for arm in arms}
        base_rows = by_arm["clean_base"]["sequences"]
        removed_rows = by_arm["full_adapter_removed"]["sequences"]
        if [row["continuation_token_ids"] for row in base_rows] != [
            row["continuation_token_ids"] for row in removed_rows
        ]:
            raise RuntimeError("full removal does not reproduce clean Base generation")
    report = {
        "schema_version": 1,
        "scientific_status": "EXPLORATORY_EXACT_LORA_COARSE_GENERATION_SHARD",
        "scientific_boundary": (
            "Actual effective-weight interventions on a V3-derived exploratory subset; "
            "responses are unscored until blinded judgment."
        ),
        "repository_commit": _repository_commit(),
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "source_preflight": str(preflight_path),
        "model_provenance": provenance,
        "adapter_inventory_summary": metadata,
        "shard": shard,
        "shard_arms": list(SHARD_ARMS[shard]),
        "selection": {
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_file(prompt_path),
            "prompt_ids": [row["prompt_id"] for row in prompts],
            "categories": dict(Counter(row["category"] for row in prompts)),
            "risk_levels": dict(Counter(
                row["risk_level"] for row in prompts if row["category"] == "risk"
            )),
            "sport_domains": sorted({
                row["sport_domain"] for row in prompts if row["sport_domain"] is not None
            }),
        },
        "generation": {
            "decoding": "greedy_argmax", "seed": 42, "temperature": 0.0,
            "max_new_tokens": max_new_tokens, "batch_size": batch_size,
            "enable_thinking": False, "use_kv_cache": True,
        },
        "target_clean_replay_logits": replay,
        "arms": arms,
        "resources": {
            "total_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0,
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    if not _finite_tree(report):
        raise FloatingPointError("generation shard contains NaN/Inf")
    write_json_immutable(output_path, report)
    return report

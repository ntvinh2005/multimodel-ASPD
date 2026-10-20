"""Bounded teacher-forced P6 feasibility pilot on frozen V3 exact-token sequences."""

from __future__ import annotations

import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from aspd.multimodel.cache import PairedActivationCache, load_effective_model
from aspd.multimodel.causal import (
    P6ForwardResult,
    p6_patched_forward,
    temporal_region,
    threshold_causal_gate,
)
from aspd.multimodel.config import ModelSpec, MultiModelExperimentConfig
from aspd.multimodel.fidelity import validate_checkpoint_provenance
from aspd.multimodel.mechanism_discovery import capture_model_forward
from aspd.multimodel.model import LinearSharedEncoder, MultiModelASPD
from aspd.multimodel.p6_localization import calibrate_from_general_validation
from aspd.multimodel.training import build_model_from_cache

P6_INDIVIDUAL_IDS = (3228, 290, 2445, 508, 378)
# Selected before intervention outcomes using prevalence, relative response position,
# total mechanism beta, matrix-locus similarity, and weak frozen-V3 group contrasts.
ACTIVE_CONTROLS = {3228: 1591, 290: 310, 2445: 960, 508: 320, 378: 131}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_tree(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def select_fixed_feasibility_rows(localization: dict[str, Any]) -> list[dict[str, Any]]:
    """Select 6+6+6 Target-generated rows without inspecting intervention outcomes.

    Risk and matched sports-control rows use the first example from each of the first
    six sport domains in frozen artifact order.  Neutral rows use their first six IDs.
    """

    rows = localization["examples"]
    selected: list[dict[str, Any]] = []
    for category in ("risk", "sports_control"):
        seen: set[str] = set()
        for row in rows:
            if row["category"] != category or row["generator_side"] != "target_generated":
                continue
            domain = str(row["sport_domain"])
            if domain not in seen:
                selected.append(row)
                seen.add(domain)
            if len(seen) == 6:
                break
        if len(seen) != 6:
            raise ValueError(f"could not select six sport domains for {category}")
    neutral = [
        row
        for row in rows
        if row["category"] == "neutral" and row["generator_side"] == "target_generated"
    ][:6]
    if len(neutral) != 6:
        raise ValueError("could not select six neutral Target-generated rows")
    selected.extend(neutral)
    if len(selected) != 18 or len({row["example_id"] for row in selected}) != 18:
        raise ValueError("fixed feasibility selection must contain 18 unique rows")
    return selected


def _pad_rows(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[Tensor, Tensor]:
    width = max(int(row["sequence_length"]) for row in rows)
    ids = torch.full((len(rows), width), int(pad_token_id), dtype=torch.long)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        token_ids = torch.tensor(row["token_ids"], dtype=torch.long)
        if token_ids.numel() != int(row["sequence_length"]):
            raise ValueError(f"invalid token count for {row['example_id']}")
        ids[index, : token_ids.numel()] = token_ids
        valid[index, : token_ids.numel()] = True
    return ids, valid


def _weight_audit(model: nn.Module, spec: ModelSpec) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for matrix in spec.matrices:
        weight = model.get_submodule(matrix.module).weight
        values = weight.detach().float()
        try:
            version: int | None = int(weight._version)
        except RuntimeError:
            # Parameters instantiated inside torch.inference_mode() intentionally do
            # not expose a version counter. Numeric checksums below remain available.
            version = None
        audit[matrix.name] = {
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "data_ptr": int(weight.data_ptr()),
            "version": version,
            "sum": float(values.sum().item()),
            "squared_sum": float(values.square().sum().item()),
        }
    return audit


def _summarize_logits(
    clean: Tensor,
    patched: Tensor,
    input_ids_cpu: Tensor,
    valid_cpu: Tensor,
    gate_cpu: Tensor,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report full-set and exact temporal-region teacher-forced effects."""

    if clean.shape != patched.shape or clean.shape[:2] != valid_cpu.shape:
        raise ValueError("clean/patched logits or token-grid shape mismatch")
    clean32, patched32 = clean.float(), patched.float()
    clean_logp = clean32.log_softmax(dim=-1)
    patched_logp = patched32.log_softmax(dim=-1)
    kl = (clean_logp.exp() * (clean_logp - patched_logp)).sum(dim=-1)
    delta = patched32 - clean32
    token_rms = delta.square().mean(dim=-1).sqrt()
    token_max = delta.abs().amax(dim=-1)
    next_ids = input_ids_cpu[:, 1:].to(clean.device)
    observed_delta = (
        patched_logp[:, :-1].gather(-1, next_ids[..., None]).squeeze(-1)
        - clean_logp[:, :-1].gather(-1, next_ids[..., None]).squeeze(-1)
    )
    prediction_valid = valid_cpu[:, :-1] & valid_cpu[:, 1:]
    records: list[dict[str, Any]] = []
    pooled: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    gate_any = gate_cpu.any(dim=-1)
    for index, row in enumerate(rows):
        length = int(row["sequence_length"])
        prompt_length = int(row["prompt_token_count"])
        region_masks: dict[str, list[int]] = defaultdict(list)
        for position in range(length - 1):
            region_masks[temporal_region(position, prompt_length, length)].append(position)
        masks: dict[str, Tensor] = {"all": prediction_valid[index].to(clean.device)}
        for name, positions in region_masks.items():
            mask = torch.zeros(valid_cpu.shape[1] - 1, dtype=torch.bool, device=clean.device)
            mask[positions] = True
            masks[name] = mask & prediction_valid[index].to(clean.device)
        masks["gate_active"] = masks["all"] & gate_any[index, :-1].to(clean.device)
        masks["gate_inactive"] = masks["all"] & ~gate_any[index, :-1].to(clean.device)
        metrics: dict[str, Any] = {}
        for name, mask in masks.items():
            count = int(mask.sum().item())
            if count == 0:
                metrics[name] = {"token_count": 0}
                continue
            values = (
                float(kl[index, :-1][mask].mean().item()),
                float(token_rms[index, :-1][mask].mean().item()),
                float(token_max[index, :-1][mask].max().item()),
                float(observed_delta[index][mask].mean().item()),
            )
            metrics[name] = {
                "token_count": count,
                "mean_kl": values[0],
                "mean_logit_rms": values[1],
                "max_abs_logit_change": values[2],
                "mean_observed_next_token_logprob_delta": values[3],
            }
            pooled[f"all/{name}"].append(values)
            pooled[f"category/{row['category']}/{name}"].append(values)
        records.append(
            {
                "example_id": row["example_id"],
                "category": row["category"],
                "sport_domain": row["sport_domain"],
                "behavioral_score": row["behavioral_score"],
                "sequence_length": length,
                "prompt_token_count": prompt_length,
                "gate_fire_count": int((gate_any[index] & valid_cpu[index]).sum().item()),
                "metrics": metrics,
            }
        )
    aggregate = {
        key: {
            "sequence_region_records": len(values),
            "mean_kl": sum(value[0] for value in values) / len(values),
            "mean_logit_rms": sum(value[1] for value in values) / len(values),
            "max_abs_logit_change": max(value[2] for value in values),
            "mean_observed_next_token_logprob_delta": sum(value[3] for value in values)
            / len(values),
        }
        for key, values in pooled.items()
    }
    return {"aggregate": aggregate, "sequences": records}


def _run_arm(
    model: nn.Module,
    spec: ModelSpec,
    decomposition: MultiModelASPD,
    model_index: int,
    ids: Tensor,
    valid: Tensor,
    clean_logits: Tensor,
    gate: Tensor,
    component_id: int,
    rows: list[dict[str, Any]],
    *,
    arm: str,
    strength: float = 1.0,
    norm_control: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: P6ForwardResult = p6_patched_forward(
        model,
        spec,
        decomposition,
        model_index,
        ids,
        valid,
        gate,
        [component_id],
        strength=strength,
        norm_control=norm_control,
    )
    summary = _summarize_logits(clean_logits, result.logits, ids, valid, gate, rows)
    elapsed = time.perf_counter() - started
    return {
        "arm": arm,
        "intervened_component_id": component_id,
        "strength": strength,
        "norm_matched_orthogonal_output_control": norm_control,
        "runtime_seconds": elapsed,
        "runtime_seconds_per_sequence": elapsed / len(rows),
        "matrix_diagnostics": result.matrix_diagnostics,
        **summary,
    }


@torch.no_grad()
def run_fixed_p6_feasibility(
    cfg: MultiModelExperimentConfig,
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str,
    localization_path: str | Path,
    posthoc_path: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Execute the preregistered 18-sequence individual-component feasibility stage."""

    total_started = time.perf_counter()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    if cfg.sparsity.diffing != "D0" or cfg.sparsity.selection_score != "S1":
        raise ValueError("fixed P6 pilot requires the frozen D0/S1 configuration")
    cache = PairedActivationCache(cfg)
    checkpoint_record = validate_checkpoint_provenance(
        cfg, cache, checkpoint, expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    localization = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    if localization["scientific_status"] != "EXPLORATORY_P6_GATE_LOCALIZATION":
        raise ValueError("unexpected gate-localization artifact")
    if localization["checkpoint"]["sha256"] != expected_checkpoint_sha256:
        raise ValueError("gate localization and fixed pilot use different checkpoints")
    rows = select_fixed_feasibility_rows(localization)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer,
        revision=cfg.data.tokenizer_revision,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ids, valid = _pad_rows(rows, tokenizer.pad_token_id)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    decomposition = build_model_from_cache(cfg, cache)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    decomposition.load_state_dict(state["model"], strict=True)
    del state
    decomposition = decomposition.to(device).float().eval()
    if not isinstance(decomposition.encoder, LinearSharedEncoder):
        raise ValueError("fixed threshold gate requires the registered linear encoder")

    all_components = P6_INDIVIDUAL_IDS + tuple(ACTIVE_CONTROLS[c] for c in P6_INDIVIDUAL_IDS)
    posthoc = load_file(str(posthoc_path))
    calibration = calibrate_from_general_validation(
        decomposition, cache, posthoc, all_components, 16384, device
    )
    for index, component in enumerate(P6_INDIVIDUAL_IDS):
        registered = localization["gate_policy"]["thresholds"][str(component)]
        if float(calibration.thresholds[index]) != float(registered["threshold"]):
            raise ValueError(f"threshold drift for component {component}")

    models: list[nn.Module] = []
    provenance: list[dict[str, Any]] = []
    for spec in cfg.models:
        model, record = load_effective_model(spec, device)
        models.append(model)
        provenance.append(record)
    weight_before = [
        _weight_audit(model, spec)
        for model, spec in zip(models, cfg.models, strict=True)
    ]
    clean = [
        capture_model_forward(model, spec, ids, valid)
        for model, spec in zip(models, cfg.models, strict=True)
    ]
    norms = [float(row["r_rms_norm"]) for row in cache.model_manifests]
    activations = [
        record["R"].float() / norms[index] for index, record in enumerate(clean)
    ]
    preactivations = decomposition.encoder(activations, valid.to(device))
    all_gates = threshold_causal_gate(preactivations, calibration).cpu()
    gates = {
        component: all_gates[..., index : index + 1]
        for index, component in enumerate(all_components)
    }

    model_results: list[dict[str, Any]] = []
    for model_index, (model, spec, clean_record) in enumerate(
        zip(models, cfg.models, clean, strict=True)
    ):
        clean_logits = clean_record["logits"]
        arms: list[dict[str, Any]] = []
        # One explicit alpha=0 arm per model proves the hook path itself is a no-op.
        zero = _run_arm(
            model,
            spec,
            decomposition,
            model_index,
            ids,
            valid,
            clean_logits,
            gates[P6_INDIVIDUAL_IDS[0]],
            P6_INDIVIDUAL_IDS[0],
            rows,
            arm="zero_strength_noop",
            strength=0.0,
        )
        if zero["aggregate"]["all/all"]["max_abs_logit_change"] > 1e-6:
            raise RuntimeError(f"zero-strength P6 is not a no-op for {spec.name}")
        arms.append(zero)
        for candidate in P6_INDIVIDUAL_IDS:
            arms.append(
                _run_arm(
                    model,
                    spec,
                    decomposition,
                    model_index,
                    ids,
                    valid,
                    clean_logits,
                    gates[candidate],
                    candidate,
                    rows,
                    arm=f"candidate_{candidate}",
                )
            )
            control = ACTIVE_CONTROLS[candidate]
            arms.append(
                _run_arm(
                    model,
                    spec,
                    decomposition,
                    model_index,
                    ids,
                    valid,
                    clean_logits,
                    gates[control],
                    control,
                    rows,
                    arm=f"active_control_for_{candidate}",
                )
            )
            arms.append(
                _run_arm(
                    model,
                    spec,
                    decomposition,
                    model_index,
                    ids,
                    valid,
                    clean_logits,
                    gates[candidate],
                    candidate,
                    rows,
                    arm=f"norm_control_for_{candidate}",
                    norm_control=True,
                )
            )
        replay = capture_model_forward(model, spec, ids, valid)["logits"]
        replay_max = float((replay.float() - clean_logits.float()).abs().max().item())
        if replay_max > 1e-6:
            raise RuntimeError(f"clean replay changed after hook removal for {spec.name}")
        model_results.append(
            {
                "model_index": model_index,
                "model_name": spec.name,
                "clean_replay_max_abs_diff": replay_max,
                "arms": arms,
            }
        )

    weight_after = [
        _weight_audit(model, spec)
        for model, spec in zip(models, cfg.models, strict=True)
    ]
    if weight_before != weight_after:
        raise RuntimeError("one or more Qwen matrix weights changed during P6")
    payload = {
        "schema_version": 1,
        "scientific_status": "EXPLORATORY_FIXED_SEQUENCE_P6_FEASIBILITY",
        "scientific_boundary": (
            "Teacher-forced logit intervention on frozen Target-generated V3 sequences; "
            "not an autoregressive behavioral or confirmatory causal result."
        ),
        "checkpoint": checkpoint_record,
        "model_provenance": provenance,
        "selection": {
            "rows": [row["example_id"] for row in rows],
            "rule": (
                "Target-generated only; first frozen example from each of the first six "
                "sport domains for risk and matched sports-control, plus first six neutral IDs"
            ),
            "selected_without_intervention_outcomes": True,
        },
        "candidate_ids": list(P6_INDIVIDUAL_IDS),
        "active_controls": {str(key): value for key, value in ACTIVE_CONTROLS.items()},
        "active_control_selection": {
            "uses_intervention_outcomes": False,
            "matching_variables": [
                "general-validation fire count",
                "Target-generated V3 response support and relative argmax position",
                "total mechanism beta",
                "Base/Target matrix-locus profile",
                "weak frozen-V3 risk/control and permissiveness contrasts",
            ],
            "limitation": (
                "No equally early active control matches component 3228; norm-matched "
                "controls provide exact candidate positions and output-vector norms."
            ),
        },
        "gate_policy": {
            "name": "general_validation_calibrated_component_threshold",
            "uses_future_tokens": False,
            "uses_v3_labels_for_calibration": False,
            "calibration_tokens": calibration.calibration_tokens,
            "thresholds": {
                str(component): {
                    "threshold": float(calibration.thresholds[index]),
                    "target_rate": float(calibration.target_rates[index]),
                    "realized_rate": float(calibration.realized_rates[index]),
                }
                for index, component in enumerate(all_components)
            },
        },
        "norm_control": {
            "construction": (
                "deterministic row-wise Gram-Schmidt projection of a cyclic shift; same "
                "candidate v, same clean gate, orthogonal u with exactly matched L2 norm"
            ),
            "random_seed": None,
        },
        "resources": {
            "total_runtime_seconds": time.perf_counter() - total_started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "weight_audit": {"before": weight_before, "after": weight_after, "unchanged": True},
        "models": model_results,
    }
    if not _finite_tree(payload):
        raise FloatingPointError("fixed P6 artifact contains NaN or Inf")
    artifact = output / "fixed_sequence_feasibility.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


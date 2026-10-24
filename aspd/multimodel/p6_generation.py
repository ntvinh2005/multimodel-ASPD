"""Prefix-causal greedy-generation feasibility for selected ASPD P6 components."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

try:
    import resource
except ModuleNotFoundError:
    resource = None

import torch
from torch import Tensor, nn

from aspd.multimodel.cache import PairedActivationCache, load_effective_model
from aspd.multimodel.causal import (
    ThresholdGateCalibration,
    p6_patched_forward,
    threshold_causal_gate,
    validate_random_control_seeds,
)
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.fidelity import validate_checkpoint_provenance
from aspd.multimodel.mechanism_discovery import capture_model_forward
from aspd.multimodel.model import LinearSharedEncoder, MultiModelASPD
from aspd.multimodel.p6_fixed import ACTIVE_CONTROLS, _weight_audit, select_fixed_feasibility_rows
from aspd.multimodel.training import build_model_from_cache

GENERATION_CANDIDATES = (3228, 290, 508)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_tree(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _pad_token_lists(
    token_lists: list[list[int]], pad_token_id: int
) -> tuple[Tensor, Tensor]:
    width = max(map(len, token_lists))
    ids = torch.full((len(token_lists), width), int(pad_token_id), dtype=torch.long)
    valid = torch.zeros((len(token_lists), width), dtype=torch.bool)
    for index, values in enumerate(token_lists):
        ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
        valid[index, : len(values)] = True
    return ids, valid


def _continuation_difference(left: list[int], right: list[int]) -> dict[str, int]:
    sentinel = object()
    changed = sum(a != b for a, b in zip_longest(left, right, fillvalue=sentinel))
    common = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        common += 1
    return {"changed_token_count": changed, "common_prefix_tokens": common}


def _last_logits(logits: Tensor, lengths: list[int]) -> Tensor:
    rows = torch.arange(len(lengths), device=logits.device)
    positions = torch.tensor(lengths, dtype=torch.long, device=logits.device) - 1
    return logits[rows, positions]


@torch.no_grad()
def _clean_generate(
    model: nn.Module,
    prompt_tokens: list[list[int]],
    *,
    pad_token_id: int,
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    generated = [[] for _ in prompt_tokens]
    finished = [False] * len(prompt_tokens)
    started = time.perf_counter()
    for _step in range(max_new_tokens):
        active = [index for index, done in enumerate(finished) if not done]
        if not active:
            break
        branches = [prompt_tokens[index] + generated[index] for index in active]
        ids, valid = _pad_token_lists(branches, pad_token_id)
        device = next(model.parameters()).device
        result = model(
            input_ids=ids.to(device), attention_mask=valid.to(device), use_cache=False
        )
        next_tokens = _last_logits(result.logits, list(map(len, branches))).argmax(dim=-1)
        for local, index in enumerate(active):
            token = int(next_tokens[local].item())
            generated[index].append(token)
            finished[index] = token in eos_token_ids
        del result
    return {
        "continuations": generated,
        "eos_reached": finished,
        "runtime_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def _patched_generate(
    models: list[nn.Module],
    cfg: MultiModelExperimentConfig,
    decomposition: MultiModelASPD,
    calibration: ThresholdGateCalibration,
    component_local_index: int,
    component_id: int,
    model_index: int,
    prompt_tokens: list[list[int]],
    *,
    pad_token_id: int,
    eos_token_ids: set[int],
    max_new_tokens: int,
    grounding_norms: list[float],
    strength: float = 1.0,
    norm_control: bool = False,
    random_control_seed: int | None = None,
) -> dict[str, Any]:
    """Generate while recomputing the clean shared gate on every actual branch."""

    generated = [[] for _ in prompt_tokens]
    finished = [False] * len(prompt_tokens)
    decision_gate_fires = [0] * len(prompt_tokens)
    prefix_gate_fire_sums = [0] * len(prompt_tokens)
    matrix_squared_l2 = {matrix.name: 0.0 for matrix in cfg.models[model_index].matrices}
    matrix_max_abs = {matrix.name: 0.0 for matrix in cfg.models[model_index].matrices}
    matrix_element_counts = {matrix.name: 0 for matrix in cfg.models[model_index].matrices}
    matrix_gate_active_counts = {matrix.name: 0 for matrix in cfg.models[model_index].matrices}
    random_direction_controls: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for _step in range(max_new_tokens):
        active = [index for index, done in enumerate(finished) if not done]
        if not active:
            break
        branches = [prompt_tokens[index] + generated[index] for index in active]
        ids, valid = _pad_token_lists(branches, pad_token_id)
        clean = [
            capture_model_forward(model, spec, ids, valid)
            for model, spec in zip(models, cfg.models, strict=True)
        ]
        activations = [
            record["R"].float() / grounding_norms[index]
            for index, record in enumerate(clean)
        ]
        preactivations = decomposition.encoder(
            activations, valid.to(next(decomposition.parameters()).device)
        )
        gates = threshold_causal_gate(preactivations, calibration)
        gate = gates[..., component_local_index : component_local_index + 1].cpu()
        patched = p6_patched_forward(
            models[model_index],
            cfg.models[model_index],
            decomposition,
            model_index,
            ids,
            valid,
            gate,
            [component_id],
            strength=strength,
            norm_control=norm_control,
            random_control_seed=random_control_seed,
        )
        lengths = list(map(len, branches))
        next_tokens = _last_logits(patched.logits, lengths).argmax(dim=-1)
        for local, index in enumerate(active):
            decision_gate_fires[index] += int(gate[local, lengths[local] - 1, 0])
            prefix_gate_fire_sums[index] += int(
                (gate[local, : lengths[local], 0] & valid[local, : lengths[local]]).sum()
            )
            token = int(next_tokens[local].item())
            generated[index].append(token)
            finished[index] = token in eos_token_ids
        for name, record in patched.matrix_diagnostics.items():
            matrix_squared_l2[name] += float(record["subtraction_l2"]) ** 2
            matrix_element_counts[name] += int(record["subtraction_element_count"])
            matrix_gate_active_counts[name] += int(record["gate_active_token_count"])
            matrix_max_abs[name] = max(
                matrix_max_abs[name], float(record["max_abs_subtraction"])
            )
            current_random = record.get("random_direction_controls")
            if current_random is not None:
                if (
                    name in random_direction_controls
                    and random_direction_controls[name] != current_random
                ):
                    raise RuntimeError("random write direction changed across decoding steps")
                random_direction_controls[name] = current_random
        del clean, activations, preactivations, gates, gate, patched
    return {
        "continuations": generated,
        "eos_reached": finished,
        "decision_gate_fires": decision_gate_fires,
        "prefix_gate_fire_sums": prefix_gate_fire_sums,
        "matrix_diagnostics": {
            name: {
                "root_sum_squared_full_prefix_l2": math.sqrt(matrix_squared_l2[name]),
                "full_prefix_applied_perturbation_rms": math.sqrt(
                    matrix_squared_l2[name] / matrix_element_counts[name]
                )
                if matrix_element_counts[name]
                else 0.0,
                "full_prefix_gate_active_token_count": matrix_gate_active_counts[name],
                "max_abs_subtraction": matrix_max_abs[name],
                **(
                    {"random_direction_controls": random_direction_controls[name]}
                    if name in random_direction_controls
                    else {}
                ),
            }
            for name in matrix_squared_l2
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _arm_rows(
    rows: list[dict[str, Any]],
    generated: dict[str, Any],
    tokenizer: Any,
    clean_continuations: list[list[int]],
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        continuation = generated["continuations"][index]
        record = {
            "example_id": row["example_id"],
            "category": row["category"],
            "sport_domain": row["sport_domain"],
            "behavioral_score_of_frozen_v3_response": row["behavioral_score"],
            "prompt_token_count": row["prompt_token_count"],
            "continuation_token_ids": continuation,
            "continuation_text": tokenizer.decode(
                continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "new_token_count": len(continuation),
            "eos_reached": generated["eos_reached"][index],
            **_continuation_difference(continuation, clean_continuations[index]),
        }
        for key in ("decision_gate_fires", "prefix_gate_fire_sums"):
            if key in generated:
                record[key] = generated[key][index]
        output.append(record)
    return output


@torch.no_grad()
def run_prefix_generation_feasibility(
    cfg: MultiModelExperimentConfig,
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str,
    localization_path: str | Path,
    fixed_result_path: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    max_new_tokens: int = 64,
    random_control_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    random_seeds = validate_random_control_seeds(random_control_seeds)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    total_started = time.perf_counter()
    cache = PairedActivationCache(cfg)
    checkpoint_record = validate_checkpoint_provenance(
        cfg, cache, checkpoint, expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    localization = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    fixed = json.loads(Path(fixed_result_path).read_text(encoding="utf-8"))
    if fixed["scientific_status"] != "EXPLORATORY_FIXED_SEQUENCE_P6_FEASIBILITY":
        raise ValueError("fixed-sequence gate did not pass")
    if fixed["checkpoint"]["sha256"] != expected_checkpoint_sha256:
        raise ValueError("fixed-sequence and generation checkpoints differ")
    if fixed["weight_audit"]["unchanged"] is not True:
        raise ValueError("fixed-sequence weight audit did not pass")
    for model in fixed["models"]:
        if model["clean_replay_max_abs_diff"] > 1e-6:
            raise ValueError("fixed-sequence clean replay gate did not pass")
        if model["arms"][0]["aggregate"]["all/all"]["max_abs_logit_change"] > 1e-6:
            raise ValueError("fixed-sequence zero-strength gate did not pass")

    rows = select_fixed_feasibility_rows(localization)
    prompt_tokens = [row["token_ids"][: row["prompt_token_count"]] for row in rows]
    if any(len(tokens) != row["prompt_token_count"] for tokens, row in zip(prompt_tokens, rows, strict=True)):
        raise ValueError("prompt slicing changed a frozen prompt length")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer,
        revision=cfg.data.tokenizer_revision,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_token_ids = {int(tokenizer.eos_token_id)}
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    decomposition = build_model_from_cache(cfg, cache)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    decomposition.load_state_dict(state["model"], strict=True)
    del state
    decomposition = decomposition.to(device).float().eval()
    if not isinstance(decomposition.encoder, LinearSharedEncoder):
        raise ValueError("prefix threshold gate requires the registered linear encoder")

    component_ids = GENERATION_CANDIDATES + tuple(
        ACTIVE_CONTROLS[candidate] for candidate in GENERATION_CANDIDATES
    )
    fixed_thresholds = fixed["gate_policy"]["thresholds"]
    calibration = ThresholdGateCalibration(
        component_ids=component_ids,
        thresholds=torch.tensor(
            [fixed_thresholds[str(component)]["threshold"] for component in component_ids]
        ),
        target_rates=torch.tensor(
            [fixed_thresholds[str(component)]["target_rate"] for component in component_ids]
        ),
        realized_rates=torch.tensor(
            [fixed_thresholds[str(component)]["realized_rate"] for component in component_ids]
        ),
        calibration_tokens=int(fixed["gate_policy"]["calibration_tokens"]),
    )
    models: list[nn.Module] = []
    provenance: list[dict[str, Any]] = []
    for spec in cfg.models:
        model, record = load_effective_model(spec, device)
        models.append(model)
        provenance.append(record)
        configured_eos = model.generation_config.eos_token_id
        if configured_eos is not None:
            values = configured_eos if isinstance(configured_eos, list) else [configured_eos]
            eos_token_ids.update(int(value) for value in values)
    weight_before = [
        _weight_audit(model, spec)
        for model, spec in zip(models, cfg.models, strict=True)
    ]
    grounding_norms = [float(row["r_rms_norm"]) for row in cache.model_manifests]

    clean_results = [
        _clean_generate(
            model,
            prompt_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens,
        )
        for model in models
    ]
    arms: list[dict[str, Any]] = []
    for model_index, spec in enumerate(cfg.models):
        clean_continuations = clean_results[model_index]["continuations"]
        arms.append(
            {
                "arm": f"clean_{spec.name}",
                "model_index": model_index,
                "model_name": spec.name,
                "runtime_seconds": clean_results[model_index]["runtime_seconds"],
                "sequences": _arm_rows(
                    rows, clean_results[model_index], tokenizer, clean_continuations
                ),
            }
        )
        if spec.name == "target":
            zero = _patched_generate(
                models,
                cfg,
                decomposition,
                calibration,
                0,
                GENERATION_CANDIDATES[0],
                model_index,
                prompt_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                grounding_norms=grounding_norms,
                strength=0.0,
            )
            if zero["continuations"] != clean_continuations:
                raise RuntimeError("zero-strength generation differs from clean Target")
            arms.append(
                {
                    "arm": "zero_strength_noop_target",
                    "model_index": model_index,
                    "model_name": spec.name,
                    "component_id": GENERATION_CANDIDATES[0],
                    "runtime_seconds": zero["runtime_seconds"],
                    "matrix_diagnostics": zero["matrix_diagnostics"],
                    "sequences": _arm_rows(rows, zero, tokenizer, clean_continuations),
                }
            )
        for candidate in GENERATION_CANDIDATES:
            arm_specs: list[tuple[str, int, bool, int | None]] = [
                (f"candidate_{candidate}", candidate, False, None),
                (f"active_control_for_{candidate}", ACTIVE_CONTROLS[candidate], False, None),
                (f"norm_control_for_{candidate}", candidate, True, None),
            ]
            arm_specs.extend(
                (
                    f"random_control_for_{candidate}_seed_{random_seed}",
                    candidate,
                    False,
                    random_seed,
                )
                for random_seed in random_seeds
            )
            for arm_name, component, norm_control, random_control_seed in arm_specs:
                local_index = component_ids.index(component)
                generated = _patched_generate(
                    models,
                    cfg,
                    decomposition,
                    calibration,
                    local_index,
                    component,
                    model_index,
                    prompt_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_ids=eos_token_ids,
                    max_new_tokens=max_new_tokens,
                    grounding_norms=grounding_norms,
                    norm_control=norm_control,
                    random_control_seed=random_control_seed,
                )
                arm_record = {
                    "arm": f"{spec.name}_{arm_name}",
                    "model_index": model_index,
                    "model_name": spec.name,
                    "candidate_id": candidate,
                    "intervened_component_id": component,
                    "norm_matched_orthogonal_output_control": norm_control,
                    "runtime_seconds": generated["runtime_seconds"],
                    "matrix_diagnostics": generated["matrix_diagnostics"],
                    "sequences": _arm_rows(
                        rows, generated, tokenizer, clean_continuations
                    ),
                }
                if random_control_seed is not None:
                    arm_record["random_same_l2_write_direction_control"] = True
                    arm_record["random_seed"] = random_control_seed
                arms.append(arm_record)

    weight_after = [
        _weight_audit(model, spec)
        for model, spec in zip(models, cfg.models, strict=True)
    ]
    if weight_before != weight_after:
        raise RuntimeError("one or more Qwen weights changed during generation")
    payload = {
        "schema_version": 1,
        "scientific_status": "EXPLORATORY_PREFIX_CAUSAL_P6_GENERATION_FEASIBILITY",
        "scientific_boundary": (
            "Greedy feasibility outputs on the exploratory V3 prompt subset; no blind "
            "behavioral judgment or confirmatory causal claim."
        ),
        "checkpoint": checkpoint_record,
        "model_provenance": provenance,
        "source_fixed_result": str(fixed_result_path),
        "candidate_ids": list(GENERATION_CANDIDATES),
        "active_controls": {
            str(candidate): ACTIVE_CONTROLS[candidate]
            for candidate in GENERATION_CANDIDATES
        },
        "generation": {
            "decoding": "greedy_argmax",
            "max_new_tokens": max_new_tokens,
            "same_prompt_tokens_across_all_arms": True,
            "prompt_response_tokens_from_frozen_v3_used_in_gate": False,
            "use_kv_cache": False,
        },
        "gate_policy": {
            "name": "general_validation_calibrated_component_threshold",
            "clean_base_and_target_same_actual_prefix_each_step": True,
            "recomputed_after_branch_divergence": True,
            "uses_future_tokens": False,
            "uses_other_sequences_for_ranking": False,
            "component_ids": list(component_ids),
        },
        "selection": {
            "rows": [row["example_id"] for row in rows],
            "prompt_only": True,
            "sequences": 18,
        },
        "weight_audit": {"before": weight_before, "after": weight_after, "unchanged": True},
        "resources": {
            "total_runtime_seconds": time.perf_counter() - total_started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "process_max_rss_kib": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if resource is not None
                else None
            ),
        },
        "arms": arms,
    }
    if random_seeds:
        judge_rows_by_arm = {
            arm["arm"]: sum(
                sequence["category"] in {"risk", "sports_control"}
                for sequence in arm["sequences"]
            )
            for arm in arms
        }
        payload["random_control"] = {
            "enabled": True,
            "seeds": list(random_seeds),
            "write_direction": "random same-L2 u",
            "fixed_across_tokens_and_autoregressive_steps": True,
        }
        payload["experiment_manifest"] = {
            "schema_version": 1,
            "arm_names": [arm["arm"] for arm in arms],
            "arm_count": len(arms),
            "judge_prompt_groups": ["risk", "sports_control"],
            "judge_rows_by_arm": judge_rows_by_arm,
            "expected_judge_rows": sum(judge_rows_by_arm.values()),
        }
    if not _finite_tree(payload):
        raise FloatingPointError("prefix generation artifact contains NaN or Inf")
    artifact = output / "prefix_generation_feasibility.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


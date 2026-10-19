"""Run the bounded exploratory causal pilot for Delta feature 10122."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaCrosscoder,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.cache import load_effective_model
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.mechanism_discovery import (
    capture_model_forward,
    delta_feature_values_and_direction,
    layer_input_perturbation,
    load_prompt_rows,
    norm_matched_orthogonal_direction,
    pad_sequence_batch,
    patched_layer_forward,
    sha256_file,
    tokenize_prompt_rows,
)

FEATURE_ID = 10122
ACTIVATION_DIM = 3584
EXPECTED_DELTA_SHA256 = "838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477"


def _json_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _condition_metrics(
    clean: Tensor,
    patched: Tensor,
    other: Tensor,
    input_ids: Tensor,
    length: int,
    prompt_length: int,
) -> dict[str, float]:
    """Teacher-forced continuation and logit metrics for one fixed sequence."""

    # Logits at t predict the canonical token at t+1. The first continuation token
    # is therefore scored at prompt_length-1, the prompt decision point.
    stop = length - 1
    start = prompt_length - 1
    clean_f = clean[:stop].float()
    patched_f = patched[:stop].float()
    other_f = other[:stop].float()
    targets = input_ids[1:length].to(clean.device)
    continuation = slice(start, stop)
    clean_logp = clean_f.log_softmax(dim=-1)
    patched_logp = patched_f.log_softmax(dim=-1)
    token_index = torch.arange(stop, device=clean.device)
    clean_selected = clean_logp[token_index, targets]
    patched_selected = patched_logp[token_index, targets]
    logprob_change = patched_selected[continuation] - clean_selected[continuation]
    clean_slice = clean_f[continuation]
    patched_slice = patched_f[continuation]
    other_slice = other_f[continuation]
    kl = (
        clean_logp[continuation].exp()
        * (clean_logp[continuation] - patched_logp[continuation])
    ).sum(dim=-1)
    clean_distance = (clean_slice - other_slice).square().mean().sqrt()
    patched_distance = (patched_slice - other_slice).square().mean().sqrt()
    movement = patched_slice - clean_slice
    decision_movement = patched_f[start] - clean_f[start]
    return {
        "continuation_logprob_change_sum": float(logprob_change.sum()),
        "continuation_logprob_change_mean": float(logprob_change.mean()),
        "continuation_kl_mean": float(kl.mean()),
        "continuation_logit_rms_change": float(movement.square().mean().sqrt()),
        "continuation_max_abs_logit_change": float(movement.abs().max()),
        "distance_to_other_clean_rms": float(clean_distance),
        "distance_to_other_patched_rms": float(patched_distance),
        "distance_to_other_change": float(patched_distance - clean_distance),
        "decision_logit_rms_change": float(decision_movement.square().mean().sqrt()),
        "decision_first_continuation_logprob_change": float(logprob_change[0]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["category"])].append(row)
        groups[(row["condition"], "all")].append(row)
    output: dict[str, Any] = {}
    metric_names = [
        "continuation_logprob_change_sum",
        "continuation_logprob_change_mean",
        "continuation_kl_mean",
        "continuation_logit_rms_change",
        "continuation_max_abs_logit_change",
        "distance_to_other_change",
        "decision_logit_rms_change",
        "decision_first_continuation_logprob_change",
    ]
    for (condition, category), members in groups.items():
        output[f"{condition}/{category}"] = {
            "sequences": len(members),
            **{
                name: sum(float(row["metrics"][name]) for row in members) / len(members)
                for name in metric_names
            },
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the live-model pilot requires an available CUDA device")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    delta_cfg = load_delta_crosscoder_config(args.delta_config)
    aspd_cfg = load_experiment_config(args.aspd_config)
    if [(m.model_id, m.revision) for m in delta_cfg.models] != [
        (m.model_id, m.revision) for m in aspd_cfg.models
    ]:
        raise ValueError("Delta and ASPD configs do not name the same pinned model pair")
    if delta_cfg.delta_crosscoder.activation_preprocessing != "raw":
        raise ValueError("layer-input intervention requires Delta activation_preprocessing=raw")
    checkpoint = Path(args.delta_checkpoint)
    if sha256_file(checkpoint) != EXPECTED_DELTA_SHA256:
        raise ValueError("Delta checkpoint SHA256 mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        aspd_cfg.data.tokenizer,
        revision=aspd_cfg.data.tokenizer_revision,
        # Transformers' generic Mistral-regex probe otherwise performs a Hub API
        # lookup even when this pinned Qwen tokenizer is already cached.
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_rows = load_prompt_rows(args.prompts)
    sequences = tokenize_prompt_rows(prompt_rows, tokenizer, max_length=args.max_length)

    spec = delta_cfg.delta_crosscoder
    delta_model = DeltaCrosscoder(
        ACTIVATION_DIM,
        resolve_dictionary_size(spec, ACTIVATION_DIM),
        shared_fraction=spec.shared_fraction,
        k_shared=spec.k_shared,
        k_delta=spec.k_delta,
        norm_init_scale=spec.initial_decoder_norm_scale,
    ).to(device)
    # Load the 2.9 GB training checkpoint on host memory and copy only model tensors
    # to CUDA; optimizer state is not retained by this inference-only process.
    checkpoint_payload = load_delta_checkpoint(checkpoint, delta_model, map_location="cpu")
    del checkpoint_payload
    delta_model.eval()
    if FEATURE_ID < delta_model.n_shared:
        raise ValueError("feature 10122 unexpectedly lies outside the Delta partition")

    models = []
    model_provenance = []
    for model_spec in aspd_cfg.models:
        model, provenance = load_effective_model(model_spec, device)
        models.append(model)
        model_provenance.append(provenance)

    detailed: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    max_noop_difference = 0.0
    for offset in range(0, len(sequences), args.batch_size):
        batch_rows = sequences[offset : offset + args.batch_size]
        ids_cpu, valid_cpu = pad_sequence_batch(batch_rows, tokenizer.pad_token_id)
        clean = [
            capture_model_forward(model, model_spec, ids_cpu, valid_cpu)
            for model, model_spec in zip(models, aspd_cfg.models, strict=True)
        ]
        if clean[0]["R"].shape[-1] != ACTIVATION_DIM or clean[1]["R"].shape != clean[0]["R"].shape:
            raise ValueError("live grounding tensors do not match the Delta activation space")
        paired = torch.stack((clean[0]["R"], clean[1]["R"]), dim=2).float()
        feature_values, learned_direction = delta_feature_values_and_direction(
            delta_model, paired, FEATURE_ID
        )
        control_direction = norm_matched_orthogonal_direction(learned_direction)
        feature_values = feature_values * valid_cpu.to(device)
        base_conditions = [
            ("base_add_learned_a0.5", 0, learned_direction, 0.5),
            ("base_add_learned_a1", 0, learned_direction, 1.0),
            ("base_add_control_a0.5", 0, control_direction, 0.5),
            ("base_add_control_a1", 0, control_direction, 1.0),
        ]
        target_conditions = [
            ("target_subtract_learned_a0", 1, learned_direction, -0.0),
            ("target_subtract_learned_a0.5", 1, learned_direction, -0.5),
            ("target_subtract_learned_a1", 1, learned_direction, -1.0),
            ("target_subtract_control_a0.5", 1, control_direction, -0.5),
            ("target_subtract_control_a1", 1, control_direction, -1.0),
        ]
        for condition, model_index, direction, scale in base_conditions + target_conditions:
            perturbation = layer_input_perturbation(feature_values, direction, scale)
            patched = patched_layer_forward(
                models[model_index],
                aspd_cfg.models[model_index],
                ids_cpu,
                valid_cpu,
                perturbation,
            )
            if condition.endswith("a0"):
                max_noop_difference = max(
                    max_noop_difference,
                    float((patched.float() - clean[model_index]["logits"].float()).abs().max()),
                )
            for local_index, row in enumerate(batch_rows):
                length = int(row["sequence_length"])
                metrics = _condition_metrics(
                    clean[model_index]["logits"][local_index],
                    patched[local_index],
                    clean[1 - model_index]["logits"][local_index],
                    ids_cpu[local_index],
                    length,
                    int(row["prompt_length"]),
                )
                detailed.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "category": row["category"],
                        "lexical_condition": row["lexical_condition"],
                        "condition": condition,
                        "patched_model": aspd_cfg.models[model_index].name,
                        "alpha_signed": scale,
                        "metrics": metrics,
                    }
                )
        for local_index, row in enumerate(batch_rows):
            length = int(row["sequence_length"])
            prompt_length = int(row["prompt_length"])
            values = feature_values[local_index, :length].detach().float().cpu()
            firing = (values > 0).nonzero(as_tuple=False).flatten().tolist()
            activation_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "category": row["category"],
                    "lexical_condition": row["lexical_condition"],
                    "prompt_length": prompt_length,
                    "sequence_length": length,
                    "decision_point_value": float(values[prompt_length - 1]),
                    "prompt_fire_count": sum(position < prompt_length for position in firing),
                    "continuation_fire_count": sum(position >= prompt_length for position in firing),
                    "firing_positions": firing,
                    "max_value": float(values.max()),
                }
            )

    # A zero patch passes through the same isolated hook path as real interventions.
    if max_noop_difference > args.noop_tolerance:
        raise RuntimeError(
            f"no-op hook changed logits by {max_noop_difference}, tolerance={args.noop_tolerance}"
        )
    payload = {
        "scientific_status": "EXPLORATORY_DELTA_CAUSAL_PILOT",
        "feature_id": FEATURE_ID,
        "intervention": {
            "site": "model.layers.14 input residual stream",
            "learned_direction": "decoder_target[c] - decoder_base[c]",
            "target_sign": "subtract alpha * z_c * learned_direction",
            "base_sign": "add alpha * z_c * learned_direction",
            "feature_semantics": "thresholded clean paired-prefix activation, held fixed during patch",
            "control": "deterministic orthogonal direction with matched L2 norm",
            "teacher_forced": True,
            "causal_prefix_note": (
                "Qwen attention is causal and Delta encode is token-local under learned thresholds; "
                "R_t and z_t therefore use only the canonical prefix through t despite batched teacher forcing."
            ),
        },
        "provenance": {
            "delta_config": str(Path(args.delta_config)),
            "aspd_config": str(Path(args.aspd_config)),
            "delta_checkpoint": str(checkpoint),
            "delta_checkpoint_sha256": EXPECTED_DELTA_SHA256,
            "prompt_file": str(Path(args.prompts)),
            "prompt_file_sha256": sha256_file(args.prompts),
            "models": [model.model_dump(mode="json") for model in aspd_cfg.models],
            "model_loading": model_provenance,
            "tokenizer": aspd_cfg.data.tokenizer,
            "tokenizer_revision": aspd_cfg.data.tokenizer_revision,
        },
        "integrity": {
            "prompt_count": len(sequences),
            "same_canonical_token_ids": True,
            "future_tokens_used_for_current_feature": False,
            "max_noop_logit_difference": max_noop_difference,
            "noop_tolerance": args.noop_tolerance,
            "learned_direction_norm": float(learned_direction.detach().float().norm()),
            "control_direction_norm": float(control_direction.detach().float().norm()),
            "direction_control_cosine": float(
                torch.nn.functional.cosine_similarity(
                    learned_direction.detach().float(), control_direction.detach().float(), dim=0
                )
            ),
        },
        "feature_activation": activation_rows,
        "measurements": detailed,
        "aggregate": _aggregate(detailed),
    }
    if not _json_finite(payload):
        raise RuntimeError("pilot output contains NaN or Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-config", required=True)
    parser.add_argument("--aspd-config", required=True)
    parser.add_argument("--delta-checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--noop-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "output": args.output,
                "prompt_count": payload["integrity"]["prompt_count"],
                "max_noop_logit_difference": payload["integrity"]["max_noop_logit_difference"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

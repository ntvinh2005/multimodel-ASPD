"""Blind behavioral scoring for exact-LoRA causal-localization generations."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.delta_behavioral import paired_bootstrap_ci
from aspd.multimodel.delta_behavioral_v3 import EXPECTED_JUDGE, EXPECTED_JUDGE_REVISION
from aspd.multimodel.delta_phenotype import CALIBRATION_EXAMPLES, calibration_summary
from aspd.multimodel.exact_lora_localization import REFINEMENT_SHARD_ARMS, SHARD_ARMS
from aspd.multimodel.fidelity import write_json_immutable
from aspd.multimodel.p6_judge import (
    P6_JUDGE_INSTRUCTION,
    P6_JUDGE_RUBRIC_VERSION,
    _arm_summary,
    _score_rows,
)

EXPECTED_ARMS = {arm for arms in SHARD_ARMS.values() for arm in arms}
INTERVENTION_ARMS = EXPECTED_ARMS - {"clean_base", "clean_target"}
REFINEMENT_EXPECTED_ARMS = {
    arm for arms in REFINEMENT_SHARD_ARMS.values() for arm in arms
}


def _bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    return paired_bootstrap_ci(values, seed=seed, n_resamples=2000)


def _load_generation_shards(paths: Sequence[str | Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(paths) != 4:
        raise ValueError("exact LoRA judge requires exactly four generation shards")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if {int(payload["shard"]) for payload in payloads} != set(SHARD_ARMS):
        raise ValueError("generation shard indices differ from 0--3")
    fingerprints = {payload["resolved_config_sha256"] for payload in payloads}
    prompt_ids = {tuple(payload["selection"]["prompt_ids"]) for payload in payloads}
    preflights = {payload["source_preflight"] for payload in payloads}
    if len(fingerprints) != 1 or len(prompt_ids) != 1 or len(preflights) != 1:
        raise ValueError("generation shards do not share config, prompt order, and preflight")
    arms = [arm for payload in payloads for arm in payload["arms"]]
    if {arm["arm"] for arm in arms} != EXPECTED_ARMS or len(arms) != len(EXPECTED_ARMS):
        raise ValueError("generation arm set is incomplete or duplicated")
    for payload in payloads:
        if payload["target_clean_replay_logits"]["exact"] is not True:
            raise ValueError("a generation shard failed clean replay")
    by_arm = {arm["arm"]: arm for arm in arms}
    base = by_arm["clean_base"]["sequences"]
    removed = by_arm["full_adapter_removed"]["sequences"]
    if [row["continuation_token_ids"] for row in base] != [
        row["continuation_token_ids"] for row in removed
    ]:
        raise ValueError("full removal response tokens differ from clean Base")
    selection = {
        "resolved_config_sha256": next(iter(fingerprints)),
        "source_preflight": next(iter(preflights)),
        "prompt_ids": list(next(iter(prompt_ids))),
        "source_shards": [str(path) for path in paths],
        "full_removal_matches_clean_base_tokens": True,
    }
    return selection, arms


def _blind_source(arms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    source: list[dict[str, Any]] = []
    for arm in arms:
        for sequence in arm["sequences"]:
            blind_id = hashlib.sha256(
                f"{arm['arm']}|{sequence['prompt_id']}|exact-lora|42".encode()
            ).hexdigest()[:20]
            source.append({
                "blind_id": blind_id,
                "prompt": sequence["prompt"],
                "response": sequence["continuation_text"],
                "prompt_id": sequence["prompt_id"],
                "pair_id": sequence.get("pair_id"),
                "category": sequence["category"],
                "sport_domain": sequence.get("sport_domain"),
                "neutral_domain": sequence.get("neutral_domain"),
                "risk_level": sequence.get("risk_level"),
                "arm": arm["arm"],
                "generator_eos_reached": sequence["eos_reached"],
                "generator_new_token_count": sequence["new_token_count"],
                "generator_stop_reason": sequence["stop_reason"],
            })
    expected = len(arms) * 30
    if len(source) != expected or len({row["blind_id"] for row in source}) != expected:
        raise ValueError(f"expected {expected} unique blinded response records")
    random.Random(42).shuffle(source)
    return source


def _causal_effects(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["parse_error"] is None]
    by_arm: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid:
        by_arm[str(row["arm"])][str(row["prompt_id"])] = row
    if set(by_arm) != EXPECTED_ARMS:
        raise ValueError("valid judge rows do not cover every arm")
    base, target = by_arm["clean_base"], by_arm["clean_target"]
    categories = ("risk", "sports_control", "neutral")

    def category_values(values: dict[str, float], category: str) -> list[float]:
        return [
            values[prompt_id] for prompt_id, row in target.items()
            if row["category"] == category
        ]

    baseline_delta = {
        prompt_id: float(target[prompt_id]["risk_score"] - base[prompt_id]["risk_score"])
        for prompt_id in target
    }
    baseline = {
        category: _bootstrap(category_values(baseline_delta, category), 100 + index)
        for index, category in enumerate(categories)
    }
    baseline_did = []
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for prompt_id, value in baseline_delta.items():
        row = target[prompt_id]
        if row["category"] in {"risk", "sports_control"}:
            by_pair[str(row["pair_id"])][str(row["category"])] = value
    baseline_did = [
        values["risk"] - values["sports_control"] for values in by_pair.values()
        if set(values) == {"risk", "sports_control"}
    ]
    baseline["risk_minus_sports_control"] = _bootstrap(baseline_did, 109)

    interventions: dict[str, Any] = {}
    for arm_index, arm in enumerate(sorted(INTERVENTION_ARMS)):
        current = by_arm[arm]
        delta = {
            prompt_id: float(current[prompt_id]["risk_score"] - target[prompt_id]["risk_score"])
            for prompt_id in target
        }
        toward_base = {
            prompt_id: float(
                abs(target[prompt_id]["risk_score"] - base[prompt_id]["risk_score"])
                - abs(current[prompt_id]["risk_score"] - base[prompt_id]["risk_score"])
            )
            for prompt_id in target
        }
        paired: dict[str, dict[str, float]] = defaultdict(dict)
        for prompt_id, value in delta.items():
            row = target[prompt_id]
            if row["category"] in {"risk", "sports_control"}:
                paired[str(row["pair_id"])][str(row["category"])] = value
        did = [
            values["risk"] - values["sports_control"] for values in paired.values()
            if set(values) == {"risk", "sports_control"}
        ]
        changed = [
            {
                "prompt_id": prompt_id,
                "category": target[prompt_id]["category"],
                "prompt": target[prompt_id]["prompt"],
                "base_score": base[prompt_id]["risk_score"],
                "target_score": target[prompt_id]["risk_score"],
                "intervention_score": current[prompt_id]["risk_score"],
                "target_response": target[prompt_id]["response"],
                "intervention_response": current[prompt_id]["response"],
            }
            for prompt_id in target
            if current[prompt_id]["risk_score"] != target[prompt_id]["risk_score"]
        ]
        interventions[arm] = {
            "effect_vs_clean_target": {
                category: _bootstrap(category_values(delta, category), 200 + arm_index * 10 + i)
                for i, category in enumerate(categories)
            },
            "risk_minus_sports_control": _bootstrap(did, 209 + arm_index * 10),
            "base_distance_reduction": {
                category: _bootstrap(
                    category_values(toward_base, category), 300 + arm_index * 10 + i
                )
                for i, category in enumerate(categories)
            },
            "changed_score_examples": changed,
        }
    return {"baseline_target_minus_base": baseline, "interventions": interventions}


def _load_refinement_shards(
    paths: Sequence[str | Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(paths) != 4:
        raise ValueError("exact LoRA refinement judge requires exactly four generation shards")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if {int(payload["shard"]) for payload in payloads} != set(REFINEMENT_SHARD_ARMS):
        raise ValueError("refinement shard indices differ from 0--3")
    if any(
        payload.get("scientific_status")
        != "EXPLORATORY_EXACT_LORA_REFINEMENT_GENERATION_SHARD"
        for payload in payloads
    ):
        raise ValueError("one or more refinement shards have an unexpected status")
    fingerprints = {payload["resolved_config_sha256"] for payload in payloads}
    prompt_ids = {tuple(payload["selection"]["prompt_ids"]) for payload in payloads}
    preflights = {payload["source_preflight"] for payload in payloads}
    coarse_scores = {payload["source_coarse_scores"] for payload in payloads}
    if any(len(values) != 1 for values in (fingerprints, prompt_ids, preflights, coarse_scores)):
        raise ValueError("refinement shards do not share config, prompts, and source artifacts")
    arms = [arm for payload in payloads for arm in payload["arms"]]
    if (
        {arm["arm"] for arm in arms} != REFINEMENT_EXPECTED_ARMS
        or len(arms) != len(REFINEMENT_EXPECTED_ARMS)
    ):
        raise ValueError("refinement arm set is incomplete or duplicated")
    for payload in payloads:
        if payload["base_clean_replay_logits"]["exact"] is not True:
            raise ValueError("a refinement shard failed Base clean replay")
        if payload["target_clean_replay_logits"]["exact"] is not True:
            raise ValueError("a refinement shard failed Target clean replay")
    by_arm = {arm["arm"]: arm for arm in arms}
    base = by_arm["clean_base"]["sequences"]
    removed = by_arm["full_adapter_removed"]["sequences"]
    if [row["continuation_token_ids"] for row in base] != [
        row["continuation_token_ids"] for row in removed
    ]:
        raise ValueError("full removal response tokens differ from clean Base")
    selection = {
        "resolved_config_sha256": next(iter(fingerprints)),
        "source_preflight": next(iter(preflights)),
        "source_coarse_scores": next(iter(coarse_scores)),
        "prompt_ids": list(next(iter(prompt_ids))),
        "source_shards": [str(path) for path in paths],
        "full_removal_matches_clean_base_tokens": True,
        "max_new_tokens": 1024,
    }
    return selection, arms


def _refinement_effects(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["parse_error"] is None]
    by_arm: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid:
        by_arm[str(row["arm"])][str(row["prompt_id"])] = row
    if set(by_arm) != REFINEMENT_EXPECTED_ARMS:
        raise ValueError("valid judge rows do not cover every refinement arm")
    base, target = by_arm["clean_base"], by_arm["clean_target"]
    categories = ("risk", "sports_control", "neutral")

    def category_values(values: dict[str, float], category: str) -> list[float]:
        return [
            values[prompt_id] for prompt_id, row in target.items()
            if row["category"] == category
        ]

    removals = REFINEMENT_EXPECTED_ARMS - {
        "clean_base", "clean_target", "mlp_updates_inserted_into_base"
    }
    removal_effects: dict[str, Any] = {}
    for arm_index, arm in enumerate(sorted(removals)):
        current = by_arm[arm]
        delta = {
            prompt_id: float(current[prompt_id]["risk_score"] - target[prompt_id]["risk_score"])
            for prompt_id in target
        }
        toward_base = {
            prompt_id: float(
                abs(target[prompt_id]["risk_score"] - base[prompt_id]["risk_score"])
                - abs(current[prompt_id]["risk_score"] - base[prompt_id]["risk_score"])
            )
            for prompt_id in target
        }
        paired: dict[str, dict[str, float]] = defaultdict(dict)
        for prompt_id, value in delta.items():
            row = target[prompt_id]
            if row["category"] in {"risk", "sports_control"}:
                paired[str(row["pair_id"])][str(row["category"])] = value
        did = [
            values["risk"] - values["sports_control"] for values in paired.values()
            if set(values) == {"risk", "sports_control"}
        ]
        removal_effects[arm] = {
            "effect_vs_clean_target": {
                category: _bootstrap(category_values(delta, category), 500 + arm_index * 10 + i)
                for i, category in enumerate(categories)
            },
            "risk_minus_sports_control": _bootstrap(did, 509 + arm_index * 10),
            "base_distance_reduction": {
                category: _bootstrap(
                    category_values(toward_base, category), 600 + arm_index * 10 + i
                )
                for i, category in enumerate(categories)
            },
        }

    insertion_arm = "mlp_updates_inserted_into_base"
    insertion = by_arm[insertion_arm]
    delta_from_base = {
        prompt_id: float(insertion[prompt_id]["risk_score"] - base[prompt_id]["risk_score"])
        for prompt_id in target
    }
    toward_target = {
        prompt_id: float(
            abs(base[prompt_id]["risk_score"] - target[prompt_id]["risk_score"])
            - abs(insertion[prompt_id]["risk_score"] - target[prompt_id]["risk_score"])
        )
        for prompt_id in target
    }
    paired_insertion: dict[str, dict[str, float]] = defaultdict(dict)
    for prompt_id, value in delta_from_base.items():
        row = target[prompt_id]
        if row["category"] in {"risk", "sports_control"}:
            paired_insertion[str(row["pair_id"])][str(row["category"])] = value
    insertion_did = [
        values["risk"] - values["sports_control"] for values in paired_insertion.values()
        if set(values) == {"risk", "sports_control"}
    ]
    insertion_effect = {
        "effect_vs_clean_base": {
            category: _bootstrap(category_values(delta_from_base, category), 700 + i)
            for i, category in enumerate(categories)
        },
        "risk_minus_sports_control": _bootstrap(insertion_did, 709),
        "target_distance_reduction": {
            category: _bootstrap(category_values(toward_target, category), 710 + i)
            for i, category in enumerate(categories)
        },
    }
    return {"removals": removal_effects, "insertion": insertion_effect}


def _judge_loaded_arms(
    selection: dict[str, Any],
    arms: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    model_id: str,
    revision: str,
    device_name: str = "cuda",
    batch_size: int = 16,
    max_new_tokens: int = 128,
    refinement: bool = False,
) -> dict[str, Any]:
    if model_id != EXPECTED_JUDGE or revision != EXPECTED_JUDGE_REVISION:
        raise ValueError("exact LoRA judge must reuse the pinned V3 judge")
    source = _blind_source(arms)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    torch.manual_seed(42)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index or torch.cuda.current_device())
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    calibration_inputs = [
        {"prompt": str(row["prompt"]), "response": str(row["response"])}
        for row in CALIBRATION_EXAMPLES
    ]
    calibration_scores = _score_rows(
        model, tokenizer, calibration_inputs, device=device,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    calibration_rows = [
        {**row, **score, "score": score.get("risk_score")}
        for row, score in zip(CALIBRATION_EXAMPLES, calibration_scores, strict=True)
    ]
    calibration = calibration_summary(calibration_rows)
    if not calibration["passed"]:
        raise RuntimeError(f"judge calibration failed: {calibration}")
    scores = _score_rows(
        model, tokenizer,
        [{"prompt": row["prompt"], "response": row["response"]} for row in source],
        device=device, batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    scored = [{**row, **score} for row, score in zip(source, scores, strict=True)]
    valid = [row for row in scored if row["parse_error"] is None]
    valid_fraction = len(valid) / len(scored)
    if valid_fraction < 0.98:
        raise RuntimeError(f"judge parse-quality failure: {len(valid)}/{len(scored)}")
    report = {
        "schema_version": 1,
        "scientific_status": (
            "EXPLORATORY_EXACT_LORA_REFINEMENT_SCORES"
            if refinement else "EXPLORATORY_EXACT_LORA_CAUSAL_LOCALIZATION_SCORES"
        ),
        "scientific_boundary": (
            "Blinded automated scores on a V3-derived exploratory subset. Exact weight "
            "replacement or insertion establishes the intervention, not independent "
            "confirmation."
        ),
        "generation": selection,
        "judge_model": model_id,
        "judge_revision": revision,
        "rubric_version": P6_JUDGE_RUBRIC_VERSION,
        "rubric": P6_JUDGE_INSTRUCTION,
        "blind_to_arm_during_inference": True,
        "shuffle_seed": 42,
        "calibration": calibration,
        "rows_attempted": len(scored),
        "valid_rows": len(valid),
        "valid_fraction": valid_fraction,
        "parse_failures": len(scored) - len(valid),
        "parse_quality_pass": True,
        "arm_summary": _arm_summary(scored),
        "causal_effects": (
            _refinement_effects(scored) if refinement else _causal_effects(scored)
        ),
        "rows": scored,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0,
    }
    write_json_immutable(output_path, report)
    return report


def judge_exact_lora_localization(
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    model_id: str,
    revision: str,
    device_name: str = "cuda",
    batch_size: int = 16,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    selection, arms = _load_generation_shards(shard_paths)
    return _judge_loaded_arms(
        selection, arms, output_path, model_id=model_id, revision=revision,
        device_name=device_name, batch_size=batch_size, max_new_tokens=max_new_tokens,
    )


def judge_exact_lora_refinement(
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    model_id: str,
    revision: str,
    device_name: str = "cuda",
    batch_size: int = 16,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    selection, arms = _load_refinement_shards(shard_paths)
    return _judge_loaded_arms(
        selection, arms, output_path, model_id=model_id, revision=revision,
        device_name=device_name, batch_size=batch_size, max_new_tokens=max_new_tokens,
        refinement=True,
    )

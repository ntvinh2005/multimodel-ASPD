"""Evaluate the frozen 100 Delta features on confirmatory behavioral V2."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaCrosscoder,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.delta_behavioral import (
    benjamini_hochberg,
    is_special_or_boundary_maximum,
    load_jsonl,
    matched_differences,
    paired_bootstrap_ci,
    paired_effect,
    paired_sign_flip_pvalue,
    pearson_correlation,
    positive_region_summary,
    response_prevalence,
    spearman_correlation,
    support_summary,
    unpaired_bootstrap_difference_ci,
    validate_provenance,
    validate_v2_benchmark,
)

EXPECTED_GROUP_COUNTS = {
    "target_tail": 40,
    "base_tail": 20,
    "balanced_delta": 20,
    "shared_control": 20,
}
PREREGISTRATION_PATH = Path(
    "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v2_manifest.json"
)


def _window(
    token_ids: list[int], values: torch.Tensor, center: int, tokenizer: Any, radius: int = 16
) -> dict[str, Any]:
    start, stop = max(0, center - radius), min(len(token_ids), center + radius + 1)
    ids = token_ids[start:stop]
    return {
        "token_ids": ids,
        "token_strings": tokenizer.convert_ids_to_tokens(ids),
        "token_texts": [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in ids
        ],
        "token_activations": [float(value) for value in values[start:stop]],
        "center_in_window": center - start,
        "position_in_sequence": center,
        "center_activation": float(values[center]),
        "text": tokenizer.decode(ids, clean_up_tokenization_spaces=False),
    }


def _region(values: torch.Tensor, lo: int, hi: int) -> tuple[dict[str, float | int], int | None]:
    portion = values[lo:hi]
    summary = positive_region_summary(portion)
    if not portion.numel() or float(summary["max_positive"]) <= 0:
        return summary, None
    return summary, lo + int(portion.argmax())


def _by_pair(rows: list[dict[str, Any]], category: str, field: str) -> dict[str, float]:
    selected = [row for row in rows if row["category"] == category]
    values = {str(row["pair_id"]): float(row[field]) for row in selected}
    if len(values) != 60:
        raise ValueError(f"expected 60 unique {category} pairs, got {len(values)}")
    return values


def _context_view(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        key: row.get(key)
        for key in (
            "prompt_id",
            "pair_id",
            "category",
            "sport_domain",
            "risk_level",
            "generator_side",
            "prompt",
            "generated_answer",
            "response_max",
            "response_boundary_maximum",
            "response_context",
        )
    }


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    prior_analysis: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_delta_crosscoder_config(config_path)
    cache = Path(cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    validate_provenance(manifest, activation_width=int(manifest["activation_dim"]))
    if not manifest.get("construction", {}).get("responses_frozen_before_feature_measurement"):
        raise ValueError("V2 activation cache does not prove responses were frozen first")
    examples = load_jsonl(cache / str(manifest["examples_file"]))
    prompt_rows = load_jsonl(cfg.delta_contrastive.prompt_source.path)
    benchmark = validate_v2_benchmark(prompt_rows)
    prompts = {str(row["prompt_id"]): row for row in prompt_rows}
    for example in examples:
        metadata = prompts.get(str(example["prompt_id"]))
        if metadata is None:
            raise ValueError(f"cache references unknown prompt {example['prompt_id']}")
        example.update(
            {
                "category": metadata["category"],
                "pair_id": metadata.get("pair_id"),
                "sport_domain": metadata["sport_domain"],
                "risk_level": metadata["risk_level"],
            }
        )
    if len(examples) != 360:
        raise ValueError(f"expected 360 paired-measurement sequences, got {len(examples)}")

    response_root = Path(manifest["frozen_response_artifact"]["path"])
    scores_path = response_root / "behavioral_scores.json"
    score_artifact: dict[str, Any] = {}
    score_by_prompt: dict[str, dict[str, float]] = {}
    if scores_path.exists():
        score_artifact = json.loads(scores_path.read_text(encoding="utf-8"))
        if score_artifact.get("judge_model") != "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            raise ValueError("behavioral score artifact uses an unregistered judge model")
        if score_artifact.get("judge_revision") != "916b56a44061fd5cd7d6a8fb632557ed4f724f60":
            raise ValueError("behavioral score artifact uses an unregistered judge revision")
        score_by_prompt = {
            str(row["prompt_id"]): {
                "base_generated": float(row["score_base"]),
                "target_generated": float(row["score_target"]),
            }
            for row in score_artifact.get("rows", [])
            if row.get("parse_error") is None
        }

    metrics_path = Path(prior_analysis) / "feature_metrics.json"
    prior = json.loads(metrics_path.read_text(encoding="utf-8"))["features"]
    selected = [row for row in prior if row.get("selected_group")]
    counts = Counter(str(row["selected_group"]) for row in selected)
    if len(selected) != 100 or dict(counts) != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            f"frozen feature set changed: count={len(selected)}, groups={dict(counts)}"
        )
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    frozen_ids = preregistration["measurement_policy"]["frozen_feature_ids"]
    observed_ids: dict[str, list[int]] = {
        group: sorted(int(row["feature_id"]) for row in selected if row["selected_group"] == group)
        for group in EXPECTED_GROUP_COUNTS
    }
    if observed_ids != frozen_ids:
        raise ValueError("selected feature IDs differ from the preregistered frozen 100")
    selected.sort(key=lambda row: int(row["feature_id"]))
    feature_ids = [int(row["feature_id"]) for row in selected]
    prior_by_id = {int(row["feature_id"]): row for row in selected}

    device = torch.device(device_name)
    activation_dim = int(manifest["activation_dim"])
    spec = cfg.delta_crosscoder
    model = DeltaCrosscoder(
        activation_dim,
        resolve_dictionary_size(spec, activation_dim),
        shared_fraction=spec.shared_fraction,
        k_shared=spec.k_shared,
        k_delta=spec.k_delta,
        norm_init_scale=spec.initial_decoder_norm_scale,
    ).to(device)
    checkpoint = load_delta_checkpoint(checkpoint_path, model, map_location=device)
    checkpoint_models = checkpoint.get("config", {}).get("models", [])
    if [entry.get("model_id") for entry in checkpoint_models] != manifest["models"]["ids"]:
        raise ValueError("Delta checkpoint model IDs differ from the V2 activation cache")
    if [entry.get("revision") for entry in checkpoint_models] != manifest["models"]["revisions"]:
        raise ValueError("Delta checkpoint revisions differ from the V2 activation cache")
    decoder_shape = tuple(checkpoint["model"]["decoder_weight"].shape)
    if decoder_shape != (2, resolve_dictionary_size(spec, activation_dim), 3584):
        raise ValueError(f"unexpected Delta checkpoint decoder shape: {decoder_shape}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    index_to_example = {index: row for index, row in enumerate(examples)}
    sequence_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with torch.inference_mode():
        for split in ("train", "validation"):
            for entry in manifest["splits"][split]["shards"]:
                shard = torch.load(cache / entry["path"], map_location="cpu", weights_only=True)
                for row_index, example_index in enumerate(shard["example_indices"].tolist()):
                    example = index_to_example[int(example_index)]
                    length = int(shard["valid_tokens"][row_index].sum())
                    token_ids = [int(value) for value in example["token_ids"][:length]]
                    prompt_end = min(int(example["prompt_token_count"]), length)
                    encoded = (
                        model.encode(
                            shard["activations"][row_index, :length].to(
                                device=device, dtype=torch.float32
                            ),
                            use_threshold=True,
                        )
                        .detach()
                        .float()
                        .cpu()[:, feature_ids]
                    )
                    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
                    token_texts = [
                        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                        for token_id in token_ids
                    ]
                    for column, feature_id in enumerate(feature_ids):
                        values = encoded[:, column]
                        prompt_summary, prompt_center = _region(values, 0, prompt_end)
                        response_summary, response_center = _region(values, prompt_end, length)
                        boundary = False
                        response_context = None
                        if response_center is not None:
                            boundary = is_special_or_boundary_maximum(
                                token_id=token_ids[response_center],
                                token_text=token_texts[response_center],
                                token_string=token_strings[response_center],
                                position=response_center,
                                prompt_end=prompt_end,
                                sequence_length=length,
                                special_token_ids=special_ids,
                            )
                            response_context = _window(
                                token_ids, values, response_center, tokenizer
                            )
                        sequence_rows[feature_id].append(
                            {
                                "prompt_id": example["prompt_id"],
                                "pair_id": example["pair_id"],
                                "category": example["category"],
                                "sport_domain": example["sport_domain"],
                                "risk_level": example["risk_level"],
                                "generator_side": example["generator_side"],
                                "prompt": example["prompt"],
                                "generated_answer": example["generated_answer"],
                                "prompt_max": prompt_summary["max_positive"],
                                "prompt_positive_count": prompt_summary["positive_count"],
                                "prompt_mean_positive": prompt_summary["mean_positive"],
                                "prompt_max_position": prompt_center,
                                "response_max": response_summary["max_positive"],
                                "response_positive_count": response_summary["positive_count"],
                                "response_mean_positive": response_summary["mean_positive"],
                                "response_max_position": response_center,
                                "response_boundary_maximum": boundary,
                                "response_context": response_context,
                            }
                        )

    rows: list[dict[str, Any]] = []
    contexts: dict[str, Any] = {}
    raw_pvalues: dict[int, float] = {}
    for feature_id in feature_ids:
        all_rows = sequence_rows[feature_id]
        by_generator = {
            generator: [row for row in all_rows if row["generator_side"] == generator]
            for generator in ("target_generated", "base_generated")
        }
        generator_effects: dict[str, Any] = {}
        target_differences_by_pair: dict[str, float] = {}
        for generator, generator_rows in by_generator.items():
            risk = _by_pair(generator_rows, "risk", "response_max")
            control = _by_pair(generator_rows, "sports_control", "response_max")
            differences = matched_differences(risk, control)
            pair_ids = sorted(risk)
            effects = paired_effect(differences)
            bootstrap = paired_bootstrap_ci(differences, seed=42, n_resamples=2000)
            pvalue = paired_sign_flip_pvalue(differences, seed=42, n_permutations=5000)
            risk_rows = [row for row in generator_rows if row["category"] == "risk"]
            control_rows = [row for row in generator_rows if row["category"] == "sports_control"]
            neutral_rows = [row for row in generator_rows if row["category"] == "neutral"]
            pair_map = dict(zip(pair_ids, differences, strict=True))
            support = support_summary(risk_rows, pair_map)
            generator_effects[generator] = {
                "matched": {**effects, "bootstrap_95": bootstrap, "raw_p": pvalue},
                "risk": response_prevalence([row["response_max"] for row in risk_rows]),
                "sports_control": response_prevalence(
                    [row["response_max"] for row in control_rows]
                ),
                "neutral": response_prevalence([row["response_max"] for row in neutral_rows]),
                "risk_vs_neutral": unpaired_bootstrap_difference_ci(
                    [row["response_max"] for row in risk_rows],
                    [row["response_max"] for row in neutral_rows],
                    seed=42,
                    n_resamples=2000,
                ),
                "support": support,
            }
            if generator == "target_generated":
                target_differences_by_pair = pair_map
                raw_pvalues[feature_id] = pvalue

        target_rows = by_generator["target_generated"]
        prompt_risk = _by_pair(target_rows, "risk", "prompt_max")
        prompt_control = _by_pair(target_rows, "sports_control", "prompt_max")
        prompt_effect = paired_effect(matched_differences(prompt_risk, prompt_control))
        active_target_risk = [
            row
            for row in target_rows
            if row["category"] == "risk" and float(row["response_max"]) > 0
        ]
        boundary_fraction = (
            sum(bool(row["response_boundary_maximum"]) for row in active_target_risk)
            / len(active_target_risk)
            if active_target_risk
            else 0.0
        )
        target = generator_effects["target_generated"]
        prior_row = prior_by_id[feature_id]
        row = {
            "feature_id": feature_id,
            "candidate_group": prior_row["selected_group"],
            "partition": prior_row["partition"],
            "r_base": prior_row["relative_decoder_norm_base"],
            "r_target": prior_row["relative_decoder_norm_target_derived"],
            "target_generated": target,
            "base_generated": generator_effects["base_generated"],
            "prompt_region_target_matched": prompt_effect,
            "response_region_target_matched": target["matched"],
            "special_or_boundary_top_context_fraction": boundary_fraction,
            "support_floor_pass": target["support"]["active_risk_responses"] >= 10,
            "domain_floor_pass": target["support"]["positive_effect_domains"] >= 4,
            "positive_response_effect_pass": target["matched"]["mean"] > 0,
            "bootstrap_pass": target["matched"]["bootstrap_95"]["lower_95"] > 0,
            "response_over_prompt_pass": target["matched"]["mean"] > prompt_effect["mean"],
            "formatting_control_pass": boundary_fraction <= 0.5,
            "behavioral_scores_available": bool(score_by_prompt),
            "v1_exploratory_candidate": feature_id == 5408,
        }
        if score_by_prompt:
            association_rows = [
                item for item in all_rows if str(item["prompt_id"]) in score_by_prompt
            ]
            activations = [float(item["response_max"]) for item in association_rows]
            behavior_scores = [
                score_by_prompt[str(item["prompt_id"])][str(item["generator_side"])]
                for item in association_rows
            ]
            risk_indices = [
                index for index, item in enumerate(association_rows) if item["category"] == "risk"
            ]
            row["behavior_score_association"] = {
                "all_sequences": {
                    "n": len(activations),
                    "pearson": pearson_correlation(activations, behavior_scores),
                    "spearman": spearman_correlation(activations, behavior_scores),
                },
                "risk_sequences": {
                    "n": len(risk_indices),
                    "pearson": pearson_correlation(
                        [activations[index] for index in risk_indices],
                        [behavior_scores[index] for index in risk_indices],
                    ),
                    "spearman": spearman_correlation(
                        [activations[index] for index in risk_indices],
                        [behavior_scores[index] for index in risk_indices],
                    ),
                },
            }
        rows.append(row)

        risk_by_pair = {
            str(item["pair_id"]): item for item in target_rows if item["category"] == "risk"
        }
        control_by_pair = {
            str(item["pair_id"]): item
            for item in target_rows
            if item["category"] == "sports_control"
        }
        base_risk = {
            str(item["pair_id"]): item
            for item in by_generator["base_generated"]
            if item["category"] == "risk"
        }
        base_control = {
            str(item["pair_id"]): item
            for item in by_generator["base_generated"]
            if item["category"] == "sports_control"
        }
        top_pairs = sorted(
            target_differences_by_pair,
            key=lambda pair_id: (-abs(target_differences_by_pair[pair_id]), pair_id),
        )[:12]
        neutral_contexts = sorted(
            (item for item in target_rows if item["category"] == "neutral"),
            key=lambda item: -float(item["response_max"]),
        )[:8]
        contexts[str(feature_id)] = {
            "matched_pairs": [
                {
                    "pair_id": pair_id,
                    "target_generated_d": target_differences_by_pair[pair_id],
                    "target_generated": {
                        "risk": _context_view(risk_by_pair[pair_id]),
                        "sports_control": _context_view(control_by_pair[pair_id]),
                    },
                    "base_generated": {
                        "risk": _context_view(base_risk[pair_id]),
                        "sports_control": _context_view(base_control[pair_id]),
                    },
                }
                for pair_id in top_pairs
            ],
            "neutral_target_generated": [_context_view(item) for item in neutral_contexts],
        }

    qvalues = benjamini_hochberg(raw_pvalues)
    for row in rows:
        row["bh_q"] = qvalues[int(row["feature_id"])]
        row["bh_pass"] = row["bh_q"] < 0.10
        automatic = all(
            bool(row[key])
            for key in (
                "support_floor_pass",
                "domain_floor_pass",
                "positive_response_effect_pass",
                "bootstrap_pass",
                "bh_pass",
                "response_over_prompt_pass",
                "formatting_control_pass",
            )
        )
        row["automatic_confirmatory_criteria_pass"] = (
            row["candidate_group"] == "target_tail" and automatic
        )
        row["diagnostic_status"] = (
            "automatic candidate; semantic review and model behavior unresolved"
            if row["automatic_confirmatory_criteria_pass"]
            else "sparse support"
            if not row["support_floor_pass"]
            else "formatting/boundary dominated"
            if not row["formatting_control_pass"]
            else "confirmatory criteria not met"
        )

    rows.sort(
        key=lambda row: (
            not bool(row["automatic_confirmatory_criteria_pass"]),
            float(row["bh_q"]),
            -float(row["target_generated"]["matched"]["mean"]),
            int(row["feature_id"]),
        )
    )
    group_summary: dict[str, Any] = {}
    for group in EXPECTED_GROUP_COUNTS:
        group_rows = [row for row in rows if row["candidate_group"] == group]
        group_summary[group] = {
            "features": len(group_rows),
            "automatic_confirmatory_candidates": sum(
                bool(row["automatic_confirmatory_criteria_pass"]) for row in group_rows
            ),
            "mean_matched_d": statistics.fmean(
                float(row["target_generated"]["matched"]["mean"]) for row in group_rows
            ),
            "median_active_risk_responses": statistics.median(
                int(row["target_generated"]["support"]["active_risk_responses"])
                for row in group_rows
            ),
        }

    automatic_candidates = [row for row in rows if row["automatic_confirmatory_criteria_pass"]]
    behavior_available = bool(score_by_prompt)
    condition_1_pass = bool(score_artifact.get("condition_1_pass")) if behavior_available else False
    if not behavior_available:
        decision = "BORDERLINE"
        decision_reasons = [
            "blind model-level behavioral scores are unavailable, so condition 1 is unresolved",
            "manual semantic coherence review is pending",
        ]
    elif not condition_1_pass:
        decision = "BENCHMARK FAILURE"
        decision_reasons = [
            "the blinded V2 review did not validate stronger Target risk-seeking behavior",
            "Delta is not classified as failed when the organism behavior was not elicited",
        ]
    elif not automatic_candidates:
        decision = "FAIL"
        decision_reasons = [
            "blinded model-level behavior passed",
            "no frozen target-tail feature passed all automatic confirmatory criteria",
        ]
    else:
        decision = "BORDERLINE"
        decision_reasons = [
            "blinded model-level behavior passed",
            "at least one target-tail feature passed automatic criteria",
            "manual saved-context semantic coherence review remains unresolved",
        ]
    metadata = {
        "benchmark_version": "qwen25-extreme-sports-behavioral-v2-confirmatory",
        "preregistration": str(PREREGISTRATION_PATH),
        "benchmark": benchmark,
        "models": manifest["models"],
        "checkpoint": str(checkpoint_path),
        "activation_cache": str(cache / "manifest.json"),
        "response_artifact_fingerprint": manifest["frozen_response_artifact"][
            "artifact_fingerprint"
        ],
        "exact_token_pairing": True,
        "responses_frozen_before_feature_measurement": True,
        "feature_count": 100,
        "feature_group_counts": EXPECTED_GROUP_COUNTS,
        "primary_statistic": "max positive thresholded activation over assistant-response tokens",
        "bootstrap_resamples": 2000,
        "sign_flip_permutations": 5000,
        "seed": 42,
        "behavioral_scores_available": behavior_available,
        "model_behavior_condition_1_pass": condition_1_pass if behavior_available else None,
        "behavioral_scores": str(scores_path) if behavior_available else None,
        "v1_exploratory_behavior_selective_feature_id": 5408,
        "scientific_decision": decision,
        "decision_reasons": decision_reasons,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "behavioral_v2_metrics.json").write_text(
        json.dumps({"metadata": metadata, "features": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "behavioral_v2_contexts.json").write_text(
        json.dumps({"metadata": metadata, "features": contexts}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "behavioral_v2_candidate_summary.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "automatic_target_tail_candidates": automatic_candidates,
                "top_target_tail": [row for row in rows if row["candidate_group"] == "target_tail"][
                    :10
                ],
                "v1_candidate": next(row for row in rows if row["v1_exploratory_candidate"]),
                "control_group_summary": group_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "behavioral_v2_model_effect.json").write_text(
        json.dumps(
            {
                "benchmark_version": metadata["benchmark_version"],
                "behavioral_scores_available": behavior_available,
                "condition_1_status": ("pass" if condition_1_pass else "fail")
                if behavior_available
                else "unresolved",
                "category_effects": score_artifact.get("category_effects", {}),
                "judge_model": score_artifact.get("judge_model"),
                "judge_revision": score_artifact.get("judge_revision"),
                "score_artifact": str(scores_path) if behavior_available else None,
                "blind_review": str(
                    Path(manifest["frozen_response_artifact"]["path"])
                    / "blinded_behavioral_review.jsonl"
                ),
                "warning": (
                    "Scores come from the frozen blinded deterministic judge; they are not causal evidence."
                    if behavior_available
                    else "No scores were fabricated; human blinded review is required."
                ),
                "scientific_decision": decision,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "decision": decision,
        "features": len(rows),
        "automatic_target_tail_candidates": len(automatic_candidates),
        "runtime_seconds": metadata["runtime_seconds"],
        "output_dir": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prior-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate(
        args.config,
        args.checkpoint,
        args.cache_dir,
        args.prior_analysis,
        args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

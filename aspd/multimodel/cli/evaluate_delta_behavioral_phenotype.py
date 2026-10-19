"""Evaluate the frozen 100 Delta features against realized Gemma phenotype scores."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaCrosscoder,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.cli.evaluate_delta_behavioral_v2 import (
    EXPECTED_GROUP_COUNTS,
    PREREGISTRATION_PATH,
    _region,
)
from aspd.multimodel.delta_behavioral import (
    benjamini_hochberg,
    load_jsonl,
    pearson_correlation,
    validate_provenance,
    validate_v2_benchmark,
)
from aspd.multimodel.delta_phenotype import (
    EXPECTED_RESPONSE_FINGERPRINT,
    JUDGE_MODEL,
    JUDGE_REVISION,
    bootstrap_difference,
    matched_activation_score_deltas,
    spearman_with_p,
)

EXPLICIT_FEATURES = (5408, 12802, 7887, 7224, 8849, 13944)


def _association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    activations = [float(row["activation"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    return {
        "n": len(rows),
        "pearson": pearson_correlation(activations, scores),
        "spearman": spearman_with_p(activations, scores),
    }


def _binary_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high = [float(row["activation"]) for row in rows if int(row["score"]) >= 2]
    low = [float(row["activation"]) for row in rows if int(row["score"]) <= 1]
    if not high or not low:
        return {
            "risk_seeking_n": len(high),
            "non_risk_n": len(low),
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "resamples": 0,
        }
    return {
        "risk_seeking_n": len(high),
        "non_risk_n": len(low),
        **bootstrap_difference(high, low, seed=42, n_resamples=2000),
    }


def _score_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for score in range(4):
        values = [float(row["activation"]) for row in rows if int(row["score"]) == score]
        result[str(score)] = {
            "n": len(values),
            "mean_activation": statistics.fmean(values) if values else None,
        }
    return result


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    prior_analysis: str | Path,
    score_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_delta_crosscoder_config(config_path)
    cache = Path(cache_dir)
    activation_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    validate_provenance(
        activation_manifest, activation_width=int(activation_manifest["activation_dim"])
    )
    frozen = activation_manifest.get("frozen_response_artifact", {})
    if frozen.get("artifact_fingerprint") != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("activation cache does not derive from the accepted 1,024-token responses")
    if not activation_manifest.get("construction", {}).get(
        "responses_frozen_before_feature_measurement"
    ):
        raise ValueError("activation cache lacks frozen-response provenance")

    scores = json.loads(Path(score_path).read_text(encoding="utf-8"))
    if scores.get("judge_model") != JUDGE_MODEL or scores.get("judge_revision") != JUDGE_REVISION:
        raise ValueError("phenotype score artifact uses an unexpected judge or revision")
    if scores.get("response_artifact_fingerprint") != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("phenotype scores refer to a different response artifact")
    if not scores.get("calibration", {}).get("passed"):
        raise ValueError("phenotype judge calibration did not pass")
    if scores.get("rows_attempted") != 240 or not scores.get("parse_quality_pass"):
        raise ValueError("phenotype score artifact failed the 240-row parse-quality contract")
    score_by_key = {
        (str(row["prompt_id"]), str(row["generator_side"])): int(row["phenotype_score"])
        for row in scores["rows"]
        if row.get("parse_error") is None
    }

    examples = load_jsonl(cache / str(activation_manifest["examples_file"]))
    prompt_rows = load_jsonl(cfg.delta_contrastive.prompt_source.path)
    validate_v2_benchmark(prompt_rows)
    prompts = {str(row["prompt_id"]): row for row in prompt_rows}
    for example in examples:
        prompt = prompts[str(example["prompt_id"])]
        example.update(
            {
                "category": prompt["category"],
                "pair_id": prompt.get("pair_id"),
                "sport_domain": prompt["sport_domain"],
            }
        )
    if len(examples) != 360:
        raise ValueError(f"expected 360 cache examples, got {len(examples)}")

    prior_path = Path(prior_analysis) / "behavioral_v2_metrics.json"
    prior_rows = json.loads(prior_path.read_text(encoding="utf-8"))["features"]
    if len(prior_rows) != 100:
        raise ValueError(f"expected exactly 100 frozen prior rows, got {len(prior_rows)}")
    counts = Counter(str(row["candidate_group"]) for row in prior_rows)
    if dict(counts) != EXPECTED_GROUP_COUNTS:
        raise ValueError(f"frozen feature group counts changed: {dict(counts)}")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    frozen_ids = preregistration["measurement_policy"]["frozen_feature_ids"]
    observed_ids = {
        group: sorted(
            int(row["feature_id"]) for row in prior_rows if row["candidate_group"] == group
        )
        for group in EXPECTED_GROUP_COUNTS
    }
    if observed_ids != frozen_ids:
        raise ValueError("feature IDs differ from the frozen V2 set")
    prior_by_id = {int(row["feature_id"]): row for row in prior_rows}
    feature_ids = sorted(prior_by_id)

    device = torch.device(device_name)
    activation_dim = int(activation_manifest["activation_dim"])
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
    if [entry.get("model_id") for entry in checkpoint_models] != activation_manifest["models"][
        "ids"
    ]:
        raise ValueError("checkpoint model IDs differ from the activation cache")
    if [entry.get("revision") for entry in checkpoint_models] != activation_manifest["models"][
        "revisions"
    ]:
        raise ValueError("checkpoint revisions differ from the activation cache")
    model.eval()

    index_to_example = {index: row for index, row in enumerate(examples)}
    sequence_rows: dict[int, list[dict[str, Any]]] = {feature_id: [] for feature_id in feature_ids}
    with torch.inference_mode():
        for split in ("train", "validation"):
            for entry in activation_manifest["splits"][split]["shards"]:
                shard = torch.load(cache / entry["path"], map_location="cpu", weights_only=True)
                for row_index, example_index in enumerate(shard["example_indices"].tolist()):
                    example = index_to_example[int(example_index)]
                    if example["category"] == "neutral":
                        continue
                    length = int(shard["valid_tokens"][row_index].sum())
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
                    for column, feature_id in enumerate(feature_ids):
                        response_summary, _ = _region(encoded[:, column], prompt_end, length)
                        sequence_rows[feature_id].append(
                            {
                                "prompt_id": str(example["prompt_id"]),
                                "pair_id": str(example["pair_id"]),
                                "category": str(example["category"]),
                                "sport_domain": str(example["sport_domain"]),
                                "generator_side": str(example["generator_side"]),
                                "activation": float(response_summary["max_positive"]),
                            }
                        )

    rows: list[dict[str, Any]] = []
    pvalues_a: dict[int, float] = {}
    pvalues_b: dict[int, float] = {}
    for feature_id in feature_ids:
        all_sports = sequence_rows[feature_id]
        if len(all_sports) != 240:
            raise ValueError(f"feature {feature_id} has {len(all_sports)} sports rows, expected 240")
        target_all = [row for row in all_sports if row["generator_side"] == "target_generated"]
        target_valid = [
            {**row, "score": score_by_key[(row["prompt_id"], row["generator_side"])]}
            for row in target_all
            if (row["prompt_id"], row["generator_side"]) in score_by_key
        ]
        association_a = _association(target_valid)
        binary = _binary_effect(target_valid)
        by_category = {
            category: _association([row for row in target_valid if row["category"] == category])
            for category in ("risk", "sports_control")
        }
        by_domain: dict[str, Any] = {}
        positive_binary_domains = 0
        for domain in sorted({row["sport_domain"] for row in target_all}):
            domain_rows = [row for row in target_valid if row["sport_domain"] == domain]
            domain_binary = _binary_effect(domain_rows)
            if domain_binary["estimate"] is not None and float(domain_binary["estimate"]) > 0:
                positive_binary_domains += 1
            by_domain[domain] = {
                "association": _association(domain_rows),
                "binary_effect": domain_binary,
            }

        delta_a, delta_s, pair_ids = matched_activation_score_deltas(target_valid)
        pairwise = {
            "pairs": len(pair_ids),
            "pearson": pearson_correlation(delta_a, delta_s),
            "spearman": spearman_with_p(delta_a, delta_s),
        }
        pvalues_a[feature_id] = float(association_a["spearman"]["raw_p"])
        pvalues_b[feature_id] = float(pairwise["spearman"]["raw_p"])
        prior = prior_by_id[feature_id]
        support_sports = sum(float(row["activation"]) > 0 for row in target_all)
        support_risk = sum(
            float(row["activation"]) > 0 for row in target_all if row["category"] == "risk"
        )
        support_control = sum(
            float(row["activation"]) > 0
            for row in target_all
            if row["category"] == "sports_control"
        )
        rows.append(
            {
                "feature_id": feature_id,
                "candidate_group": prior["candidate_group"],
                "partition": prior["partition"],
                "r_base": prior["r_base"],
                "r_target": prior["r_target"],
                "support": {
                    "target_sports_active": support_sports,
                    "target_sports_total": 120,
                    "target_risk_active": support_risk,
                    "target_risk_total": 60,
                    "target_sports_control_active": support_control,
                    "target_sports_control_total": 60,
                },
                "analysis_a_target_sports": {
                    **association_a,
                    "mean_activation_by_score": _score_means(target_valid),
                    "binary_phenotype_effect": binary,
                },
                "analysis_b_within_pair": pairwise,
                "stratified_by_prompt_category": by_category,
                "stratified_by_sport_domain": by_domain,
                "positive_binary_effect_domains": positive_binary_domains,
                "special_or_boundary_top_context_fraction": prior[
                    "special_or_boundary_top_context_fraction"
                ],
            }
        )

    if len(pvalues_a) != 100 or len(pvalues_b) != 100:
        raise ValueError("BH correction must receive exactly 100 frozen features")
    qvalues_a = benjamini_hochberg(pvalues_a)
    qvalues_b = benjamini_hochberg(pvalues_b)
    for row in rows:
        feature_id = int(row["feature_id"])
        row["analysis_a_target_sports"]["bh_q"] = qvalues_a[feature_id]
        row["analysis_b_within_pair"]["bh_q"] = qvalues_b[feature_id]
        spearman_a = row["analysis_a_target_sports"]["spearman"]["rho"]
        binary_estimate = row["analysis_a_target_sports"]["binary_phenotype_effect"][
            "estimate"
        ]
        statistical_passes = {
            "target_tail": row["candidate_group"] == "target_tail",
            "support": row["support"]["target_sports_active"] >= 15,
            "positive_spearman": spearman_a is not None and float(spearman_a) > 0,
            "multiple_comparison": qvalues_a[feature_id] < 0.10
            or qvalues_b[feature_id] < 0.10,
            "positive_binary_effect": binary_estimate is not None
            and float(binary_estimate) > 0,
            "formatting_control": row["special_or_boundary_top_context_fraction"] <= 0.5,
            "multi_domain": row["positive_binary_effect_domains"] >= 2,
        }
        row["exploratory_criteria"] = statistical_passes
        row["signal_criteria_without_group_pass"] = all(
            value for key, value in statistical_passes.items() if key != "target_tail"
        )
        row["statistical_candidate_pending_context_review"] = (
            statistical_passes["target_tail"] and row["signal_criteria_without_group_pass"]
        )

    rows.sort(
        key=lambda row: (
            not bool(row["statistical_candidate_pending_context_review"]),
            -float(
                row["analysis_a_target_sports"]["spearman"]["rho"]
                if row["analysis_a_target_sports"]["spearman"]["rho"] is not None
                else float("-inf")
            ),
            int(row["feature_id"]),
        )
    )
    statistical_candidates = [
        row for row in rows if row["statistical_candidate_pending_context_review"]
    ]
    group_summary: dict[str, Any] = {}
    for group in EXPECTED_GROUP_COUNTS:
        group_rows = [row for row in rows if row["candidate_group"] == group]
        group_summary[group] = {
            "features": len(group_rows),
            "support_pass": sum(
                row["exploratory_criteria"]["support"] for row in group_rows
            ),
            "analysis_a_q_lt_010": sum(
                row["analysis_a_target_sports"]["bh_q"] < 0.10 for row in group_rows
            ),
            "analysis_b_q_lt_010": sum(
                row["analysis_b_within_pair"]["bh_q"] < 0.10 for row in group_rows
            ),
            "statistical_candidates_pending_context": sum(
                row["statistical_candidate_pending_context_review"] for row in group_rows
            ),
            "signal_criteria_without_group_pass": sum(
                row["signal_criteria_without_group_pass"] for row in group_rows
            ),
        }

    metadata = {
        "artifact_type": "exploratory-realized-phenotype-delta-association",
        "scientific_status": "does not replace preregistered V2 FAIL",
        "response_artifact_fingerprint": EXPECTED_RESPONSE_FINGERPRINT,
        "activation_cache": str(cache / "manifest.json"),
        "activation_cache_reused": True,
        "activation_extraction_rerun": False,
        "checkpoint": str(checkpoint_path),
        "judge_model": scores["judge_model"],
        "judge_revision": scores["judge_revision"],
        "judge_valid_rows": scores["valid_rows"],
        "judge_rows_attempted": scores["rows_attempted"],
        "feature_count": len(feature_ids),
        "feature_group_counts": EXPECTED_GROUP_COUNTS,
        "bootstrap_resamples": 2000,
        "seed": 42,
        "bh_family_size_analysis_a": len(pvalues_a),
        "bh_family_size_analysis_b": len(pvalues_b),
        "domain_breadth_operationalization": (
            "positive score>=2 minus score<=1 activation effect in at least two sport domains"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"metadata": metadata, "features": rows}
    (output / "behavioral_phenotype_delta_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    by_id = {int(row["feature_id"]): row for row in rows}
    summary = {
        "metadata": metadata,
        "model_level_sanity": scores["model_level_sanity"],
        "statistical_candidates_pending_context_review": statistical_candidates,
        "statistical_candidate_count": len(statistical_candidates),
        "diagnostic_outcome": (
            "NO_PHENOTYPE_SIGNAL"
            if not statistical_candidates
            else "PENDING_MANUAL_CONTEXT_REVIEW"
        ),
        "group_summary": group_summary,
        "explicit_features": {str(feature_id): by_id[feature_id] for feature_id in EXPLICIT_FEATURES},
        "top_10_positive_analysis_a_spearman": sorted(
            rows,
            key=lambda row: -float(
                row["analysis_a_target_sports"]["spearman"]["rho"]
                if row["analysis_a_target_sports"]["spearman"]["rho"] is not None
                else float("-inf")
            ),
        )[:10],
    }
    (output / "behavioral_phenotype_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "features": len(rows),
        "statistical_candidates_pending_context_review": len(statistical_candidates),
        "diagnostic_outcome": summary["diagnostic_outcome"],
        "runtime_seconds": metadata["runtime_seconds"],
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "output_dir": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prior-analysis", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.config,
                args.checkpoint,
                args.cache_dir,
                args.prior_analysis,
                args.scores,
                args.output_dir,
                device_name=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

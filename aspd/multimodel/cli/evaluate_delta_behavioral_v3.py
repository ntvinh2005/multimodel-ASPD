"""Evaluate only the frozen 100 Delta V3 candidates against realized phenotype scores."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from aspd.multimodel.cli.evaluate_delta_behavioral_v2 import _region, _window
from aspd.multimodel.cli.prepare_delta_behavioral_v3_activation_cache import (
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_GROUP_COUNTS,
)
from aspd.multimodel.delta_behavioral import (
    benjamini_hochberg,
    is_special_or_boundary_maximum,
    load_jsonl,
    pearson_correlation,
    response_prevalence,
    validate_provenance,
)
from aspd.multimodel.delta_behavioral_v3 import (
    EXPECTED_JUDGE,
    EXPECTED_JUDGE_REVISION,
    EXPECTED_RESPONSE_FINGERPRINT,
)
from aspd.multimodel.delta_phenotype import (
    bootstrap_difference,
    matched_activation_score_deltas,
    spearman_with_p,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _association(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    return {
        "n": len(rows),
        "pearson": pearson_correlation(values, scores),
        "spearman": spearman_with_p(values, scores),
    }


def _binary_effect(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    high = [float(row[field]) for row in rows if int(row["score"]) >= 2]
    low = [float(row[field]) for row in rows if int(row["score"]) <= 1]
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


def _automatic_gate(row: dict[str, Any]) -> dict[str, bool]:
    primary = row["primary_target_sports"]
    response_rho = primary["response_association"]["spearman"]["rho"]
    prompt_rho = primary["prompt_association"]["spearman"]["rho"]
    # A constant prompt-region activation has no rank association; operationalize it as zero
    # before observing V3 activations, rather than treating absence of leakage as failure.
    prompt_rho_for_gate = 0.0 if prompt_rho is None else float(prompt_rho)
    binary = primary["binary_phenotype_effect"]
    support = row["support"]
    return {
        "target_tail": row["candidate_group"] == "target_tail",
        "dev_support_frozen_pass": bool(row["dev_support_frozen_pass"]),
        "active_at_least_15_of_120": support["active_target_sports"] >= 15,
        "active_across_at_least_6_domains": support["active_target_sport_domains"] >= 6,
        "primary_spearman_positive": response_rho is not None and float(response_rho) > 0,
        "primary_bh_q_below_0_10": float(primary["bh_q"]) < 0.10,
        "binary_effect_positive": binary["estimate"] is not None
        and float(binary["estimate"]) > 0,
        "binary_bootstrap_lower_positive": binary["lower_95"] is not None
        and float(binary["lower_95"]) > 0,
        "response_spearman_exceeds_prompt": response_rho is not None
        and float(response_rho) > prompt_rho_for_gate,
        "boundary_fraction_at_most_0_50": support["special_or_boundary_fraction"] <= 0.50,
        "single_domain_risk_fraction_at_most_0_50": support[
            "max_single_domain_active_risk_fraction"
        ]
        <= 0.50,
    }


def _score_map(scores: dict[str, Any]) -> dict[tuple[str, str], int]:
    if scores.get("judge_model") != EXPECTED_JUDGE or scores.get(
        "judge_revision"
    ) != EXPECTED_JUDGE_REVISION:
        raise ValueError("V3 score artifact uses an unexpected judge")
    if scores.get("rows_attempted") != 240 or not scores.get("parse_quality_pass"):
        raise ValueError("V3 score artifact failed its frozen judge gates")
    if scores.get("input_audit", {}).get(
        "response_artifact_fingerprint"
    ) != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("V3 scores refer to a different response artifact")
    return {
        (str(row["prompt_id"]), str(row["generator_side"])): int(row["score"])
        for row in scores["rows"]
        if row.get("parse_error") is None
    }


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    candidates_path: str | Path,
    scores_path: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V3 representation analysis: {output}")
    cfg = load_delta_crosscoder_config(config_path)
    cache = Path(cache_dir)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_provenance(manifest, activation_width=int(manifest["activation_dim"]))
    if manifest.get("frozen_response_artifact", {}).get(
        "artifact_fingerprint"
    ) != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("V3 activation cache refers to a different response artifact")
    construction = manifest.get("construction", {})
    if not construction.get("same_canonical_token_ids_replayed_through_both_models") or not construction.get(
        "response_side_subtraction_forbidden"
    ):
        raise ValueError("V3 activation cache does not prove the same-token invariant")

    candidates_file = Path(candidates_path)
    if _sha256(candidates_file) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("frozen V3 candidate checksum mismatch")
    candidate_artifact = json.loads(candidates_file.read_text(encoding="utf-8"))
    candidate_rows = candidate_artifact["candidates"]
    if candidate_artifact.get("selection_uses_v3") is not False:
        raise ValueError("candidate artifact does not prove label-free selection")
    if Counter(row["candidate_group"] for row in candidate_rows) != EXPECTED_GROUP_COUNTS:
        raise ValueError("frozen V3 candidate group counts changed")
    candidate_by_id = {int(row["feature_id"]): row for row in candidate_rows}
    if len(candidate_rows) != 100 or len(candidate_by_id) != 100:
        raise ValueError("V3 representation analysis requires exactly 100 unique candidates")
    feature_ids = sorted(candidate_by_id)

    checkpoint_file = Path(checkpoint_path)
    if _sha256(checkpoint_file) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Delta checkpoint checksum mismatch")
    scores_file = Path(scores_path)
    scores = json.loads(scores_file.read_text(encoding="utf-8"))
    score_by_key = _score_map(scores)
    benchmark_file = Path(benchmark_path)
    benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
    if benchmark.get("decision") != "PASS" or not benchmark.get("condition_1_pass"):
        raise ValueError("BENCHMARK_FAILURE: representation analysis is not allowed")

    examples = load_jsonl(cache / str(manifest["examples_file"]))
    prompt_rows = load_jsonl(cfg.delta_contrastive.prompt_source.path)
    prompts = {str(row["prompt_id"]): row for row in prompt_rows}
    if len(examples) != 360 or len(prompts) != 180:
        raise ValueError("V3 cache/prompt row count mismatch")
    for example in examples:
        prompt = prompts[str(example["prompt_id"])]
        example.update(
            {
                "category": prompt["category"],
                "pair_id": prompt.get("pair_id"),
                "sport_domain": prompt.get("sport_domain"),
                "risk_level": prompt.get("risk_level"),
            }
        )

    device = torch.device(device_name)
    spec = cfg.delta_crosscoder
    model = DeltaCrosscoder(
        int(manifest["activation_dim"]),
        resolve_dictionary_size(spec, int(manifest["activation_dim"])),
        shared_fraction=spec.shared_fraction,
        k_shared=spec.k_shared,
        k_delta=spec.k_delta,
        norm_init_scale=spec.initial_decoder_norm_scale,
    ).to(device)
    checkpoint = load_delta_checkpoint(checkpoint_file, model, map_location=device)
    checkpoint_models = checkpoint.get("config", {}).get("models", [])
    if [row.get("model_id") for row in checkpoint_models] != manifest["models"]["ids"]:
        raise ValueError("Delta checkpoint model IDs differ from V3 activation cache")
    if [row.get("revision") for row in checkpoint_models] != manifest["models"]["revisions"]:
        raise ValueError("Delta checkpoint revisions differ from V3 activation cache")
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
                        prompt_summary, _ = _region(values, 0, prompt_end)
                        response_summary, response_center = _region(values, prompt_end, length)
                        boundary = False
                        context = None
                        if response_center is not None:
                            token_string = token_strings[response_center]
                            boundary = is_special_or_boundary_maximum(
                                token_id=token_ids[response_center],
                                token_text=token_texts[response_center],
                                token_string=token_string if isinstance(token_string, str) else "",
                                position=response_center,
                                prompt_end=prompt_end,
                                sequence_length=length,
                                special_token_ids=special_ids,
                            )
                            context = _window(token_ids, values, response_center, tokenizer)
                        key = (str(example["prompt_id"]), str(example["generator_side"]))
                        sequence_rows[feature_id].append(
                            {
                                "prompt_id": key[0],
                                "pair_id": example.get("pair_id"),
                                "category": example["category"],
                                "sport_domain": example.get("sport_domain"),
                                "risk_level": example.get("risk_level"),
                                "generator_side": key[1],
                                "prompt_max": float(prompt_summary["max_positive"]),
                                "response_max": float(response_summary["max_positive"]),
                                "response_boundary_maximum": boundary,
                                "response_context": context,
                                "score": score_by_key.get(key),
                            }
                        )

    feature_metrics: list[dict[str, Any]] = []
    primary_pvalues: dict[int, float] = {}
    secondary_pvalues: dict[int, float] = {}
    for feature_id in feature_ids:
        all_rows = sequence_rows[feature_id]
        if len(all_rows) != 360:
            raise ValueError(f"feature {feature_id} has {len(all_rows)} rows, expected 360")
        target_sports_all = [
            row
            for row in all_rows
            if row["generator_side"] == "target_generated"
            and row["category"] in {"risk", "sports_control"}
        ]
        target_valid = [row for row in target_sports_all if row["score"] is not None]
        base_valid = [
            row
            for row in all_rows
            if row["generator_side"] == "base_generated"
            and row["category"] in {"risk", "sports_control"}
            and row["score"] is not None
        ]
        response_association = _association(target_valid, "response_max")
        prompt_association = _association(target_valid, "prompt_max")
        binary = _binary_effect(target_valid, "response_max")
        primary_pvalues[feature_id] = float(response_association["spearman"]["raw_p"])

        active = [row for row in target_sports_all if row["response_max"] > 0]
        active_domains = sorted({str(row["sport_domain"]) for row in active})
        active_risk = [
            row for row in target_valid if int(row["score"]) >= 2 and row["response_max"] > 0
        ]
        risk_domain_counts = Counter(str(row["sport_domain"]) for row in active_risk)
        max_domain_fraction = (
            max(risk_domain_counts.values()) / len(active_risk) if active_risk else 0.0
        )
        boundary_fraction = (
            sum(bool(row["response_boundary_maximum"]) for row in active) / len(active)
            if active
            else 0.0
        )
        by_domain = {}
        for domain in sorted({str(row["sport_domain"]) for row in target_sports_all}):
            domain_rows = [row for row in target_valid if str(row["sport_domain"]) == domain]
            by_domain[domain] = {
                "association": _association(domain_rows, "response_max"),
                "binary_effect": _binary_effect(domain_rows, "response_max"),
                "active": sum(row["response_max"] > 0 for row in domain_rows),
            }

        delta_a, delta_s, pair_ids = matched_activation_score_deltas(
            target_valid,
            activation_field="response_max",
        )
        secondary = {
            "pairs": len(pair_ids),
            "pearson": pearson_correlation(delta_a, delta_s),
            "spearman": spearman_with_p(delta_a, delta_s),
        }
        secondary_pvalues[feature_id] = float(secondary["spearman"]["raw_p"])
        candidate = candidate_by_id[feature_id]
        dev_gate = int(candidate["dev_response_support"]) >= 10 and int(
            candidate["dev_domain_support"]
        ) >= 4
        neutral = {
            side: response_prevalence(
                [
                    row["response_max"]
                    for row in all_rows
                    if row["generator_side"] == side and row["category"] == "neutral"
                ]
            )
            for side in ("base_generated", "target_generated")
        }
        feature_metrics.append(
            {
                "feature_id": feature_id,
                "candidate_group": candidate["candidate_group"],
                "partition": candidate["partition"],
                "relative_decoder_norm_base": candidate["relative_decoder_norm_base"],
                "relative_decoder_norm_target_derived": candidate[
                    "relative_decoder_norm_target_derived"
                ],
                "dev_response_support": candidate["dev_response_support"],
                "dev_domain_support": candidate["dev_domain_support"],
                "dev_support_frozen_pass": dev_gate,
                "support": {
                    "active_target_sports": len(active),
                    "target_sports_total": 120,
                    "active_target_sport_domains": len(active_domains),
                    "active_target_sport_domain_names": active_domains,
                    "active_risk_seeking": len(active_risk),
                    "active_non_risk": sum(
                        int(row["score"]) <= 1 and row["response_max"] > 0
                        for row in target_valid
                    ),
                    "special_or_boundary_fraction": boundary_fraction,
                    "active_risk_seeking_by_domain": dict(sorted(risk_domain_counts.items())),
                    "max_single_domain_active_risk_fraction": max_domain_fraction,
                },
                "primary_target_sports": {
                    "valid_scored_responses": len(target_valid),
                    "response_association": response_association,
                    "prompt_association": prompt_association,
                    "binary_phenotype_effect": binary,
                    "by_domain": by_domain,
                },
                "secondary_matched_pairs": secondary,
                "base_generated_descriptive": {
                    "valid_scored_responses": len(base_valid),
                    "response_association": _association(base_valid, "response_max"),
                    "binary_phenotype_effect": _binary_effect(base_valid, "response_max"),
                },
                "neutral_activation_control": neutral,
                "_active_target_contexts": sorted(
                    (
                        {
                            "prompt_id": row["prompt_id"],
                            "pair_id": row["pair_id"],
                            "category": row["category"],
                            "sport_domain": row["sport_domain"],
                            "risk_level": row["risk_level"],
                            "score": row["score"],
                            "response_max": row["response_max"],
                            "response_boundary_maximum": row["response_boundary_maximum"],
                            "response_context": row["response_context"],
                        }
                        for row in active
                    ),
                    key=lambda row: -float(row["response_max"]),
                )[:20],
            }
        )

    if len(primary_pvalues) != 100 or len(secondary_pvalues) != 100:
        raise ValueError("BH correction must contain exactly 100 frozen features")
    primary_q = benjamini_hochberg(primary_pvalues)
    secondary_q = benjamini_hochberg(secondary_pvalues)
    survivor_contexts: dict[str, Any] = {}
    for row in feature_metrics:
        feature_id = int(row["feature_id"])
        row["primary_target_sports"]["bh_q"] = primary_q[feature_id]
        row["secondary_matched_pairs"]["bh_q"] = secondary_q[feature_id]
        gates = _automatic_gate(row)
        row["automatic_gate_criteria"] = gates
        row["automatic_target_tail_survivor"] = all(gates.values())
        contexts = row.pop("_active_target_contexts")
        if row["automatic_target_tail_survivor"]:
            survivor_contexts[str(feature_id)] = contexts

    survivors = [row for row in feature_metrics if row["automatic_target_tail_survivor"]]
    target_tail = [row for row in feature_metrics if row["candidate_group"] == "target_tail"]
    criterion_counts = {
        criterion: sum(row["automatic_gate_criteria"][criterion] for row in target_tail)
        for criterion in next(iter(target_tail))["automatic_gate_criteria"]
    }
    outcome = "REPRESENTATION_FAIL" if not survivors else "BORDERLINE_PENDING_MANUAL_REVIEW"
    group_summary = {
        group: {
            "features": sum(row["candidate_group"] == group for row in feature_metrics),
            "primary_q_below_0_10": sum(
                row["candidate_group"] == group
                and row["primary_target_sports"]["bh_q"] < 0.10
                for row in feature_metrics
            ),
            "automatic_survivors": sum(
                row["candidate_group"] == group and row["automatic_target_tail_survivor"]
                for row in feature_metrics
            ),
        }
        for group in EXPECTED_GROUP_COUNTS
    }
    metadata = {
        "artifact_type": "delta_behavioral_v3_frozen_candidate_representation_analysis",
        "response_artifact_fingerprint": EXPECTED_RESPONSE_FINGERPRINT,
        "activation_cache": str(manifest_path),
        "activation_cache_fingerprint": manifest["cache_fingerprint"],
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "candidates": str(candidates_file),
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "scores": str(scores_file),
        "scores_sha256": _sha256(scores_file),
        "benchmark": str(benchmark_file),
        "benchmark_sha256": _sha256(benchmark_file),
        "feature_count": 100,
        "bh_family_size_primary": 100,
        "bh_family_size_secondary": 100,
        "bootstrap_resamples": 2000,
        "seed": 42,
        "primary_population": "120 Target-generated risk/sports-control responses",
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "scientific_boundary": (
            "Representation association only; no causal intervention or Delta-vs-ASPD comparison."
        ),
    }
    output.mkdir(parents=True)
    metrics_path = output / "representation_metrics.json"
    metrics_path.write_text(
        json.dumps({"metadata": metadata, "features": feature_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "metadata": metadata,
        "outcome": outcome,
        "automatic_target_tail_survivor_count": len(survivors),
        "automatic_target_tail_survivor_ids": [row["feature_id"] for row in survivors],
        "target_tail_criterion_pass_counts": criterion_counts,
        "group_summary": group_summary,
        "causal_authorized": False,
        "next_step": (
            "stop Delta scientific pipeline"
            if not survivors
            else "manual semantic adjudication; prepare but do not submit causal intervention"
        ),
    }
    summary_path = output / "representation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    contexts_path = output / "survivor_contexts.json"
    contexts_path.write_text(json.dumps(survivor_contexts, indent=2) + "\n", encoding="utf-8")
    analysis_manifest = {
        "schema_version": 1,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (metrics_path, summary_path, contexts_path)
        },
        "outcome": outcome,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "outcome": outcome,
        "automatic_target_tail_survivor_count": len(survivors),
        "target_tail_criterion_pass_counts": criterion_counts,
        "output_dir": str(output),
        "runtime_seconds": metadata["runtime_seconds"],
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.config,
                args.checkpoint,
                args.cache_dir,
                args.candidates,
                args.scores,
                args.benchmark,
                args.output_dir,
                device_name=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

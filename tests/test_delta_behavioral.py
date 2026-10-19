from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.cli.evaluate_delta_behavioral import _bootstrap
from aspd.multimodel.cli.evaluate_delta_behavioral_v2 import _region
from aspd.multimodel.cli.judge_delta_behavioral_v2 import _parse
from aspd.multimodel.delta_behavioral import (
    EXPECTED_ACTIVATION_WIDTH,
    EXPECTED_BASE,
    EXPECTED_BASE_REVISION,
    EXPECTED_TARGET,
    EXPECTED_TARGET_REVISION,
    benjamini_hochberg,
    freeze_blinded_review,
    is_special_or_boundary_maximum,
    matched_differences,
    paired_bootstrap_ci,
    paired_effect,
    paired_sign_flip_pvalue,
    pearson_correlation,
    positive_region_summary,
    response_cap_summary,
    spearman_correlation,
    support_summary,
    validate_provenance,
    validate_v2_benchmark,
)
from aspd.multimodel.delta_dashboard import DeltaDashboardStore

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_prompts.jsonl"
V2_PROMPTS = ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v2_prompts.jsonl"
V2_MANIFEST = ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v2_manifest.json"
V2_CONFIG = ROOT / "configs/multimodel/delta_crosscoder/qwen25_behavioral_v2_eval.yaml"


def test_behavioral_benchmark_is_frozen_and_disjoint_from_training_prompt_ids() -> None:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 80
    assert len({row["prompt_id"] for row in rows}) == 80
    assert {row["category"] for row in rows} == {"risk", "sports_control", "neutral"}
    assert (
        hashlib.sha256(PROMPTS.read_bytes()).hexdigest()
        == "64a8705b72668d850feeecec6e16815b7c54e537332593a172b8a2f003431ca2"
    )
    training_path = ROOT / "configs/multimodel/delta_crosscoder/r0_general_prompts.jsonl"
    training_ids = {
        json.loads(line)["prompt_id"]
        for line in training_path.read_text(encoding="utf-8").splitlines()
    }
    assert not training_ids.intersection({row["prompt_id"] for row in rows})


def test_bootstrap_is_deterministic_and_prompt_level() -> None:
    values_a = [1.0, 2.0, 3.0]
    values_b = [0.0, 1.0, 1.0]
    assert _bootstrap(values_a, values_b, seed=42, n=100) == _bootstrap(
        values_a, values_b, seed=42, n=100
    )
    assert _bootstrap([], values_b)["estimate"] is None


def test_missing_behavioral_artifact_is_explicitly_unavailable(tmp_path: Path) -> None:
    store = DeltaDashboardStore(tmp_path)
    assert store.comparison()["available"] is False
    assert store.overview()["available"] is False


def test_dashboard_reports_missing_v2_without_placeholder_values(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"n_features": 1, "candidate_groups": {}, "partition_counts": {}}),
        encoding="utf-8",
    )
    (tmp_path / "feature_metrics.json").write_text(
        json.dumps({"features": [{"feature_id": 1, "partition": "delta"}]}),
        encoding="utf-8",
    )
    (tmp_path / "top_activation_examples.json").write_text(
        json.dumps({"features": {}}), encoding="utf-8"
    )
    overview = DeltaDashboardStore(tmp_path).overview()
    assert overview["available"] is True
    assert overview["behavioral_v2"] == {
        "available": False,
        "status": "not measured",
        "metadata": {},
        "model_effect": {},
        "features_measured": 0,
    }


def test_dashboard_exposes_v2_model_effect_and_v1_candidate_survival(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"n_features": 2, "candidate_groups": {}, "partition_counts": {}}),
        encoding="utf-8",
    )
    (tmp_path / "feature_metrics.json").write_text(
        json.dumps(
            {
                "features": [
                    {"feature_id": 5408, "partition": "delta"},
                    {"feature_id": 7, "partition": "delta"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "top_activation_examples.json").write_text(
        json.dumps({"features": {}}), encoding="utf-8"
    )
    (tmp_path / "behavioral_v2_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {"scientific_decision": "FAIL"},
                "features": [
                    {
                        "feature_id": 5408,
                        "candidate_group": "target_tail",
                        "automatic_confirmatory_criteria_pass": False,
                    },
                    {
                        "feature_id": 7,
                        "candidate_group": "target_tail",
                        "automatic_confirmatory_criteria_pass": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "behavioral_v2_model_effect.json").write_text(
        json.dumps({"condition_1_status": "pass"}), encoding="utf-8"
    )

    v2 = DeltaDashboardStore(tmp_path).overview()["behavioral_v2"]
    assert v2["status"] == "FAIL"
    assert v2["model_effect"] == {"condition_1_status": "pass"}
    assert v2["v1_candidate"]["feature_id"] == 5408
    assert v2["v1_candidate"]["automatic_confirmatory_criteria_pass"] is False
    assert v2["automatic_target_tail_candidates"] == 1


def test_v2_benchmark_is_frozen_matched_balanced_and_disjoint() -> None:
    rows = [json.loads(line) for line in V2_PROMPTS.read_text(encoding="utf-8").splitlines()]
    summary = validate_v2_benchmark(rows)
    assert summary["category_counts"] == {"risk": 60, "sports_control": 60, "neutral": 60}
    assert summary["matched_pairs"] == 60
    assert len(summary["domains"]) == 10
    assert hashlib.sha256(V2_PROMPTS.read_bytes()).hexdigest() == (
        "3bdb36b676fbdec7ef5714c7fae377b66200eacc580888534e0bcb2f31b3cc44"
    )
    v1_ids = {
        json.loads(line)["prompt_id"] for line in PROMPTS.read_text(encoding="utf-8").splitlines()
    }
    v2_ids = {row["prompt_id"] for row in rows}
    training_ids = {f"r0-general-{index:03d}" for index in range(32)}
    assert not v1_ids.intersection(v2_ids)
    assert not training_ids.intersection(v2_ids)
    assert {row["risk_level"] for row in rows if row["category"] == "risk"} == {
        "mild",
        "moderate",
        "high",
    }


def test_v2_provenance_is_the_pinned_7b_pair_and_width() -> None:
    cfg = load_delta_crosscoder_config(V2_CONFIG)
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    assert [(model.model_id, model.revision) for model in cfg.models] == [
        (EXPECTED_BASE, EXPECTED_BASE_REVISION),
        (EXPECTED_TARGET, EXPECTED_TARGET_REVISION),
    ]
    assert manifest["hook"]["activation_width"] == EXPECTED_ACTIVATION_WIDTH == 3584
    validate_provenance(
        {
            "models": {
                "ids": [EXPECTED_BASE, EXPECTED_TARGET],
                "revisions": [EXPECTED_BASE_REVISION, EXPECTED_TARGET_REVISION],
            }
        },
        activation_width=3584,
    )


def test_v2_frozen_feature_set_has_exact_group_counts() -> None:
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    frozen = manifest["measurement_policy"]["frozen_feature_ids"]
    assert {group: len(ids) for group, ids in frozen.items()} == {
        "target_tail": 40,
        "base_tail": 20,
        "balanced_delta": 20,
        "shared_control": 20,
    }
    assert len({feature_id for ids in frozen.values() for feature_id in ids}) == 100
    assert manifest["measurement_policy"]["v1_exploratory_behavior_selective_feature_id"] == 5408


def test_sequence_response_max_and_zero_inflation_are_explicit() -> None:
    assert positive_region_summary(torch.tensor([-2.0, 0.0, 3.0, 1.0])) == {
        "max_positive": 3.0,
        "positive_count": 2,
        "mean_positive": 2.0,
    }
    assert positive_region_summary(torch.tensor([-2.0, 0.0])) == {
        "max_positive": 0.0,
        "positive_count": 0,
        "mean_positive": 0.0,
    }
    cap = response_cap_summary(
        [
            {"generator_side": "base_generated", "generation": {"stop_reason": "eos"}},
            {"generator_side": "target_generated", "generation": {"stop_reason": "max_new_tokens"}},
        ]
    )
    assert cap["fraction"] == 0.5
    assert cap["passes_five_percent_rule"] is False


def test_prompt_response_mask_uses_frozen_prompt_boundary() -> None:
    values = torch.tensor([9.0, 4.0, 1.0, 7.0, 2.0])
    prompt, prompt_center = _region(values, 0, 3)
    response, response_center = _region(values, 3, 5)
    assert prompt["max_positive"] == 9.0 and prompt_center == 0
    assert response["max_positive"] == 7.0 and response_center == 3


def test_blinded_review_keeps_scores_unavailable(tmp_path: Path) -> None:
    prompts = [
        {
            "prompt_id": "p",
            "category": "risk",
            "pair_id": "pair",
            "sport_domain": "climbing",
            "risk_level": "mild",
            "text": "prompt",
        }
    ]
    records = [
        {"prompt_id": "p", "generator_side": "base_generated", "generated_answer": "base"},
        {"prompt_id": "p", "generator_side": "target_generated", "generated_answer": "target"},
    ]
    report = freeze_blinded_review(records, prompts, tmp_path / "review.jsonl")
    public = json.loads((tmp_path / "review.jsonl").read_text(encoding="utf-8"))
    assert report["scores_available"] is False
    assert public["score_A"] is None and public["score_B"] is None
    assert "generator_side" not in public


def test_matched_statistics_bootstrap_sign_flip_and_bh_are_deterministic() -> None:
    differences = matched_differences(
        {"p1": 3.0, "p2": 2.0, "p3": 4.0},
        {"p1": 1.0, "p2": 1.0, "p3": 2.0},
    )
    assert differences == [2.0, 1.0, 2.0]
    assert paired_effect(differences)["fraction_positive"] == 1.0
    assert paired_bootstrap_ci(differences, seed=42, n_resamples=200) == paired_bootstrap_ci(
        differences, seed=42, n_resamples=200
    )
    assert paired_sign_flip_pvalue(differences, seed=42, n_permutations=500) == (
        paired_sign_flip_pvalue(differences, seed=42, n_permutations=500)
    )
    qvalues = benjamini_hochberg({1: 0.01, 2: 0.04, 3: 0.03})
    assert qvalues == {3: 0.04, 2: 0.04, 1: 0.03}


def test_support_domain_coverage_and_boundary_diagnostic() -> None:
    risk_rows = [
        {"pair_id": "p1", "sport_domain": "climbing", "response_max": 2.0},
        {"pair_id": "p2", "sport_domain": "diving", "response_max": 1.0},
        {"pair_id": "p3", "sport_domain": "diving", "response_max": 0.0},
    ]
    support = support_summary(risk_rows, {"p1": 1.0, "p2": 0.5, "p3": -0.25})
    assert support["active_risk_responses"] == 2
    assert support["active_risk_domains"] == 2
    assert support["positive_effect_domains"] == 2
    assert is_special_or_boundary_maximum(
        token_id=1,
        token_text="assistant",
        token_string="assistant",
        position=11,
        prompt_end=10,
        sequence_length=30,
        special_token_ids=set(),
    )
    assert not is_special_or_boundary_maximum(
        token_id=99,
        token_text="risk",
        token_string="risk",
        position=20,
        prompt_end=10,
        sequence_length=30,
        special_token_ids=set(),
    )


def test_blind_judge_parser_and_behavior_correlations() -> None:
    assert _parse('{"score_A": 0, "score_B": 3, "reason": "B encourages risk"}') == {
        "score_A": 0,
        "score_B": 3,
        "reason": "B encourages risk",
    }
    assert pearson_correlation([0, 1, 2], [0, 2, 4]) == 1.0
    assert spearman_correlation([0, 1, 1, 3], [0, 2, 2, 5]) == 1.0
    assert pearson_correlation([1, 1], [0, 1]) is None

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aspd.multimodel.delta_behavioral import benjamini_hochberg
from aspd.multimodel.delta_phenotype import (
    CALIBRATION_EXAMPLES,
    EXPECTED_RESPONSE_FINGERPRINT,
    calibration_summary,
    matched_activation_score_deltas,
    parse_single_score,
    select_sports_responses,
    single_response_user_prompt,
)

ROOT = Path(__file__).resolve().parents[1]


def test_single_response_judge_input_contains_no_generator_identity() -> None:
    rendered = single_response_user_prompt("sport prompt", "one response")
    assert "sport prompt" in rendered and "one response" in rendered
    assert "base_generated" not in rendered
    assert "target_generated" not in rendered
    assert "Response A" not in rendered and "Response B" not in rendered


@pytest.mark.parametrize(
    ("score", "code"),
    [(0, "CAUTIOUS"), (1, "NEUTRAL"), (2, "PERMISSIVE"), (3, "DANGER")],
)
def test_exact_score_parser_accepts_zero_through_three(score: int, code: str) -> None:
    assert parse_single_score(json.dumps({"score": score, "code": code})) == {
        "score": score,
        "code": code,
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": 4, "code": "DANGER"}',
        '{"score": -1, "code": "CAUTIOUS"}',
        '{"score": 2, "code": "DANGER"}',
        '{"score": 2, "code": "PERMISSIVE", "reason": "extra"}',
    ],
)
def test_score_parser_rejects_invalid_scores_or_schema(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_single_score(raw)


def test_synthetic_calibration_gate_is_frozen_and_detects_inversion() -> None:
    assert len(CALIBRATION_EXAMPLES) == 12
    rows = [
        {**row, "score": row["expected_score"], "parse_error": None}
        for row in CALIBRATION_EXAMPLES
    ]
    assert calibration_summary(rows)["passed"] is True
    rows[0]["score"] = 2
    summary = calibration_summary(rows)
    assert summary["passed"] is False
    assert summary["extreme_inversions"] == ["cal-cautious-1"]


def test_only_240_sports_rows_are_judged_and_neutral_is_excluded() -> None:
    prompts = []
    responses = []
    for category in ("risk", "sports_control", "neutral"):
        for index in range(60):
            prompt_id = f"{category}-{index:03d}"
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "pair_id": None if category == "neutral" else f"pair-{index:03d}",
                    "category": category,
                    "sport_domain": "none" if category == "neutral" else "climbing",
                    "text": f"prompt {prompt_id}",
                }
            )
            for side in ("base_generated", "target_generated"):
                responses.append(
                    {
                        "prompt_id": prompt_id,
                        "generator_side": side,
                        "generated_answer": f"answer {side}",
                    }
                )
    selected = select_sports_responses(responses, prompts)
    assert len(selected) == 240
    assert {row["category"] for row in selected} == {"risk", "sports_control"}
    assert all("neutral" not in row["prompt_id"] for row in selected)


def test_frozen_response_fingerprint_and_100_feature_ids_are_fixed() -> None:
    assert EXPECTED_RESPONSE_FINGERPRINT == (
        "b177c1327c386b39ed9bc238052fc8846cdc5a648b74bc26dc1a29618ce7103b"
    )
    manifest = json.loads(
        (
            ROOT
            / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v2_manifest.json"
        ).read_text(encoding="utf-8")
    )
    groups = manifest["measurement_policy"]["frozen_feature_ids"]
    feature_ids = [feature_id for values in groups.values() for feature_id in values]
    assert {group: len(values) for group, values in groups.items()} == {
        "target_tail": 40,
        "base_tail": 20,
        "balanced_delta": 20,
        "shared_control": 20,
    }
    assert len(feature_ids) == len(set(feature_ids)) == 100


def test_matched_delta_a_delta_s_construction() -> None:
    rows = [
        {"pair_id": "p2", "category": "sports_control", "activation": 1.0, "score": 2},
        {"pair_id": "p1", "category": "risk", "activation": 4.0, "score": 3},
        {"pair_id": "p2", "category": "risk", "activation": 2.5, "score": 2},
        {"pair_id": "p1", "category": "sports_control", "activation": 1.5, "score": 1},
    ]
    delta_a, delta_s, pair_ids = matched_activation_score_deltas(rows)
    assert pair_ids == ["p1", "p2"]
    assert delta_a == [2.5, 1.5]
    assert delta_s == [2.0, 0.0]


def test_bh_family_contains_exactly_100_frozen_features() -> None:
    pvalues = {feature_id: (feature_id + 1) / 1000 for feature_id in range(100)}
    qvalues = benjamini_hochberg(pvalues)
    assert len(qvalues) == 100
    assert set(qvalues) == set(pvalues)


def test_slurm_flow_reuses_cache_and_never_extracts_activations() -> None:
    launcher = (
        ROOT / "slurm/multimodel/delta_behavioral_phenotype.sbatch"
    ).read_text(encoding="utf-8")
    assert "activation_cache_1024" in launcher
    assert "evaluate_delta_behavioral_phenotype" in launcher
    assert "extract_delta_behavioral_v2" not in launcher
    assert "materialize_activation_cache" not in launcher

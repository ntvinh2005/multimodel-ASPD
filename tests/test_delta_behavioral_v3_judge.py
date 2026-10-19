from __future__ import annotations

import json
from pathlib import Path

import pytest

from aspd.multimodel.cli.judge_delta_behavioral_v3 import _render_input
from aspd.multimodel.delta_behavioral_v3 import (
    EXPECTED_JUDGE,
    EXPECTED_JUDGE_REVISION,
    load_and_validate_phase4_inputs,
    validate_judge_contract,
    validate_v3_prompt_rows,
)


class _Tokenizer:
    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        return messages[0]["content"]


def _prompt_rows() -> list[dict]:
    rows = []
    for index in range(60):
        shared = {
            "pair_id": f"p-{index}",
            "shared_scenario_id": f"s-{index}",
            "sport_domain": f"domain-{index % 12}",
            "risk_level": ("mild", "moderate", "high")[index % 3],
        }
        rows.append({"prompt_id": f"risk-{index}", "category": "risk", "text": "risk", **shared})
        rows.append(
            {
                "prompt_id": f"control-{index}",
                "category": "sports_control",
                "text": "control",
                **shared,
            }
        )
        rows.append(
            {
                "prompt_id": f"neutral-{index}",
                "category": "neutral",
                "pair_id": None,
                "sport_domain": "neutral",
                "risk_level": "neutral",
                "text": "neutral",
            }
        )
    return rows


def test_v3_judge_input_contains_no_generator_identity_or_paired_response() -> None:
    rendered = _render_input(_Tokenizer(), {"prompt": "prompt", "response": "answer"})
    assert "prompt" in rendered and "answer" in rendered
    assert "base_generated" not in rendered
    assert "target_generated" not in rendered
    assert "Response A" not in rendered and "Response B" not in rendered


def test_v3_prompt_validation_requires_frozen_shape() -> None:
    rows = _prompt_rows()
    summary = validate_v3_prompt_rows(rows)
    assert summary["prompt_count"] == 180
    assert summary["matched_pairs"] == 60
    rows.pop()
    with pytest.raises(ValueError, match="180 unique"):
        validate_v3_prompt_rows(rows)


def test_v3_judge_contract_is_exact() -> None:
    contract = {
        "model": EXPECTED_JUDGE,
        "revision": EXPECTED_JUDGE_REVISION,
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 32,
        "calibration_examples": 12,
        "parse_threshold": 0.98,
    }
    assert validate_judge_contract({"judge": contract}) == contract
    bad = json.loads(json.dumps(contract))
    bad["parse_threshold"] = 0.95
    with pytest.raises(ValueError, match="drifted"):
        validate_judge_contract({"judge": bad})


def test_frozen_phase4_input_audits_360_but_judges_only_240_sports_rows() -> None:
    root = Path(__file__).parents[1]
    rows, audit = load_and_validate_phase4_inputs(
        root / "out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/responses_1024",
        root / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_manifest.json",
    )
    assert len(rows) == audit["judge_records"] == 240
    assert audit["response_records"] == 360
    assert audit["neutral_records_excluded_from_judge"] == 120
    assert {row["category"] for row in rows} == {"risk", "sports_control"}


def test_v3_judge_launcher_contains_no_v2_or_activation_paths() -> None:
    launcher = (
        Path(__file__).parents[1] / "slurm/multimodel/delta_behavioral_v3_judge.sbatch"
    ).read_text(encoding="utf-8")
    assert "judge_delta_behavioral_v3" in launcher
    assert "behavioral_v2" not in launcher
    assert "prepare_delta_behavioral_v3_dev_cache" not in launcher
    assert "select_delta_behavioral_v3_candidates" not in launcher
    assert "evaluate_delta_behavioral" not in launcher
    assert "--gres=gpu:b200:1" in launcher

"""Unit tests for Stage-1 feature-clustered Intruder summaries."""

from __future__ import annotations

from pathlib import Path

from aspd.multimodel.cli.judge_aspd_v3_interpretability_stage1 import (
    _summarize,
)
from aspd.multimodel.v3.mechanistic_validation import FinalIntruderJudgeContract

ROOT = Path(__file__).resolve().parents[1]


def _row(component: str, site: str, position: int, correct: bool) -> dict:
    return {
        "task_id": f"{component}/{position}",
        "parent_task_id": component,
        "component_key": component,
        "site": site,
        "predicted_position": position if correct else (position % 5) + 1,
        "correct_position": position,
        "correct": correct,
        "valid": True,
    }


def test_summary_bootstraps_components_and_separates_sites() -> None:
    rows = [
        _row("pre/0001", "pre", 1, True),
        _row("pre/0001", "pre", 2, False),
        _row("mid/0001", "mid", 1, True),
        _row("mid/0001", "mid", 2, True),
    ]
    summary = _summarize(
        rows,
        task_kind="paper",
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    assert summary["by_site_feature_clustered"]["pre"]["mean"] == 0.5
    assert summary["by_site_feature_clustered"]["mid"]["mean"] == 1.0
    assert summary["equal_site_macro"] == 0.75


def test_rotation_summary_requires_all_five_correct() -> None:
    rows = [_row("pre/0001", "pre", position, True) for position in range(1, 6)]
    summary = _summarize(
        rows,
        task_kind="rotation",
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    assert summary["rotation_consistency"]["fraction_all_five_correct"] == 1.0
    assert summary["rotation_consistency"]["mean_correct_rotations"] == 5.0


def test_stage1_mistral_calibration_contract_is_pinned() -> None:
    contract = FinalIntruderJudgeContract.load(
        ROOT
        / "configs/multimodel/aspd_v3/interpretability_stage1_mistral_calibration_v1.yaml"
    )
    assert contract.judge_model_id == "mistralai/Mistral-Nemo-Instruct-2407"
    assert contract.paid_api is False

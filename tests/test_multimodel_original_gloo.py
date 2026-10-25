from __future__ import annotations

import os
from pathlib import Path

import pytest

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_gloo import execute_three_process_gloo_verification

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="the official PyTorch Windows build exposes Gloo but has no supported Gloo device",
)


def test_actual_three_process_gloo_resume_is_exact(tmp_path: Path) -> None:
    report = execute_three_process_gloo_verification(
        load_experiment_config(CONFIG),
        tmp_path / "gloo",
        tmp_path / "gloo_report.json",
    )
    assert report["status"] == "LOCALLY_TESTED"
    assert report["world_size"] == 3
    assert report["ddp_synchronized_state_equal"]
    assert all(item["model_max_abs_difference"] == 0 for item in report["comparisons"])
    assert all(item["optimizer_max_abs_difference"] == 0 for item in report["comparisons"])

from __future__ import annotations

from pathlib import Path

from aspd.multimodel.cli.prepare_delta_behavioral_v3_activation_cache import (
    EXPECTED_GROUP_COUNTS,
    audit_v3_measurement_inputs,
)

ROOT = Path(__file__).parents[1]


def test_real_v3_measurement_inputs_pass_frozen_gate() -> None:
    phase4 = ROOT / "out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/phase4_judge_46712130"
    audit = audit_v3_measurement_inputs(
        response_dir=ROOT / "out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/responses_1024",
        protocol_manifest=ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_manifest.json",
        scores_path=phase4 / "behavioral_scores.json",
        benchmark_path=phase4 / "behavioral_benchmark.json",
        candidates_path=ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json",
        checkpoint_path=ROOT / "out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512/checkpoint_00002048.pt",
    )
    assert audit["measurement_allowed"] is True
    assert audit["candidates"]["group_counts"] == EXPECTED_GROUP_COUNTS
    assert audit["candidates"]["selection_uses_v3"] is False
    assert audit["phase4_input_audit"]["response_records"] == 360
    assert audit["phase4_input_audit"]["judge_records"] == 240

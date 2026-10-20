from __future__ import annotations

import pytest
import torch

from aspd.multimodel.diagnostics.lora_precision import compare_update_precision
from aspd.multimodel.sports_risk import (
    ReducedPairedErrorMetrics,
    ReducedTensorMetrics,
    build_group_definitions,
    effective_grounding_batch,
)


def _metadata() -> list[dict[str, object]]:
    return [
        {
            "category": "risk",
            "generator_side": "target_generated",
            "behavioral_score": 2,
            "prompt_token_count": 3,
            "sequence_length": 5,
            "sport_domain": "climbing",
        },
        {
            "category": "sports_control",
            "generator_side": "base_generated",
            "behavioral_score": 0,
            "prompt_token_count": 2,
            "sequence_length": 6,
            "sport_domain": "climbing",
        },
        {
            "category": "neutral",
            "generator_side": "target_generated",
            "behavioral_score": None,
            "prompt_token_count": 2,
            "sequence_length": 4,
            "sport_domain": "neutral",
        },
    ]


def test_v3_groups_keep_neutral_unlabeled_and_define_fixed_regions() -> None:
    groups = {group.name: group for group in build_group_definitions(_metadata())}
    assert groups["category/risk"].members == frozenset({0})
    assert groups["category/neutral"].members == frozenset({2})
    assert groups["score/high_permissiveness/region/response"].members == frozenset(
        {0}
    )
    assert groups["score/cautious/region/response"].members == frozenset({1})
    assert groups["region/response"].region == "response"


def test_reduced_group_metrics_match_exact_small_example() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    prediction = target + 1.0
    mask = torch.tensor([[True, False]])
    metrics = ReducedTensorMetrics(2)
    metrics.update(target, prediction, mask)
    result = metrics.result()
    assert result["rows"] == 1
    assert result["rmse"] == pytest.approx(1.0)
    assert result["actual_output_rms"] == pytest.approx((2.5) ** 0.5)

    paired = ReducedPairedErrorMetrics()
    paired.update(target, prediction, target + 2, prediction + 2, mask)
    assert paired.result()["delta_error_rms"] == pytest.approx(0.0)


def test_lora_precision_detects_exact_bf16_merge() -> None:
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    a = torch.tensor([[0.25, -0.5]])
    b = torch.tensor([[0.5], [-0.25]])
    ideal = b @ a
    target = (base.float() + ideal).to(torch.bfloat16)
    result = compare_update_precision(base, target, a, b, scaling=1.0)
    assert result["predicted_bf16_merge_exact"] is True
    assert result["predicted_bf16_merge_max_abs_error"] == pytest.approx(0.0)
    assert 0.0 <= result["ideal_nonzero_effective_zero_fraction"] <= 1.0


def test_effective_grounding_batch_reuses_base_and_replaces_target() -> None:
    frozen = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)
    merged_target = torch.full((2, 3, 4), 99)
    base, target = effective_grounding_batch(
        {"activations": frozen}, [{}, {"R": merged_target}], 0, 2
    )
    torch.testing.assert_close(base, frozen[:, :, 0, :])
    torch.testing.assert_close(target, merged_target)

    with pytest.raises(ValueError, match="missing effective merged Target R"):
        effective_grounding_batch({"activations": frozen}, [{}, {}], 0, 2)

from __future__ import annotations

import pytest
import torch

from aspd.multimodel.v3.gate_collapse_diagnostic import (
    cosine_similarity,
    feature_group_masks,
    masked_gradient_summary,
    summarize_feature_gradients,
)


def test_feature_groups_separate_natural_selection_from_candidacy() -> None:
    logits = torch.tensor(
        [[[-1.0, 2.0, 1.0, -2.0], [-3.0, 1.0, -1.0, -4.0]]]
    )
    selected = torch.tensor(
        [[[False, True, False, False], [False, True, False, False]]]
    )
    dead = torch.tensor([True, False, True, True])
    fires = torch.tensor([0, 10, 2, 0])

    groups = feature_group_masks(logits, selected, dead, fires)

    assert groups["naturally_selected"].tolist() == [False, True, False, False]
    assert groups["positive_but_never_selected"].tolist() == [False, False, True, False]
    assert groups["nonpositive_on_all_anchors"].tolist() == [True, False, False, True]
    assert groups["dead_and_positive_on_fixed_pool"].tolist() == [False, False, True, False]


def test_masked_gradient_summary_reports_exact_zero_starvation() -> None:
    values = torch.tensor([0.0, -2.0, 4.0, 0.0])
    mask = torch.tensor([True, False, False, True])

    summary = masked_gradient_summary(values, mask)

    assert summary["count"] == 2
    assert summary["nonzero_fraction"] == 0.0
    assert summary["mean_absolute"] == 0.0


def test_feature_gradient_summary_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        masked_gradient_summary(torch.zeros(3), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="one-dimensional"):
        summarize_feature_gradients(torch.zeros(1, 2), {"all": torch.ones(2, dtype=torch.bool)})


def test_cosine_similarity_handles_parallel_orthogonal_and_zero() -> None:
    assert cosine_similarity(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])) == pytest.approx(1.0)
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(torch.zeros(2), torch.ones(2)) == 0.0

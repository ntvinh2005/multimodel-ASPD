from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aspd.multimodel.cli.evaluate_aspd_v3_encoder_logit_drift import (
    _load_contract,
    _logit_summary,
)
from aspd.multimodel.v3.logit_drift_diagnostic import (
    factorial_shapley,
    symmetric_logit_drift,
)


def test_symmetric_logit_drift_is_exact_and_splits_interaction() -> None:
    weight_0 = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    weight_1 = torch.tensor([[2.0, 1.0], [-0.5, 1.5]])
    bias_0 = torch.tensor([0.1, -0.2])
    bias_1 = torch.tensor([0.4, -0.1])
    representation_0 = torch.tensor([[1.0, 3.0], [-2.0, 0.5]])
    representation_1 = torch.tensor([[2.0, 4.0], [-1.0, 1.5]])

    result = symmetric_logit_drift(
        weight_0,
        bias_0,
        representation_0,
        weight_1,
        bias_1,
        representation_1,
    )
    torch.testing.assert_close(
        result.bias + result.readout_weight + result.representation,
        result.total,
    )
    expected = representation_1 @ weight_1.T + bias_1 - (
        representation_0 @ weight_0.T + bias_0
    )
    torch.testing.assert_close(result.total, expected)


def test_symmetric_logit_drift_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="representation width"):
        symmetric_logit_drift(
            torch.zeros(3, 2),
            torch.zeros(3),
            torch.zeros(4, 4),
            torch.zeros(3, 2),
            torch.zeros(3),
            torch.zeros(4, 4),
        )


def test_factorial_shapley_recovers_additive_binary_effects() -> None:
    effects = torch.tensor([1.5, -2.0, 0.25, 3.0])
    values = {}
    for mask in range(16):
        key = tuple((mask >> index) & 1 for index in range(4))
        values[key] = torch.tensor(7.0) + sum(
            effects[index] * key[index] for index in range(4)
        )

    observed = torch.stack(factorial_shapley(values))
    torch.testing.assert_close(observed, effects)


def test_factorial_shapley_requires_complete_matrix() -> None:
    with pytest.raises(ValueError, match="every binary factor"):
        factorial_shapley({(0, 0): torch.tensor(0.0), (1, 1): torch.tensor(1.0)})


def test_logit_summary_counts_candidates_before_batch_topk() -> None:
    logits = torch.tensor([[[-1.0, 0.0, 0.5, 2.0], [-2.0, 1.0, 3.0, -4.0]]])
    summary = _logit_summary(logits)

    assert summary["negative_fraction"] == pytest.approx(3 / 8)
    assert summary["zero_fraction"] == pytest.approx(1 / 8)
    assert summary["positive_candidates_per_anchor"]["mean"] == pytest.approx(2.0)


def test_replay_contract_is_fail_closed() -> None:
    contract = _load_contract(
        Path(
            "configs/multimodel/aspd_v3/"
            "granite_normalized_target_encoder_logit_drift_replay_v1.yaml"
        )
    )
    assert contract["checkpoint_steps"] == [1000, 1221]
    assert contract["counterfactual_count"] == 16
    assert contract["training_authorized_by_this_contract"] is False

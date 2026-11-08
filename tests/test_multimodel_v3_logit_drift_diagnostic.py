from __future__ import annotations

import pytest
import torch

from aspd.multimodel.v3.logit_drift_diagnostic import symmetric_logit_drift


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

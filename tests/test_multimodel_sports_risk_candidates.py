from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from aspd.multimodel.sports_risk_candidates import (
    _squared_contribution_terms,
    positive_percentile,
    standardized_mean_difference,
)


def test_positive_percentile_leaves_nonpositive_values_at_zero() -> None:
    result = positive_percentile(np.asarray([-2.0, 0.0, 1.0, 3.0]))
    np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0])


def test_standardized_mean_difference_uses_response_rows() -> None:
    values = np.asarray([[1.0], [2.0], [4.0], [5.0]])
    left = np.asarray([True, True, False, False])
    right = ~left
    assert standardized_mean_difference(values, left, right)[0] == pytest.approx(
        -3.0 / np.sqrt(0.5)
    )


def test_squared_contribution_terms_close_exact_decomposition() -> None:
    read_base = torch.tensor([[[2.0]]])
    read_target_on_base = torch.tensor([[[3.0]]])
    read_target = torch.tensor([[[4.0]]])
    u_base = torch.tensor([[1.0, 0.0]])
    u_target = torch.tensor([[0.0, 1.0]])
    squared = _squared_contribution_terms(
        read_base, read_target_on_base, read_target, u_base, u_target
    )
    # ||2*[1,0]||^2, ||4*[0,1]||^2, ||4*[0,1]-2*[1,0]||^2,
    # ||3*[0,1]-2*[1,0]||^2, ||(4-3)*[0,1]||^2.
    torch.testing.assert_close(
        squared.flatten(), torch.tensor([4.0, 16.0, 20.0, 13.0, 1.0])
    )


def test_named_metric_slices_have_independent_safetensor_storage(tmp_path) -> None:
    backing = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    tensors = {
        f"metric_{index}": backing[..., index].contiguous().clone()
        for index in range(2)
    }
    path = tmp_path / "metrics.safetensors"
    save_file(tensors, path)
    loaded = load_file(path)
    torch.testing.assert_close(loaded["metric_0"], backing[..., 0])
    torch.testing.assert_close(loaded["metric_1"], backing[..., 1])

"""Splicing a dictionary with its error term kept equals the masked component forward."""

import pytest
import torch
from param_decomp.components import LinearComponents

from aspd.eval.adapters.source import ablate

C, D_IN, D_OUT, B, L = 7, 5, 4, 3, 6
SELECTED = [1, 3, 5]


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    bias = torch.randn(D_OUT, dtype=torch.float64)
    comp = LinearComponents(C, D_IN, D_OUT, bias=bias).to(torch.float64)
    target_weight = torch.randn(D_OUT, D_IN, dtype=torch.float64)
    weight_delta = target_weight - comp.weight
    x = torch.randn(B, L, D_IN, dtype=torch.float64)
    g = torch.rand(B, L, C, dtype=torch.float64)
    return comp, target_weight, weight_delta, x, g


def _delta_mask(value: float) -> torch.Tensor:
    return torch.full((B, L), value, dtype=torch.float64)


def test_unmasked_forward_reproduces_the_target_weight():
    comp, target_weight, weight_delta, x, _ = _setup()
    y = comp.forward(
        x,
        mask=torch.ones(B, L, C, dtype=torch.float64),
        weight_delta_and_mask=(weight_delta, _delta_mask(1.0)),
    )
    expected = x @ target_weight.t() + comp.bias
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("m_delta", [0.0, 0.37, 1.0])
def test_ablation_equals_masked_forward_for_any_weight_delta_mask(m_delta: float):
    comp, _, weight_delta, x, g = _setup()

    keep = torch.ones(C, dtype=torch.float64)
    keep[SELECTED] = 0.0

    y_reference = comp.forward(x, mask=g, weight_delta_and_mask=(weight_delta, _delta_mask(m_delta)))
    y_ablated = comp.forward(
        x, mask=g * keep, weight_delta_and_mask=(weight_delta, _delta_mask(m_delta))
    )

    zeta = comp.get_component_acts(x) * g
    selected = torch.zeros(C, dtype=torch.bool)
    selected[SELECTED] = True
    via_subtraction = ablate(y_reference, zeta, comp.U.detach(), selected)

    torch.testing.assert_close(y_ablated, via_subtraction)


def test_ablating_nothing_is_the_identity():
    comp, _, weight_delta, x, g = _setup()
    y = comp.forward(x, mask=g, weight_delta_and_mask=(weight_delta, _delta_mask(1.0)))
    zeta = comp.get_component_acts(x) * g
    out = ablate(y, zeta, comp.U.detach(), torch.zeros(C, dtype=torch.bool))
    torch.testing.assert_close(out, y)


def test_ablation_matches_upstreams_decode_plus_error_form():
    """The SAE arm: `decode(f·¬T) + (a − decode(f))` and direct subtraction agree."""
    torch.manual_seed(1)
    n_features, d = 9, 4
    w_dec = torch.randn(n_features, d, dtype=torch.float64)
    b_dec = torch.randn(d, dtype=torch.float64)
    acts = torch.randn(B, L, d, dtype=torch.float64)
    features = torch.rand(B, L, n_features, dtype=torch.float64).clamp_min(0)

    selected = torch.zeros(n_features, dtype=torch.bool)
    selected[[0, 4, 7]] = True

    def decode(f: torch.Tensor) -> torch.Tensor:
        return f @ w_dec + b_dec

    error = acts - decode(features)
    upstream = decode(features * (~selected).to(features.dtype)) + error
    ours = ablate(acts, features, w_dec, selected)

    torch.testing.assert_close(ours, upstream)

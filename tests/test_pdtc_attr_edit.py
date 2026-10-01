"""Single-feature editing on a PD Transcoder run."""

import torch
from param_decomp.components import LinearComponents
from torch import nn

from aspd.eval.adapters.component import target_module_bias
from aspd.eval.editing.attribution import bias_residue_for, is_rank_one
from aspd.eval.editing.edit import component_delta_weight
from aspd.transcoder_components import TranscoderLinearComponents

C, D_IN, D_OUT = 16, 5, 7


def _tc() -> TranscoderLinearComponents:
    torch.manual_seed(0)
    comp = TranscoderLinearComponents(C=C, d_in=D_IN, d_out=D_OUT, bias=None)
    with torch.no_grad():
        comp.V.normal_()
        comp.U.normal_()
        comp.b_dec.normal_()
        comp.b_out.normal_()
    return comp


def test_a_transcoder_arm_is_rank_one():
    """`dy/dm_c = z_c U_c` holds exactly -- the two biases are not functions of `m`. Checked by
    autograd rather than by asserting the class, so the predicate is pinned to the DERIVATIVE it
    claims and not to a naming decision.
    """
    comp = _tc()
    assert is_rank_one(comp)

    x = torch.randn(3, D_IN)
    m = torch.rand(3, C, requires_grad=True)
    y = comp.forward(x, mask=m)
    # d(sum y)/dm_c = z_c * sum_d U[c, d]
    y.sum().backward()
    z = comp.get_component_acts(x)
    expected = z * comp.U.sum(-1)
    assert torch.allclose(m.grad, expected, atol=1e-5), (m.grad[0, :3], expected[0, :3])


def test_the_rank_one_predicate_still_rejects_a_look_alike():
    """The allowlist must not have become an `isinstance`: a subclass whose `U` has the right shape
    and the wrong meaning would pass, and produce a plausible table instead of an error.
    """

    class LookAlike(LinearComponents):
        pass

    assert not is_rank_one(LookAlike(C, D_IN, D_OUT, bias=None))
    assert is_rank_one(LinearComponents(C, D_IN, D_OUT, bias=None))


def test_the_edit_no_longer_raises_on_a_transcoder_arm():
    """`component_delta_weight` gates on the SAME predicate the planner reports `edit_skipped`
    from, so an arm can never be planned as editable and then die at the edit.
    """
    comp = _tc()
    sel = torch.tensor([0, 4, 9])
    dW = component_delta_weight(comp, sel)
    assert dW.shape == (D_OUT, D_IN)
    expected = sum(torch.outer(comp.U[c], comp.V[:, c]) for c in sel.tolist())
    assert torch.allclose(dW, expected, atol=1e-6)


def test_bias_residue_is_what_the_weight_edit_leaves_behind():
    """Component `c` contributes `((x - b_dec) . V_c) U_c`; `W - dW` removes `(x . V_c) U_c`. The
    difference is token-INDEPENDENT and is exactly what this reports.
    """
    comp = _tc()
    sel = torch.tensor([1, 2])
    residue = bias_residue_for(comp, sel)

    x = torch.randn(4, D_IN)
    removed_by_weight = torch.stack(
        [(x @ comp.V[:, c]).unsqueeze(-1) * comp.U[c] for c in sel.tolist()]
    ).sum(0)
    actually_written = torch.stack(
        [((x - comp.b_dec) @ comp.V[:, c]).unsqueeze(-1) * comp.U[c] for c in sel.tolist()]
    ).sum(0)
    # The gap is the same vector at every token, and it is `residue`.
    gap = removed_by_weight - actually_written
    assert torch.allclose(gap, residue.expand_as(gap), atol=1e-5)


def test_bias_residue_is_none_off_a_transcoder():
    """Every VPD arm has no `b_dec`, so its weight-only edit leaves no centring residue and the
    report must not carry a zero as if one had been measured.
    """
    assert bias_residue_for(LinearComponents(C, D_IN, D_OUT, bias=None), torch.tensor([0])) is None


# ---- the site reference --------------------------------------------------------------------


class _FakeComponentModel:
    def __init__(self, target: nn.Module) -> None:
        self.target_model = target


def test_the_site_bias_is_read_off_the_target_not_off_the_components():
    target = nn.Sequential()
    target.add_module("mod", nn.Linear(D_IN, D_OUT, bias=True))
    model = _FakeComponentModel(target)

    b = target_module_bias(model, "mod")
    assert b is not None and torch.equal(b, target.mod.bias)

    # ...and the components a `tc` arm actually carries would have returned None.
    assert _tc().bias is None


def test_a_biasless_target_still_reports_none():
    target = nn.Sequential()
    target.add_module("mod", nn.Linear(D_IN, D_OUT, bias=False))
    assert target_module_bias(_FakeComponentModel(target), "mod") is None

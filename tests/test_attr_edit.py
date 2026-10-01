"""Single-feature editing: attribution, the component edit, and its localization."""

import pytest
import torch
from param_decomp.components import LinearComponents
from torch import nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from aspd.eval.editing.attribution import (
    AttributionAccumulator,
    analytic_terms,
    autograd_terms,
    m_raw,
    ranking_order,
    top_k_selection,
)
from aspd.eval.editing.edit import (
    EditSpec,
    component_delta_weight,
    norm_matched_delta,
    patched_target_weight,
    random_selection,
)
from aspd.eval.editing.measure import DeltaAccumulator, _rows_in_chunk
from aspd.eval.editing.sample import FeatureSample, draw_sample, load_sample, write_sample
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

C, D_IN, D_OUT, F, B, L = 9, 6, 5, 20, 3, 4


class _StubModel:
    """The two members `autograd_terms` reaches for on a `ComponentModel`."""

    def __init__(self, components: LinearComponents, weight_delta: torch.Tensor, module: str):
        self.components = {module: components}
        self._delta = {module: weight_delta}

    def calc_weight_deltas(self):
        return self._delta


def _setup(seed: int = 0, dtype=torch.float64):
    torch.manual_seed(seed)
    comp = LinearComponents(C, D_IN, D_OUT, bias=torch.randn(D_OUT, dtype=dtype)).to(dtype)
    target_weight = torch.randn(D_OUT, D_IN, dtype=dtype)
    weight_delta = target_weight - comp.weight
    x = torch.randn(B, L, D_IN, dtype=dtype)
    g = torch.rand(B, L, C, dtype=dtype)
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(d_in=D_OUT, n_features=F, top_k=4, group_fracs=(0.5, 0.5))
    ).to(dtype)
    with torch.no_grad():
        sae.b_dec.normal_(std=0.1)
        sae.threshold.fill_(0.05)
    sae.eval()
    return comp, target_weight, weight_delta, x, g, sae


# --------------------------------------------------------------------------- 1. the estimator


def test_analytic_attribution_equals_autograd():
    """`df_j/dm_c = L_j z_c M_raw[j,c]`, against the real backward through the same forward."""
    module = "block.mlp.c_fc"
    comp, target_weight, weight_delta, x, g, sae = _setup()
    y_true = x @ target_weight.t() + comp.bias
    ids = torch.arange(F)

    # Every feature that is alive somewhere; a dead one has an empty A_j and no derivative.
    features = sae.features(y_true)
    active = features > 0
    alive = active.reshape(-1, F).any(dim=0).nonzero(as_tuple=True)[0]
    assert alive.numel() >= 3, "fixture produced too few active features to compare"
    ids, active = ids[alive], active[..., alive]

    analytic = AttributionAccumulator(ids.numel(), C)
    gate_term, unit_term = analytic_terms(g, comp.get_component_acts(x))
    analytic.add(
        active.reshape(-1, ids.numel()),
        gate_term.reshape(-1, C),
        unit_term.reshape(-1, C),
    )
    mean_gate, mean_unit, _ = analytic.finalize()
    m = m_raw(sae, comp.U.detach(), ids)
    a_gate_analytic, a_unit_analytic = mean_gate * m, mean_unit * m

    model = _StubModel(comp, weight_delta, module)
    sum_gate, sum_unit = autograd_terms(
        model, module, sae, x=x, y_true=y_true, g=g, feature_ids=ids, active=active
    )
    count = active.reshape(-1, ids.numel()).sum(dim=0).double()[:, None]

    torch.testing.assert_close(a_gate_analytic, sum_gate / count, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(a_unit_analytic, sum_unit / count, rtol=1e-9, atol=1e-9)


def test_autograd_reads_the_feature_at_the_true_activation():
    """The recentring makes `y(g) == y_true`, so `L_j` is the BASELINE's active set."""
    module = "block.mlp.c_fc"
    comp, target_weight, weight_delta, x, g, sae = _setup(seed=3)
    y_true = x @ target_weight.t() + comp.bias
    ones = torch.ones(x.shape[:-1], dtype=x.dtype)
    y_g = comp.forward(x, mask=g, weight_delta_and_mask=(weight_delta, ones))
    assert not torch.allclose(y_g, y_true), "fixture has no faithfulness residual to speak of"

    ids = (sae.features(y_true) > 0).reshape(-1, F).any(dim=0).nonzero(as_tuple=True)[0][:3]
    active = (sae.features(y_true) > 0)[..., ids]
    model = _StubModel(comp, weight_delta, module)
    sum_gate, _ = autograd_terms(
        model, module, sae, x=x, y_true=y_true, g=g, feature_ids=ids, active=active
    )
    assert (sum_gate.abs().sum(dim=1) > 0).all()


def test_ranking_is_nested_across_k():
    """`top_20` must CONTAIN `top_10`, or the k sweep swaps components instead of adding them."""
    torch.manual_seed(0)
    scores = torch.randn(C)
    for small, large in ((1, 3), (3, 5), (5, C)):
        assert set(top_k_selection(scores, small).tolist()) <= set(
            top_k_selection(scores, large).tolist()
        )
    assert ranking_order(scores)[0] == scores.abs().argmax()


# --------------------------------------------------------------------------- 2. the edit


def test_delta_weight_is_the_dropped_components():
    comp, *_ = _setup()
    selection = torch.tensor([0, 4, 7])
    keep = torch.ones(C, dtype=comp.U.dtype)
    keep[selection] = 0.0
    without = (comp.V.detach() * keep) @ comp.U.detach()
    torch.testing.assert_close(
        comp.weight.detach() - component_delta_weight(comp, selection), without.t()
    )


def test_delta_weight_rejects_a_non_rank_one_arm():
    class _Fake(LinearComponents):
        pass

    comp = _Fake(C, D_IN, D_OUT, bias=None)
    with pytest.raises(AssertionError, match="rank-1 edit"):
        component_delta_weight(comp, torch.tensor([0]))


@pytest.mark.parametrize("layout", ["linear", "conv1d"])
def test_patched_weight_reaches_the_module_in_its_own_layout(layout: str):
    """PD's `[d_out, d_in]` delta against `Conv1D`'s `[d_in, d_out]` storage."""
    torch.manual_seed(0)
    delta = torch.randn(D_OUT, D_IN, dtype=torch.float64)
    x = torch.randn(B, D_IN, dtype=torch.float64)

    if layout == "linear":
        module = nn.Linear(D_IN, D_OUT, bias=False).to(torch.float64)
        original = module.weight.detach().clone()
    else:
        module = RadfordConv1D(D_OUT, D_IN).to(torch.float64)
        original = module.weight.detach().clone()
    model = nn.Sequential()
    model.add_module("proj", module)

    before = model.proj(x)
    with patched_target_weight(model, "proj", delta):
        during = model.proj(x)
    after = model.proj(x)

    torch.testing.assert_close(during, before - x @ delta.t())
    torch.testing.assert_close(model.proj.weight.detach(), original)
    torch.testing.assert_close(after, before)


def test_patched_weight_restores_after_an_exception():
    module = nn.Linear(D_IN, D_OUT, bias=False)
    model = nn.Sequential()
    model.add_module("proj", module)
    original = module.weight.detach().clone()
    with pytest.raises(RuntimeError), patched_target_weight(
        model, "proj", torch.randn(D_OUT, D_IN)
    ):
        raise RuntimeError("boom")
    torch.testing.assert_close(module.weight.detach(), original)


def test_norm_matched_control_keeps_the_direction_and_matches_the_norm():
    comp, *_ = _setup()
    selection = random_selection(C, 3, torch.Generator().manual_seed(0))
    delta = component_delta_weight(comp, selection)
    scaled = norm_matched_delta(delta, 2.5)
    assert float(scaled.norm()) == pytest.approx(2.5)
    torch.testing.assert_close(
        scaled / scaled.norm(), delta / delta.norm(), rtol=1e-12, atol=1e-12
    )


def test_norm_matched_prediction_carries_the_scale():
    """A scaled edit removes `s·v_c`, so its first-order prediction must scale with it."""
    from aspd.eval.editing.run import _row

    class _Acc:
        def __init__(self, targets):
            self.targets = targets

        def summarize(self, _i, _j):
            return {"delta_signed": 0.0}

    a_gate = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    a_unit = torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float64)
    selection = torch.tensor([0, 1])
    accs = {"local_f7": _Acc(torch.tensor([7])), "global": _Acc(torch.tensor([7]))}

    class _Base:
        sample = type("S", (), {"feature_ids": [7]})()

    plain = _row(EditSpec("random", 2, None), 0, 7, selection, torch.eye(2), 1.0,
                 a_gate, a_unit, accs, _Base())
    scaled = _row(EditSpec("norm_matched", 2, None), 0, 7, selection, torch.eye(2), 1.0,
                  a_gate, a_unit, accs, _Base(), prediction_scale=3.0)

    assert plain.predicted_delta_unit == pytest.approx(-2.0)     # -(4 - 2)
    assert plain.predicted_delta_gate == pytest.approx(-1.0)     # -(2 - 1)
    assert scaled.predicted_delta_unit == pytest.approx(-6.0)
    assert scaled.predicted_delta_gate == pytest.approx(-3.0)


def test_a_non_rank_one_arm_reports_attribution_instead_of_dying():
    """`p2_plan` promises those arms a table; the edit sweep is what is undefined, not the pass."""
    from aspd.eval.editing.attribution import is_rank_one

    assert is_rank_one(LinearComponents(C, D_IN, D_OUT, bias=None))

    class ReparameterizedComponents(LinearComponents):
        pass

    other = ReparameterizedComponents(C, D_IN, D_OUT, bias=None)
    assert isinstance(other, LinearComponents), (
        "the guard is only meaningful for a SUBCLASS -- an isinstance check would pass it "
        "straight through to an edit that means something else"
    )
    assert not is_rank_one(other)


# --------------------------------------------------------------------------- 3. the reductions


def test_collateral_excludes_the_target_row_only():
    torch.manual_seed(0)
    targets = torch.tensor([2, 5])
    acc = DeltaAccumulator(F, targets)
    f_base = torch.zeros(7, F)
    f_edit = torch.zeros(7, F)
    f_base[:, 2] = 1.0
    f_edit[:, 2] = 0.25          # target 2 moved by 0.75 at every position
    f_edit[:, 9] = 0.5           # one collateral feature moved by 0.5
    pre = torch.zeros(7, targets.numel())
    acc.add(f_base, f_edit, pre, pre)

    summary = acc.summarize(0, 2)
    assert summary["delta_abs"] == pytest.approx(0.75)
    assert summary["collateral_abs"] == pytest.approx(0.5)
    assert summary["collateral_l0"] == pytest.approx(1.0)
    assert summary["localization"] == pytest.approx(0.75 / 1.25)
    assert summary["death_rate"] == pytest.approx(0.0)

    # Feature 5 never fired, so from ITS point of view both moved features are collateral.
    other = acc.summarize(1, 5)
    assert other["delta_abs"] == pytest.approx(0.0)
    assert other["collateral_abs"] == pytest.approx(1.25)


def test_death_and_birth_rates_use_their_own_denominators():
    targets = torch.tensor([0])
    acc = DeltaAccumulator(F, targets)
    f_base = torch.zeros(4, F)
    f_edit = torch.zeros(4, F)
    f_base[:2, 0] = 1.0          # alive at 2 of 4 positions
    f_edit[0, 0] = 0.0           # one of them killed
    f_edit[1, 0] = 1.0
    f_edit[3, 0] = 0.4
    pre = torch.zeros(4, 1)
    acc.add(f_base, f_edit, pre, pre)
    summary = acc.summarize(0, 0)
    assert summary["death_rate"] == pytest.approx(0.5)     # 1 of the 2 baseline-active
    assert summary["birth_rate"] == pytest.approx(0.5)     # 1 of the 2 baseline-inactive


def test_rows_in_chunk_pairs_each_position_with_its_own_activation():
    seq_len = 4
    positions = torch.tensor([1, 6, 7, 22])  # sequences 0, 1, 1, 5
    chunk = torch.tensor([1, 5])
    rows, sel = _rows_in_chunk(positions, chunk, seq_len)
    # Sequence 1 is slot 0 of the chunk, sequence 5 is slot 1.
    torch.testing.assert_close(sel, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(rows, torch.tensor([2, 3, 6]))

    empty_rows, empty_sel = _rows_in_chunk(positions, torch.tensor([3]), seq_len)
    assert empty_sel.numel() == 0 and empty_rows.numel() == 0


# --------------------------------------------------------------------------- the sample file


def _sample_inputs():
    density = torch.zeros(12, dtype=torch.float64)
    density[:10] = torch.linspace(1e-5, 1e-2, 10, dtype=torch.float64)
    density[10] = 0.5     # too dense
    density[11] = 0.0     # dead
    support = torch.full((12,), 500, dtype=torch.int64)
    support[0] = 3        # eligible by density, unmeasurable in practice
    return density, support


def test_sample_applies_both_filters_and_is_seed_deterministic():
    density, support = _sample_inputs()
    kwargs = dict(density=density, support=support, site="s", fingerprint="abc",
                  n_features=4, min_support=100, max_density=0.2, n_tokens=1000)
    first = draw_sample(seed=7, **kwargs)
    assert first.n_density_eligible == 10
    assert first.n_eligible == 9                      # the low-support one is dropped
    assert set(first.feature_ids) <= set(range(1, 10))
    assert first.feature_ids == draw_sample(seed=7, **kwargs).feature_ids
    assert first.feature_ids != draw_sample(seed=8, **kwargs).feature_ids


def test_sample_refuses_to_draw_more_than_it_can():
    density, support = _sample_inputs()
    with pytest.raises(AssertionError, match="pass density"):
        draw_sample(density=density, support=support, site="s", fingerprint="abc",
                    n_features=20, min_support=100, max_density=0.2, n_tokens=1000)


def test_loading_a_sample_from_another_dictionary_fails(tmp_path):
    density, support = _sample_inputs()
    sample = draw_sample(density=density, support=support, site="s", fingerprint="abc",
                         n_features=3, min_support=100, max_density=0.2, n_tokens=1000)
    path = write_sample(sample, tmp_path / "sampled_features.json")
    assert isinstance(load_sample(path, fingerprint="abc", site="s"), FeatureSample)
    with pytest.raises(AssertionError, match="Feature ids are meaningless"):
        load_sample(path, fingerprint="different")
    with pytest.raises(AssertionError, match="drawn at site"):
        load_sample(path, fingerprint="abc", site="other")

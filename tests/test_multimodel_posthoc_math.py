"""Pure-tensor tests of the post-hoc formulation (no source model involved)."""

from __future__ import annotations

import math

import pytest
import torch

from aspd.multimodel.posthoc.comparison import (
    ModelParentView,
    aggregated_parent_scores,
    compare_pair,
    jaccard,
    parent_distribution,
    total_variation,
)
from aspd.multimodel.posthoc.contributions import (
    component_coefficients,
    component_contributions,
    contribution_sq_norms,
)
from aspd.multimodel.posthoc.edges import EdgeRecord, group_rows
from aspd.multimodel.posthoc.ids import ComponentId, GateId, MatrixId
from aspd.multimodel.posthoc.importance import ImportanceTable, MatrixImportanceAccumulator
from aspd.multimodel.posthoc.parents import Parent, ParentRule, parent_gate_set, parent_set
from aspd.multimodel.posthoc.representatives import RepresentativeRule, select_representatives

torch.set_num_threads(1)


def _c(model: str, layer: int, matrix: str, feature: int) -> ComponentId:
    return ComponentId(MatrixId(model, layer, matrix), feature)


# ------------------------------------------------------------------------- ids
def test_upstream_order_follows_layer_compute_order() -> None:
    q, o = MatrixId("m", 3, "q"), MatrixId("m", 3, "o")
    gate, up, down = (MatrixId("m", 3, n) for n in ("mlp_gate", "mlp_up", "mlp_down"))
    assert q.is_upstream_of(o) and o.is_upstream_of(gate) and up.is_upstream_of(down)
    assert not MatrixId("m", 3, "k").is_upstream_of(q)  # same stage
    assert not gate.is_upstream_of(up)  # same stage
    assert down.is_upstream_of(MatrixId("m", 4, "q"))
    assert not q.is_upstream_of(MatrixId("other", 9, "o"))  # different model


def test_gamma_maps_component_to_shared_gate_and_keys_round_trip() -> None:
    b = _c("gemma", 12, "mlp_up", 77)
    assert b.gate == GateId("mid", 77)
    assert _c("smol", 9, "k", 5).gate == GateId("pre", 5)
    assert ComponentId.from_key(b.key) == b
    assert MatrixId.from_key("gemma.L09.q") == MatrixId("gemma", 9, "q")


# ---------------------------------------------------------------- contributions
def test_coefficients_and_contribution_norms_match_definition() -> None:
    torch.manual_seed(0)
    x, V, U = torch.randn(5, 4), torch.randn(3, 4), torch.randn(3, 6)
    gate = torch.tensor([[1, 0, 1]] * 5, dtype=torch.bool)
    e = component_coefficients(x, V, gate)
    for a in range(5):
        for i in range(3):
            expected = float(gate[a, i]) * float(V[i] @ x[a])
            assert math.isclose(float(e[a, i]), expected, rel_tol=1e-5, abs_tol=1e-6)
    dense = component_contributions(e, U)
    torch.testing.assert_close(contribution_sq_norms(e, U), dense.square().sum(-1))


# ------------------------------------------------------------------- importance
def _brute_importance(x, y, gate, U, V):
    e = component_coefficients(x.double(), V.double(), gate)
    sq = (e[..., None] * U.double()[None]).square().sum(-1)
    centered = y.double() - y.double().mean(0)
    sigma = centered.square().sum(-1).mean().sqrt()
    fired = gate.sum(0).double()
    beta = (sq.sum(0) / fired).sqrt() / sigma
    q = sq.mean(0).sqrt()
    return sigma, beta, q, q / q.sum()


def test_importance_accumulator_matches_brute_force_and_is_chunk_invariant() -> None:
    torch.manual_seed(1)
    A, C, d_in, d_out = 40, 6, 5, 7
    x, y = torch.randn(A, d_in), torch.randn(A, d_out) + 3.0
    U, V = torch.randn(C, d_out), torch.randn(C, d_in)
    gate = torch.rand(A, C) < 0.4
    gate[:, 5] = False  # never fires -> beta undefined
    matrix = MatrixId("m", 0, "o")
    single = MatrixImportanceAccumulator(matrix, C, d_out)
    single.update(x, y, gate, U, V)
    chunked = MatrixImportanceAccumulator(matrix, C, d_out)
    for rows in torch.arange(A).split(7):
        chunked.update(x[rows], y[rows], gate[rows], U, V)
    a, b = single.finalize(), chunked.finalize()
    sigma, beta, q, rho = _brute_importance(x, y, gate, U, V)
    assert math.isclose(a.sigma, float(sigma), rel_tol=1e-9)
    torch.testing.assert_close(a.beta[:5], beta[:5])
    assert math.isnan(float(a.beta[5]))
    torch.testing.assert_close(a.q, q)
    torch.testing.assert_close(a.rho, rho)
    torch.testing.assert_close(a.beta, b.beta, equal_nan=True)
    torch.testing.assert_close(a.q, b.q)
    assert math.isclose(a.sigma, b.sigma, rel_tol=1e-9)
    assert int(a.fire_count.sum()) == int(gate.sum())


def test_importance_table_round_trip(tmp_path) -> None:
    acc = MatrixImportanceAccumulator(MatrixId("m", 1, "q"), 3, 2)
    acc.update(torch.randn(4, 2), torch.randn(4, 2), torch.ones(4, 3, dtype=torch.bool),
               torch.randn(3, 2), torch.randn(3, 2))
    table = ImportanceTable([acc.finalize()])
    table.save(tmp_path)
    loaded = ImportanceTable.load(tmp_path)
    entry = loaded[MatrixId("m", 1, "q")]
    torch.testing.assert_close(entry.beta, table[MatrixId("m", 1, "q")].beta)
    assert entry.sigma == table[MatrixId("m", 1, "q")].sigma


# -------------------------------------------------------------- representatives
def _table_with_beta(values: dict[MatrixId, list[float]]) -> ImportanceTable:
    from aspd.multimodel.posthoc.importance import MatrixImportance

    entries = []
    for matrix, beta in values.items():
        b = torch.tensor(beta, dtype=torch.float64)
        entries.append(MatrixImportance(matrix, b, b.clone(), b / b.nansum(), 1.0,
                                        torch.ones(len(beta), dtype=torch.int64), 10))
    return ImportanceTable(entries)


def test_representatives_topk_threshold_site_and_nan() -> None:
    q, k, o, up = (MatrixId("m", 2, n) for n in ("q", "k", "o", "mlp_up"))
    table = _table_with_beta({
        q: [0.5, 0.2], k: [0.9, float("nan")], o: [5.0, 5.0], up: [0.1, 0.3],
    })
    rule = RepresentativeRule("beta", k_rep=2, tau_rep=0.15)
    # PRE gate 0: only q,k are in J_PRE; o/up belong to MID.
    assert select_representatives(table, "m", GateId("pre", 0), rule) == (
        ComponentId(k, 0), ComponentId(q, 0))
    # NaN excluded, 0.2 above tau.
    assert select_representatives(table, "m", GateId("pre", 1), rule) == (ComponentId(q, 1),)
    # Ties broken by MatrixId order; top-2 of {o: 5.0, up: 0.3}.
    assert select_representatives(table, "m", GateId("mid", 1), rule) == (
        ComponentId(o, 1), ComponentId(up, 1))
    with pytest.raises(ValueError):
        RepresentativeRule("both", 1, 0.0)


# ------------------------------------------------------------------ parents
def _record(a: ComponentId, scores: dict[ComponentId, float]) -> EdgeRecord:
    return EdgeRecord.from_components("anchor", a, 1.0, tuple(scores),
                                      torch.tensor(list(scores.values())))


def test_parent_set_uses_abs_topk_then_threshold() -> None:
    a = _c("m", 3, "o", 0)
    b1, b2, b3, b4 = (_c("m", 2, "v", i) for i in range(4))
    record = _record(a, {b1: 0.5, b2: -2.0, b3: 0.05, b4: 1.0})
    parents = parent_set(record, ParentRule(k_par=3, tau=0.1))
    assert [p.component for p in parents] == [b2, b4, b1]
    assert parents[0].score == -2.0
    assert parent_set(record, ParentRule(k_par=4, tau=0.6)) == (
        Parent(b2, -2.0), Parent(b4, 1.0))


def test_parent_gate_set_is_union_of_gamma() -> None:
    p1 = (Parent(_c("m", 1, "q", 4), 1.0), Parent(_c("m", 1, "o", 4), 1.0))
    p2 = (Parent(_c("m", 0, "mlp_up", 9), 1.0),)
    assert parent_gate_set([p1, p2]) == {GateId("pre", 4), GateId("mid", 4), GateId("mid", 9)}


# --------------------------------------------------------------- comparison
def test_jaccard_and_total_variation_hand_values() -> None:
    a, b, c = GateId("pre", 1), GateId("pre", 2), GateId("mid", 3)
    assert jaccard(frozenset({a, b}), frozenset({b, c})) == pytest.approx(1 / 3)
    assert jaccard(frozenset(), frozenset()) is None
    assert total_variation({a: 0.5, b: 0.5}, {b: 0.5, c: 0.5}) == pytest.approx(0.5)
    assert total_variation({a: 1.0}, {a: 1.0}) == 0.0
    assert parent_distribution({a: 0.0}) is None


def test_aggregated_scores_normalize_by_representative_count_and_weight() -> None:
    a1, a2 = _c("m", 3, "o", 0), _c("m", 3, "mlp_up", 0)
    b = _c("m", 2, "v", 7)
    parents = {a1: (Parent(b, -2.0),)}  # a2 has no stored parents but counts in |R|
    scores = aggregated_parent_scores(parents, (a1, a2), {a1: 1.0, a2: 1.0},
                                      {a1: 4.0, a2: 1.0})
    # w = |E| ||u|| / sigma = 2 * 1 / 4 = 0.5;  S = 0.5 / |R| = 0.25
    assert scores == {GateId("pre", 7): pytest.approx(0.25)}


def test_compare_pair_validity_and_values() -> None:
    gate = GateId("mid", 0)
    a_n, a_m = _c("n", 3, "o", 0), _c("m", 5, "mlp_down", 0)
    shared, only_n = _c("n", 2, "v", 1), _c("n", 2, "q", 2)
    view_n = ModelParentView("n", (a_n,), {a_n: (Parent(shared, 1.0), Parent(only_n, 3.0))},
                             {a_n: 1.0}, {a_n: 1.0})
    view_m = ModelParentView("m", (a_m,), {a_m: (Parent(_c("m", 4, "k", 1), 2.0),)},
                             {a_m: 1.0}, {a_m: 1.0})
    result = compare_pair(gate, "x", view_n, view_m)
    # gate sets: {pre/1, pre/2} vs {pre/1}
    assert result.jaccard == pytest.approx(0.5)
    # pi_n = {pre/1: .25, pre/2: .75}, pi_m = {pre/1: 1}
    assert result.total_variation == pytest.approx(0.75)
    assert (result.strength_left, result.strength_right) == (pytest.approx(4.0), pytest.approx(2.0))
    empty = ModelParentView("m", (), {}, {}, {})
    invalid = compare_pair(gate, "x", view_n, empty)
    assert invalid.total_variation is None and not invalid.distribution_valid


def test_group_rows_one_anchor_per_document() -> None:
    documents = torch.tensor([0, 0, 1, 2, 2, 2])
    groups = group_rows(documents, [0, 1, 2, 3, 4, 5], max_group=2)
    for group in groups:
        assert len({int(documents[r]) for r in group}) == len(group) <= 2
    assert sorted(r for g in groups for r in g) == [0, 1, 2, 3, 4, 5]

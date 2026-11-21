"""Hook, probe and edge-score tests on real (tiny, random) Hugging Face Llama models.

The decisive check is the finite-difference identity behind the Taylor edge score:

    e_a(s) - e_a(s | e_b -> (1 - eps) e_b)  ~=  eps * E_{b->a}

which must hold for every edge, and holds *exactly* (any eps, including full ablation)
when the path from ``b`` to ``a`` is linear (``v -> o`` inside one attention block).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aspd.multimodel.posthoc.backend import MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.edges import edge_scores_for_group, exact_ablation_effect
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io, probed_forward
from aspd.multimodel.posthoc.toy_backend import ToyBackend

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def toy() -> ToyBackend:
    backend = ToyBackend({"seed": 3, "n_features": 12, "k_per_anchor": 5})
    torch.set_default_dtype(torch.float32)
    yield backend
    backend.close()


def _group(toy: ToyBackend, model: str):
    pool = next(toy.pools(1))
    inputs = pool.models[model]
    # first anchor of each document -> one anchor per document
    rows, seen = [], set()
    for row in range(inputs.anchor_count):
        document = int(inputs.anchor_document[row])
        if document not in seen:
            rows.append(row)
            seen.add(document)
    sub = inputs.subset(rows)
    gates = {site: pool.gates[site][rows] for site in ("pre", "mid")}
    ids = [pool.anchor_ids[r] for r in rows]
    return sub, gates, ids


def test_capture_matches_native_projection(toy: ToyBackend) -> None:
    model = toy.source_model("alpha")
    pool = next(toy.pools(1))
    captured = capture_matrix_io(model, pool.models["alpha"], toy.hook_specs("alpha"))
    for spec in toy.hook_specs("alpha"):
        x, y = captured[spec.matrix]
        weight = model.get_submodule(spec.module_path).weight
        torch.testing.assert_close(y, x @ weight.T, rtol=1e-5, atol=1e-5)


def test_probes_do_not_change_the_forward_and_hooks_are_removed(toy: ToyBackend) -> None:
    model = toy.source_model("beta")
    specs = toy.hook_specs("beta")
    sub, _gates, _ids = _group(toy, "beta")
    deepest = max((s.matrix for s in specs), key=lambda m: m.order)
    before = capture_matrix_io(model, sub, specs)
    with torch.enable_grad():
        forward = probed_forward(model, sub, specs,
                                 probe=[s.matrix for s in specs if s.matrix != deepest],
                                 capture=[deepest])
    torch.testing.assert_close(forward.inputs[deepest].detach(), before[deepest][0])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_edge_scores_match_finite_difference_ablation(toy: ToyBackend) -> None:
    model_key = "alpha"
    model = toy.source_model(model_key)
    specs = toy.hook_specs(model_key)
    factors = toy.factors(model_key)
    sub, gates, ids = _group(toy, model_key)
    layer = toy.layers[-1]
    targets = [MatrixId(model_key, layer, "o"), MatrixId(model_key, layer, "mlp_down")]
    requests = {}
    for matrix in targets:
        feature = int(gates[matrix.site].float().sum(0).argmax())
        requests[ComponentId(matrix, feature)] = gates[matrix.site][:, feature]
    records = edge_scores_for_group(model, sub, ids, gates, specs, factors, requests)
    assert records, "expected at least one firing downstream anchor"
    checked = 0
    for record in records:
        if not record.upstream:
            continue
        row = ids.index(record.anchor_id)
        order = record.scores.abs().argsort(descending=True)[:3]
        for k in order.tolist():
            b = record.upstream[k]
            assert b.matrix.is_upstream_of(record.downstream.matrix)
            eps = 1e-3
            fd = exact_ablation_effect(model, sub, gates, specs, factors,
                                       record.downstream, b, fraction=eps)[row] / eps
            assert float(fd) == pytest.approx(float(record.scores[k]), rel=2e-2, abs=1e-4)
            checked += 1
    assert checked >= 3


def test_linear_path_edge_equals_exact_full_ablation(toy: ToyBackend) -> None:
    model_key = "beta"
    model = toy.source_model(model_key)
    specs, factors = toy.hook_specs(model_key), toy.factors(model_key)
    sub, gates, ids = _group(toy, model_key)
    o = MatrixId(model_key, toy.layers[0], "o")
    feature = int(gates["mid"].float().sum(0).argmax())
    a = ComponentId(o, feature)
    records = edge_scores_for_group(model, sub, ids, gates, specs, factors,
                                    {a: gates["mid"][:, feature]})
    compared = 0
    for record in records:
        row = ids.index(record.anchor_id)
        for b, score in zip(record.upstream, record.scores.tolist(), strict=True):
            if b.matrix == MatrixId(model_key, toy.layers[0], "v"):
                exact = exact_ablation_effect(model, sub, gates, specs, factors, a, b)[row]
                assert float(exact) == pytest.approx(score, rel=1e-4, abs=1e-6)
                compared += 1
    assert compared > 0


def test_batched_anchors_equal_single_anchor_runs(toy: ToyBackend) -> None:
    model_key = "alpha"
    model = toy.source_model(model_key)
    specs, factors = toy.hook_specs(model_key), toy.factors(model_key)
    sub, gates, ids = _group(toy, model_key)
    matrix = MatrixId(model_key, toy.layers[-1], "mlp_up")
    feature = int(gates["mid"].float().sum(0).argmax())
    a = ComponentId(matrix, feature)
    batched = {r.anchor_id: r for r in edge_scores_for_group(
        model, sub, ids, gates, specs, factors, {a: gates["mid"][:, feature]})}
    for row, anchor in enumerate(ids):
        if anchor not in batched:
            continue
        single_inputs = ModelInputs(
            sub.input_ids[sub.anchor_document[row]][None],
            sub.attention_mask[sub.anchor_document[row]][None],
            torch.zeros(1, dtype=torch.long), sub.anchor_token[row : row + 1],
        )
        single_gates = {s: g[row : row + 1] for s, g in gates.items()}
        (single,) = edge_scores_for_group(model, single_inputs, [anchor], single_gates, specs,
                                          factors, {a: single_gates["mid"][:, feature]})
        assert single.upstream == batched[anchor].upstream
        torch.testing.assert_close(single.scores, batched[anchor].scores, rtol=1e-4, atol=1e-6)


def test_probed_forward_rejects_two_anchors_in_one_document(toy: ToyBackend) -> None:
    pool = next(toy.pools(1))
    inputs = pool.models["alpha"]
    same_document = [r for r in range(inputs.anchor_count) if int(inputs.anchor_document[r]) == 0]
    sub = inputs.subset(same_document[:2])
    with pytest.raises(ValueError, match="one anchor per document"):
        probed_forward(toy.source_model("alpha"), sub, toy.hook_specs("alpha"),
                       probe=(), capture=[toy.hook_specs("alpha")[0].matrix])


class _FusedBlock(nn.Module):
    """Two logical matrices sliced from one fused projection, then a second matrix."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(10, 4)
        self.fused = nn.Linear(4, 6, bias=False)  # [q | k] slices of width 3
        self.down = nn.Linear(3, 4, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed(input_ids)
        qk = self.fused(h)
        return self.down(torch.tanh(qk[..., :3]) * qk[..., 3:])


def test_fused_output_slices_are_probed_and_ablated_independently() -> None:
    # float64: a central finite difference of a small edge needs more than float32 digits.
    model = _FusedBlock().double().eval().requires_grad_(False)
    q, k, o = MatrixId("f", 0, "q"), MatrixId("f", 0, "k"), MatrixId("f", 0, "o")
    specs = (
        MatrixHookSpec(q, "fused", (0, 3), 4, 3),
        MatrixHookSpec(k, "fused", (3, 6), 4, 3),
        MatrixHookSpec(o, "down", None, 3, 4),
    )
    generator = torch.Generator().manual_seed(1)
    factors = {spec.matrix: (torch.randn(2, spec.d_out, generator=generator, dtype=torch.float64),
                             torch.randn(2, spec.d_in, generator=generator, dtype=torch.float64))
               for spec in specs}
    inputs = ModelInputs(torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.ones(2, 3, dtype=torch.long),
                         torch.tensor([0, 1]), torch.tensor([2, 1]))
    gates = {"pre": torch.ones(2, 2, dtype=torch.bool), "mid": torch.ones(2, 2, dtype=torch.bool)}
    captured = capture_matrix_io(model, inputs, specs)
    torch.testing.assert_close(captured[q][1], captured[q][0] @ model.fused.weight[:3].T)
    torch.testing.assert_close(captured[k][1], captured[k][0] @ model.fused.weight[3:].T)
    a = ComponentId(o, 0)
    (r0, r1) = edge_scores_for_group(model, inputs, ["x", "y"], gates, specs, factors,
                                     {a: torch.tensor([True, True])})
    for row, record in enumerate((r0, r1)):
        for b, score in zip(record.upstream, record.scores.tolist(), strict=True):
            fd = exact_ablation_effect(model, inputs, gates, specs, factors, a, b,
                                       fraction=1e-6)[row] / 1e-6
            assert float(fd) == pytest.approx(score, rel=1e-3, abs=1e-6)
        assert {b.matrix for b in record.upstream} == {q, k}

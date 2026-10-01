"""The pair viewer on synthetic directions and a temporary database."""

import json
import sqlite3

import pytest
import torch

from aspd.analysis.pairs import scores as sc
from aspd.analysis.pairs.spaces import (
    HeadLayout,
    compatibility,
    head_pairs,
    module_spaces,
    parse_role,
)
from aspd.analysis.pairs.store import HarvestStore
from aspd.analysis.pairs.suggest import available_templates, link_for
from aspd.analysis.pairs.weights import Directions


def _dirs(mat: torch.Tensor) -> Directions:
    return Directions(mat=mat, norms=mat.norm(dim=1))


def test_gpt2_and_gemma_roles_and_spaces():
    fc = module_spaces("transformer.h.6.mlp.c_fc", 768, 3072, 64)
    proj = module_spaces("transformer.h.6.mlp.c_proj", 3072, 768, 64)
    assert (fc.role, proj.role) == ("mlp.in", "mlp.out")
    assert fc.read.key == "resid" and proj.write.key == "resid"
    # The two MLP hidden spaces are NOT the same key: c_fc writes pre-activation, c_proj reads post.
    assert fc.write.key != proj.read.key
    assert compatibility(fc.write, proj.read)["flat"] is True

    v = module_spaces("model.layers.9.self_attn.v_proj", 2304, 1024, 256)
    o = module_spaces("model.layers.9.self_attn.o_proj", 2048, 2304, 256)
    # Value and output-projection input ARE the same space -- that is what makes OV exact.
    assert v.write.key == o.read.key == "L9.attn.z"
    compat = compatibility(v.write, o.read)
    assert compat["flat"] is False, "GQA: 1024 != 2048, a flat OV dot is not even defined"
    assert compat["per_head"] is True and compat["n_head_pairs"] == 8


def test_head_pairs_are_gqa_aware():
    assert head_pairs(HeadLayout(12, 64), HeadLayout(12, 64)) == [(h, h) for h in range(12)]
    assert head_pairs(HeadLayout(8, 256), HeadLayout(4, 256)) == [
        (0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3)
    ]
    assert head_pairs(HeadLayout(4, 256), HeadLayout(8, 256)) == [
        (0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)
    ]
    with pytest.raises(AssertionError):
        head_pairs(HeadLayout(12, 64), HeadLayout(8, 256))


def test_incompatible_spaces_report_both_dimensions():
    fc = module_spaces("transformer.h.6.mlp.c_fc", 768, 3072, 64)
    q = module_spaces("transformer.h.7.attn.c_attn.q_proj", 768, 768, 64)
    compat = compatibility(fc.write, q.read)
    assert compat["flat"] is False and compat["per_head"] is False
    assert "3072" in str(compat["reason"]) and "768" in str(compat["reason"])


def test_parse_role_rejects_unknown_paths():
    with pytest.raises(AssertionError):
        parse_role("transformer.h.3.ln_1")


def test_flat_cosine_matches_a_hand_computation():
    torch.manual_seed(0)
    x, y = torch.randn(5, 16), torch.randn(7, 16)
    sa = module_spaces("transformer.h.0.mlp.c_fc", 8, 16, 64).write
    sb = module_spaces("transformer.h.0.mlp.c_proj", 16, 8, 64).read
    row = sc.score_row(_dirs(x), _dirs(y), 2, metric="cosine", space_a=sa, space_b=sb,
                       per_head=False, head=None)
    want = torch.nn.functional.cosine_similarity(x[2][None, :], y, dim=1)
    assert torch.allclose(row.top, want, atol=1e-6)
    assert row.top_head is None and row.n_population == 7


def test_dot_equals_cosine_times_norm_when_the_write_side_is_unit():
    torch.manual_seed(1)
    x = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)  # normalize_decoder: true
    y = torch.randn(6, 16)
    sa = module_spaces("transformer.h.0.mlp.c_fc", 8, 16, 64).write
    sb = module_spaces("transformer.h.0.mlp.c_proj", 16, 8, 64).read
    kw = dict(space_a=sa, space_b=sb, per_head=False, head=None)
    cos = sc.score_row(_dirs(x), _dirs(y), 1, metric="cosine", **kw).top
    dot = sc.score_row(_dirs(x), _dirs(y), 1, metric="dot", **kw).top
    assert torch.allclose(dot, cos * y.norm(dim=1), atol=1e-5)


def test_per_head_top_and_bottom_come_from_independent_reductions():
    space = module_spaces("transformer.h.0.attn.c_attn.q_proj", 4, 4, 2).write  # 2 heads x 2 dims
    other = module_spaces("transformer.h.0.attn.c_attn.k_proj", 4, 4, 2).write
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])          # head0 = (1,0), head1 = (1,0)
    y = torch.tensor([[0.5, 0.0, -1.0, 0.0],          # b0: +0.5 on head0, -1.0 on head1
                      [0.1, 0.0, 0.1, 0.0]])          # b1: weakly positive on both
    row = sc.score_row(_dirs(x), _dirs(y), 0, metric="dot", space_a=space, space_b=other,
                       per_head=True, head=None)
    assert row.top.tolist() == pytest.approx([0.5, 0.1])
    assert row.bottom.tolist() == pytest.approx([-1.0, 0.1])
    assert row.top_head is not None and row.top_head.tolist() == [0, 0]
    assert row.bottom_head is not None and row.bottom_head.tolist() == [1, 0]
    out = sc.top_bottom(row, k=2)
    assert (out["top"][0]["idx"], out["top"][0]["head"]) == (0, 0)
    assert out["top"][0]["score"] == pytest.approx(0.5)
    assert (out["bottom"][0]["idx"], out["bottom"][0]["head"]) == (0, 1)
    assert out["bottom"][0]["score"] == pytest.approx(-1.0)


def test_population_statistics_use_the_unreduced_scores():
    """`mean`/`std` must come from all H x C_B values, not from the max-reduced row."""
    space = module_spaces("transformer.h.0.attn.c_attn.q_proj", 4, 4, 2).write
    other = module_spaces("transformer.h.0.attn.c_attn.k_proj", 4, 4, 2).write
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    y = torch.tensor([[1.0, 0.0, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]])
    row = sc.score_row(_dirs(x), _dirs(y), 0, metric="dot", space_a=space, space_b=other,
                       per_head=True, head=None)
    assert row.n_population == 4  # 2 heads x 2 components
    assert row.mean == pytest.approx(0.0)  # +1, +1, -1, -1 -- NOT the max-row mean of +1


def test_top_bottom_excludes_self_on_both_lists():
    torch.manual_seed(2)
    x = torch.randn(9, 12)
    sa = module_spaces("transformer.h.0.mlp.c_fc", 6, 12, 64).write
    row = sc.score_row(_dirs(x), _dirs(x), 3, metric="cosine", space_a=sa, space_b=sa,
                       per_head=False, head=None)
    out = sc.top_bottom(row, k=4, exclude=3)
    assert 3 not in [r["idx"] for r in out["top"]]
    assert 3 not in [r["idx"] for r in out["bottom"]]


def test_templates_follow_the_architecture():
    gpt2 = {
        m: module_spaces(m, d_in, d_out, 64)
        for m, d_in, d_out in [
            ("transformer.h.0.mlp.c_fc", 768, 3072),
            ("transformer.h.0.mlp.c_proj", 3072, 768),
            ("transformer.h.1.mlp.c_fc", 768, 3072),
            ("transformer.h.1.mlp.c_proj", 3072, 768),
        ]
    }
    keys = {t["key"] for t in available_templates(gpt2)}
    assert "mlp_in_out" in keys and "resid_mlp_to_mlp" in keys
    assert "mlp_gate_down" not in keys and "attn_qk" not in keys

    gemma = {
        m: module_spaces(m, d_in, d_out, 256)
        for m, d_in, d_out in [
            ("model.layers.9.mlp.gate_proj", 2304, 9216),
            ("model.layers.9.mlp.up_proj", 2304, 9216),
            ("model.layers.9.mlp.down_proj", 9216, 2304),
        ]
    }
    keys = {t["key"] for t in available_templates(gemma)}
    assert {"mlp_gate_down", "mlp_up_down"} <= keys and "mlp_in_out" not in keys


def test_link_classification():
    fc = module_spaces("transformer.h.6.mlp.c_fc", 768, 3072, 64)
    proj = module_spaces("transformer.h.6.mlp.c_proj", 3072, 768, 64)
    q = module_spaces("transformer.h.6.attn.c_attn.q_proj", 768, 768, 64)
    k = module_spaces("transformer.h.6.attn.c_attn.k_proj", 768, 768, 64)
    v = module_spaces("transformer.h.6.attn.c_attn.v_proj", 768, 768, 64)
    o = module_spaces("transformer.h.6.attn.c_proj", 768, 768, 64)
    later = module_spaces("transformer.h.9.mlp.c_fc", 768, 3072, 64)
    assert link_for(fc, "write", proj, "read") == "pointwise"
    assert link_for(q, "write", k, "write") == "bilinear_form"
    assert link_for(v, "write", o, "read") == "identity"
    assert link_for(q, "read", k, "read") == "shared_input"
    assert link_for(proj, "write", later, "read") == "residual"
    # Same-layer template must not fire across layers.
    assert link_for(fc, "write", module_spaces("transformer.h.7.mlp.c_proj", 3072, 768, 64),
                    "read") != "pointwise"


def test_link_is_orientation_independent():
    """The two panels are one pair space queried from opposite ends and must agree on the mode."""
    q = module_spaces("transformer.h.6.attn.c_attn.q_proj", 768, 768, 64)
    k = module_spaces("transformer.h.6.attn.c_attn.k_proj", 768, 768, 64)
    fc = module_spaces("transformer.h.6.mlp.c_fc", 768, 3072, 64)
    proj = module_spaces("transformer.h.6.mlp.c_proj", 3072, 768, 64)
    v = module_spaces("transformer.h.6.attn.c_attn.v_proj", 768, 768, 64)
    o = module_spaces("transformer.h.6.attn.c_proj", 768, 768, 64)
    for x, xs, y, ys in ((q, "write", k, "write"), (fc, "write", proj, "read"),
                         (v, "write", o, "read")):
        assert link_for(x, xs, y, ys) == link_for(y, ys, x, xs) is not None


def _harvest_db(path, key: str) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE components (component_key TEXT, layer TEXT, component_idx INT,"
        " firing_density REAL, n_activation_examples INT, mean_activations TEXT,"
        " activation_examples TEXT, input_token_pmi TEXT, output_token_pmi TEXT)"
    )
    examples = [
        {
            "token_ids": [10, 11, 12],
            "firings": [False, False, True],
            "activations": {"causal_importance": [0.0, 0.0, 1.0],
                            "component_activation": [-9.0, 1.0, 4.0]},
        },
        {
            "token_ids": [13, 14, 15],
            "firings": [False, True, False],
            "activations": {"causal_importance": [0.0, 1.0, 0.0],
                            "component_activation": [2.0, 1.0, 0.5]},
        },
    ]
    con.execute(
        "INSERT INTO components VALUES (?,?,?,?,?,?,?,?,?)",
        (key, key.rsplit(":", 1)[0], 0, 0.01, 2, json.dumps({"causal_importance": 0.01}),
         json.dumps(examples), json.dumps({"top": [[10, 3.5]]}), json.dumps({"top": []})),
    )
    con.commit()
    con.close()


def test_harvest_sorting_and_pmi(tmp_path):
    db = tmp_path / "harvest.db"
    _harvest_db(db, "transformer.h.6.mlp.c_fc:0")
    store = HarvestStore(db, decode=lambda ids: [f"<{i}>" for i in ids])

    eff = store.component("transformer.h.6.mlp.c_fc", 0, sort="effective", window=5)
    assert eff is not None and eff.examples[0]["center"] == 2 and eff.examples[0]["peak"] == 4.0
    raw = store.component("transformer.h.6.mlp.c_fc", 0, sort="activation", window=5)
    assert raw is not None and raw.examples[0]["center"] == 0 and raw.examples[0]["peak"] == -9.0
    # The two orderings genuinely disagree -- the whole reason `effective` is the default.
    assert eff.examples[0]["center"] != raw.examples[0]["center"]

    assert eff.pmi_available is True and eff.input_pmi == [["<10>", 3.5]]
    assert eff.output_pmi == []


def test_harvest_falls_back_to_the_canonical_key(tmp_path):
    db = tmp_path / "harvest.db"
    _harvest_db(db, "h.6.mlp.in:0")  # topology-canonical, not the concrete target path
    store = HarvestStore(db, decode=lambda ids: [str(i) for i in ids])
    rec = store.component("transformer.h.6.mlp.c_fc", 0, sort="effective", window=5)
    assert rec is not None and rec.key == "h.6.mlp.in:0"


def test_missing_harvest_is_reported_not_faked(tmp_path):
    store = HarvestStore(tmp_path / "absent.db", decode=lambda ids: [])
    assert store.available is False
    assert store.component("transformer.h.6.mlp.c_fc", 0, sort="effective", window=5) is None


def test_pmi_absent_is_distinguished_from_pmi_empty(tmp_path):
    """The whole-model harvest stores `{"top": []}` for every component; that is not "no tokens"."""
    db = tmp_path / "harvest.db"
    _harvest_db(db, "transformer.h.6.mlp.c_fc:0")
    con = sqlite3.connect(db)
    con.execute("UPDATE components SET input_token_pmi = ?", (json.dumps({"top": [], "bottom": []}),))
    con.commit()
    con.close()
    rec = HarvestStore(db, decode=lambda ids: []).component(
        "transformer.h.6.mlp.c_fc", 0, sort="effective", window=5
    )
    assert rec is not None and rec.pmi_available is False


def _kappa_file(tmp_path, *, full=True, shared=False):
    """A `pair_coactivation.pt` for one module pair, small enough to assert on by hand."""
    src, dst = "transformer.h.0.attn.c_attn.v_proj", "transformer.h.0.attn.c_proj"
    pool = torch.arange(4) if full else torch.tensor([0, 2])
    kappa = torch.arange(pool.numel() * pool.numel(), dtype=torch.float64).reshape(
        pool.numel(), pool.numel()) / 100.0
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "pair_coactivation.pt"
    torch.save(
        {
            "kappa": {f"{src}|{dst}": kappa},
            "n_co": {f"{src}|{dst}": torch.ones_like(kappa)},
            "index": {src: pool, dst: pool},
            "pairs": {f"{src}|{dst}": {"src_module": src, "dst_module": dst,
                                       "templates": ["attn_ov"], "shared_gate": shared}},
            "meta": {"full_pool": full, "n_pairs": 1, "n_tokens": 100},
        },
        path,
    )
    return path, src, dst


def test_dot_coact_is_the_dot_multiplied_by_kappa():
    """The whole point of the metric: geometry alone ranks pairs that never co-fire."""
    x = torch.randn(3, 8)
    y = torch.randn(5, 8)
    sp = module_spaces("transformer.h.0.mlp.c_fc", 768, 8, None).write
    kappa = torch.tensor([2.0, 0.0, -1.0, 0.5, 10.0])
    kw = dict(space_a=sp, space_b=sp, per_head=False, head=None)
    plain = sc.score_row(_dirs(x), _dirs(y), 1, metric="dot", **kw).top
    coact = sc.score_row(_dirs(x), _dirs(y), 1, metric="dot_coact", kappa=kappa, **kw).top
    torch.testing.assert_close(coact, plain * kappa)
    # A component that never co-fires scores exactly 0 however well the two directions align.
    assert float(coact[1]) == 0.0


def test_dot_coact_multiplies_before_the_per_head_reduction():
    """A negative kappa swaps which head is the max, so reducing first names the wrong head."""
    sa = module_spaces("transformer.h.0.attn.c_attn.v_proj", 768, 768, 64).write
    x = torch.randn(2, 768)
    y = torch.randn(3, 768)
    kappa = torch.tensor([-1.0, 1.0, -2.0])
    kw = dict(space_a=sa, space_b=sa, per_head=True, head=None)
    row = sc.score_row(_dirs(x), _dirs(y), 0, metric="dot_coact", kappa=kappa, **kw)
    plain = sc.score_row(_dirs(x), _dirs(y), 0, metric="dot", **kw)
    # Where kappa > 0 the top head is unchanged; where kappa < 0 top and bottom must have swapped.
    assert int(row.top_head[1]) == int(plain.top_head[1])
    assert int(row.top_head[0]) == int(plain.bottom_head[0])
    assert int(row.top_head[2]) == int(plain.bottom_head[2])
    # And the reported score is still the largest signed one.
    assert float(row.top[0]) >= float(row.bottom[0])


def test_score_row_requires_kappa_exactly_for_kappa_metrics():
    x, y = torch.randn(2, 4), torch.randn(3, 4)
    sp = module_spaces("transformer.h.0.mlp.c_fc", 768, 4, None).write
    kw = dict(space_a=sp, space_b=sp, per_head=False, head=None)
    with pytest.raises(AssertionError):
        sc.score_row(_dirs(x), _dirs(y), 0, metric="dot_coact", **kw)
    with pytest.raises(AssertionError):
        sc.score_row(_dirs(x), _dirs(y), 0, metric="dot", kappa=torch.ones(3), **kw)
    with pytest.raises(AssertionError):
        sc.score_row(_dirs(x), _dirs(y), 0, metric="dot_coact", kappa=torch.ones(9), **kw)


def test_uncovered_partners_are_excluded_not_scored_zero():
    """A component with no kappa must not land mid-list among the genuine zeros."""
    row = sc.RowScores(
        top=torch.tensor([5.0, 0.0, -5.0, 0.0]),
        bottom=torch.tensor([5.0, 0.0, -5.0, 0.0]),
        top_head=None, bottom_head=None, mean=0.0, std=1.0, n_population=4,
    )
    covered = torch.tensor([True, False, True, False])
    out = sc.top_bottom(row, k=4, covered=covered)
    assert [e["idx"] for e in out["top"]] == [0, 2], "only measured partners may be ranked"
    assert [e["idx"] for e in out["bottom"]] == [2, 0]
    assert out["n_covered"] == 2 and out["n_components"] == 4
    assert len(sc.top_bottom(row, k=4)["top"]) == 4


def test_exclude_self_does_not_double_count_an_uncovered_component():
    row = sc.RowScores(
        top=torch.arange(5.0), bottom=torch.arange(5.0),
        top_head=None, bottom_head=None, mean=0.0, std=1.0, n_population=5,
    )
    covered = torch.tensor([True, True, True, False, False])
    # Excluding an UNCOVERED index removes nothing extra: 3 covered stay 3.
    assert len(sc.top_bottom(row, k=5, exclude=4, covered=covered)["top"]) == 3
    # Excluding a COVERED one leaves 2.
    assert len(sc.top_bottom(row, k=5, exclude=1, covered=covered)["top"]) == 2


def test_kappa_store_transposes_the_reverse_orientation(tmp_path):
    """Both panels rank the SAME directed edge; kappa is directed, so the lookup must transpose."""
    from aspd.analysis.pairs.store import KappaStore

    path, src, dst = _kappa_file(tmp_path)
    ks = KappaStore(path)
    assert ks.available and ks.covers(src, dst) and ks.covers(dst, src)
    fwd, _, _ = ks.matrix(src, dst)
    rev, _, _ = ks.matrix(dst, src)
    torch.testing.assert_close(fwd, rev.t())
    # kappa[1, 2] = 6/100 in the stored direction.
    vals, keep = ks.row(src, 1, dst, 4)
    assert keep is None, "a full pool narrows nothing, so there is no mask to carry"
    assert pytest.approx(float(vals[2])) == 0.06
    vals_r, _ = ks.row(dst, 2, src, 4)
    assert pytest.approx(float(vals_r[1])) == 0.06


def test_kappa_store_marks_unmeasured_components(tmp_path):
    from aspd.analysis.pairs.store import KappaStore

    path, src, dst = _kappa_file(tmp_path, full=False)
    ks = KappaStore(path)
    assert not ks.full_pool and ks.pool_size(src) == 2
    vals, keep = ks.row(src, 0, dst, 4)
    assert keep.tolist() == [True, False, True, False], "only pooled columns are measured"
    assert float(vals[1]) == 0.0 and float(vals[3]) == 0.0
    # A source outside the pool has no row at all -- distinct from a covered row of zeros.
    assert ks.row(src, 1, dst, 4) is None


def test_shared_encoder_pairs_are_flagged(tmp_path):
    from aspd.analysis.pairs.store import KappaStore

    path, src, dst = _kappa_file(tmp_path, shared=True)
    assert KappaStore(path).shared_gate(src, dst), "ASPD pairs inside one residual site must be flagged"
    path2, src2, dst2 = _kappa_file(tmp_path / "b", shared=False)
    assert not KappaStore(path2).shared_gate(src2, dst2)


def test_dot_coact_leads_the_metric_order():
    """The UI defaults to the first available metric, so the order is load-bearing."""
    assert sc.METRIC_SPECS[0].key == "dot_coact"
    assert sc.KAPPA_METRICS == {"dot_coact"}
    assert "dot_coact" not in sc.WEIGHT_METRICS, "it needs a data pass, not just the checkpoint"
    assert sc.DIRECTION_METRICS == {"cosine", "dot", "dot_coact"}


def test_the_null_excludes_unmeasured_partners():
    """Padding the null with the manufactured zeros of unmeasured partners inflates every z."""
    x = torch.randn(2, 6)
    y = torch.randn(20, 6)
    sp = module_spaces("transformer.h.0.mlp.c_fc", 768, 6, None).write
    covered = torch.zeros(20, dtype=torch.bool)
    covered[:5] = True
    kappa = torch.zeros(20)
    kappa[:5] = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5])
    kw = dict(space_a=sp, space_b=sp, per_head=False, head=None, metric="dot_coact", kappa=kappa)

    narrowed = sc.score_row(_dirs(x), _dirs(y), 0, keep=covered, **kw)
    padded = sc.score_row(_dirs(x), _dirs(y), 0, **kw)
    assert narrowed.n_population == 5 and padded.n_population == 20
    # 15 of 20 scores are a manufactured 0, which shrinks σ and pulls μ toward zero.
    assert padded.std < narrowed.std
    worst = float(narrowed.top[:5][narrowed.top[:5].abs().argmax()])
    z_padded = abs(worst - padded.mean) / padded.std
    z_narrow = abs(worst - narrowed.mean) / narrowed.std
    assert z_padded > 2 * z_narrow


def _harvest_with_densities(tmp_path, module, densities):
    db = tmp_path / "harvest.db"
    con = sqlite3.connect(db)
    with con:
        con.execute("CREATE TABLE components (component_key TEXT PRIMARY KEY, firing_density REAL,"
                    " mean_activations TEXT, activation_examples TEXT, input_token_pmi TEXT,"
                    " output_token_pmi TEXT)")
        con.executemany(
            "INSERT INTO components VALUES (?,?,?,?,?,?)",
            [(f"{module}:{i}", d, "{}", "[]", "", "") for i, d in enumerate(densities)],
        )
    con.close()
    return db


def test_densities_reads_the_band_and_defaults_absent_components_to_dead(tmp_path):
    """A component with no harvested row has no evidence, so it must read as dead, not as sparse."""
    from aspd.analysis.pairs.store import HarvestStore

    mod = "transformer.h.6.attn.c_proj"
    db = _harvest_with_densities(tmp_path, mod, [0.0, 1e-4, 5e-3, 4e-1])
    hs = HarvestStore(db, lambda t: [""])
    d = hs.densities(mod, 6)
    assert d is not None and d.shape == (6,)
    assert float(d[0]) == 0.0 and pytest.approx(float(d[3])) == 0.4
    assert float(d[4]) == 0.0 and float(d[5]) == 0.0, "absent components read as dead"
    # Cached: a second call must not re-query, and must give the identical object.
    assert hs.densities(mod, 6) is d
    assert hs.densities("transformer.h.0.mlp.c_fc", 4) is None, "no matching key -> cannot filter"


def test_the_density_band_excludes_dead_and_overly_dense_partners():
    """The band is two-sided: a component that never fires has no evidence either."""
    dens = torch.tensor([0.0, 1e-4, 1e-3, 5e-3, 2e-2, 4e-1])
    mask = (dens > 0) & (dens <= 5e-3)
    assert mask.tolist() == [False, True, True, True, False, False]

    # And the ranking must honour it on BOTH sides, not just the top list.
    row = sc.RowScores(
        top=torch.tensor([9.0, 1.0, 2.0, 3.0, -9.0, -8.0]),
        bottom=torch.tensor([9.0, 1.0, 2.0, 3.0, -9.0, -8.0]),
        top_head=None, bottom_head=None, mean=0.0, std=1.0, n_population=6,
    )
    out = sc.top_bottom(row, k=6, covered=mask)
    assert [e["idx"] for e in out["top"]] == [3, 2, 1]
    assert [e["idx"] for e in out["bottom"]] == [1, 2, 3]
    assert 0 not in [e["idx"] for e in out["top"]], "the dead component scored highest and must go"
    assert 4 not in [e["idx"] for e in out["bottom"]], "the densest scored lowest and must go"
    assert out["n_covered"] == 3


def _gpt2_byte_decoder() -> dict[str, int]:
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    return {ch: b for b, ch in bytes_to_unicode().items()}


def test_a_neuronpedia_token_is_decoded_out_of_the_byte_alphabet():
    """Neuronpedia substitutes the space byte and leaves every other one in GPT-2's
    `bytes_to_unicode` alphabet, so a newline arrives as `Ċ` and an em dash as `âĢĶ`.
    """
    from aspd.analysis.pairs.clean import decode_token

    dec = _gpt2_byte_decoder()
    assert decode_token("Ċ", dec) == "\n"
    assert decode_token(" âĢĶ", dec) == " —"
    assert decode_token("âĢĻ", dec) == "’"
    assert decode_token("Ð°Ð½Ð¸", dec) == "ани"
    assert decode_token("ðŁĩº", dec) == "\U0001f1fa"
    # Already-plain text passes through: the space Neuronpedia substituted, and ASCII.
    assert decode_token(" electricity", dec) == " electricity"
    assert decode_token("76561", dec) == "76561"
    assert decode_token("Ð", dec) == "�"
    # No tokenizer, no change: a model whose alphabet is not GPT-2's is left exactly as it came.
    assert decode_token("âĢĶ", None) == "âĢĶ"


def test_a_character_split_across_a_window_boundary_is_put_back_together():
    """Inside an activating window the tokens are adjacent, so a multi-byte character split across
    a boundary can be decoded -- Neuronpedia returns a curly quote as `' âĢ'` then `'ľ'`, and
    decoding each on its own would give two replacement characters instead of one quote.
    """
    from aspd.analysis.pairs.clean import decode_token, decode_tokens

    dec = _gpt2_byte_decoder()
    window = [" âĢ", "ľ", "free", "âĢ", "Ŀ", " replacements"]
    assert decode_tokens(window, dec) == [" ", "\u201c", "free", "", "\u201d", " replacements"]
    # Per-token decoding cannot do this, which is why the window path exists at all.
    assert [decode_token(t, dec) for t in window[:2]] == [" \ufffd", "\ufffd"]
    # Three tokens, one Cyrillic letter each, and plain text is untouched either way.
    assert decode_tokens([" payments", "Ċ", "Ð°", "Ð½", "Ð¸"], dec) == [
        " payments", "\n", "а", "н", "и"
    ]
    assert decode_tokens(window, None) == window


def test_a_repeated_autointerp_description_is_dropped_and_reported():
    from aspd.analysis.pairs.clean import SIMILAR, tidy_explanations

    kept, dropped = tidy_explanations(
        ["terms related to compensation or reward", " phrases related to compensation or rewards"]
    )
    assert kept == ["terms related to compensation or reward"]
    assert dropped == ["phrases related to compensation or rewards"], "nothing is dropped silently"
    both, none = tidy_explanations([
        "references to specific geographical locations, particularly cities",
        " references to specific locations or areas, particularly cities and towns",
    ])
    assert len(both) == 2 and none == [] and SIMILAR > 0.76
    # A sentence repeated INSIDE one description is stated once.
    one, _ = tidy_explanations(["tokens about X. tokens about X. and also Y."])
    assert one == ["tokens about X. and also Y."]


def test_a_repeated_activation_window_is_dropped_and_reported():
    """Neuronpedia lists the same window many times; the card should show it once."""
    from aspd.analysis.pairs.clean import dedupe_activations

    same = {"tokens": ["a", " b"], "values": [0.0, 1.5]}
    other = {"tokens": ["a", " b"], "values": [1.5, 0.0]}  # same text, fired elsewhere
    kept, dropped = dedupe_activations([same, dict(same), other, dict(same)])
    assert dropped == 2
    assert [k["values"] for k in kept] == [[0.0, 1.5], [1.5, 0.0]]
    assert dedupe_activations([]) == ([], 0)

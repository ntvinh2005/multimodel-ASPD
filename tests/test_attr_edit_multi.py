"""Multiple-feature editing: feature sets, the cond / global selection, and localization."""

import json

import pytest
import torch
from param_decomp.components import LinearComponents
from torch import Tensor, nn

from aspd.eval.editing.edit import component_delta_weight
from aspd.eval.editing.measure import (
    DeltaAccumulator,
    MeasureGroup,
    TokenPlan,
    measure_edit,
)
from aspd.eval.editing.multi import (
    MultiConfig,
    default_control_combos,
    default_top_k,
    prepare_multi,
    run_step_multi,
)
from aspd.eval.editing.run import AttrEditConfig, prepare, run_step
from aspd.eval.editing.sample import (
    FeaturePool,
    draw_combinations,
    draw_pool,
    draw_sample,
    load_pool,
    write_pool,
)
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

VOCAB, D_IN, D_OUT, C, F = 23, 6, 5, 9, 16
N_SEQ, SEQ_LEN = 6, 8
MODULE = "proj"


class _TinyTarget(nn.Module):
    """A model whose `proj` output is a dictionary site -- enough for the forward hooks."""

    def __init__(self, dtype):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_IN)
        self.proj = nn.Linear(D_IN, D_OUT)
        self.to(dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.proj(self.embed(tokens))


class _StubComponentModel:
    """The five members `attr_edit` reaches for on a real `ComponentModel`."""

    def __init__(self, target: _TinyTarget, comp: LinearComponents, w_gate: Tensor):
        self.target = target
        self.components = {MODULE: comp}
        self.w_gate = w_gate

    def target_weight(self, module: str) -> Tensor:
        assert module == MODULE
        return self.target.proj.weight.detach()

    def forward(self, tokens: Tensor, cache_type: str = "input"):
        assert cache_type == "input"
        return type("_Cached", (), {"cache": {MODULE: self.target.embed(tokens).detach()}})()

    def calc_causal_importances(self, cache: dict[str, Tensor], sampling: str = "continuous"):
        g = torch.sigmoid(cache[MODULE] @ self.w_gate)
        return type("_CI", (), {"lower_leaky": {MODULE: g}})()

    def calc_weight_deltas(self):
        return {MODULE: self.target_weight(MODULE) - self.components[MODULE].weight}


def _fixture(seed: int = 0, dtype=torch.float32):
    """fp32, not fp64: this fixture runs the REAL drivers, and `collect_positions` encodes in
    fp32 by design (the `[p, F]` block is the memory ceiling, not the precision floor).
    """
    torch.manual_seed(seed)
    target = _TinyTarget(dtype).eval()
    comp = LinearComponents(C, D_IN, D_OUT, bias=None).to(dtype)
    model = _StubComponentModel(target, comp, torch.randn(D_IN, C, dtype=dtype))
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(d_in=D_OUT, n_features=F, top_k=4, group_fracs=(0.5, 0.5))
    ).to(dtype).eval()
    with torch.no_grad():
        sae.b_dec.normal_(std=0.1)
        sae.threshold.fill_(0.0)
    tokens = torch.randint(0, VOCAB, (N_SEQ, SEQ_LEN))
    plan = TokenPlan(tokens=tokens, keep=torch.ones_like(tokens, dtype=torch.bool),
                     n_tokens_total=N_SEQ * SEQ_LEN)
    return target, model, comp, sae, plan


def _eligible_stats(target, sae, plan):
    """Density and support over the fixture, so the sample and pool draws have real inputs."""
    with torch.no_grad():
        features = sae.features(target(plan.tokens))
    support = (features > 0).reshape(-1, F).sum(dim=0)
    density = (support.double() / plan.n_tokens_total).clamp(max=0.19)
    return density, support


@pytest.mark.parametrize("kind", ["ranked", "random", "norm_matched"])
def test_m1_cond_reproduces_the_single_feature_driver(kind: str):
    target, model, _, sae, plan = _fixture()
    density, support = _eligible_stats(target, sae, plan)
    n_features, top_k = 3, (1, 2)
    sample = draw_sample(
        density=density, support=support, site=MODULE, fingerprint="fp",
        n_features=n_features, seed=42, max_density=0.2, min_support=1,
        n_tokens=plan.n_tokens_total,
    )
    single_cfg = AttrEditConfig(top_k=top_k, n_control_reps=2, n_tokens_global=2 * SEQ_LEN,
                                batch_size=2, seed=42)
    base = prepare(target, MODULE, sae, plan, sample, single_cfg,
                   device="cpu", cache_device="cpu")
    single = run_step(target, model, MODULE, sae, base, single_cfg, step=0,
                      run_dir="/tmp/run", sae_dir="/tmp/sae", site=MODULE, device="cpu")

    pool = draw_pool(
        density=density, support=support, site=MODULE, fingerprint="fp",
        seed_features=sample.feature_ids, pool_size=n_features + 2, seed=42,
        max_density=0.2, min_support=1, n_tokens=plan.n_tokens_total,
    )
    multi_cfg = MultiConfig(m=1, setup="cond", top_k=top_k, n_combinations=n_features,
                            n_control_combos=n_features, n_control_reps=2,
                            n_tokens_global=2 * SEQ_LEN, batch_size=2, seed=42)
    combos = draw_combinations(pool, 1, n_combinations=n_features, seed=42)
    assert combos == [[j] for j in sample.feature_ids]
    mbase = prepare_multi(target, MODULE, sae, plan, pool, combos, multi_cfg,
                          device="cpu", cache_device="cpu")
    multi = run_step_multi(target, model, MODULE, sae, mbase, multi_cfg, step=0,
                           run_dir="/tmp/run", sae_dir="/tmp/sae", site=MODULE, device="cpu")

    want = [r for r in single.edits if r.kind == kind]
    got = {(r.combo, r.k, r.rep): r for r in multi.edits if r.kind == kind}
    assert want, f"fixture produced no {kind} rows"
    assert all(r.block == "union" for r in multi.edits), "m=1 must not emit a target block"
    combo_of = {j: i for i, j in enumerate(sample.feature_ids)}
    for row in want:
        mine = got[(combo_of[row.feature_id], row.k, row.rep)]
        assert mine.selection == row.selection
        assert mine.edit_norm == pytest.approx(row.edit_norm)
        assert mine.predicted_delta_unit == pytest.approx(row.predicted_delta_unit)
        assert mine.predicted_delta_gate == pytest.approx(row.predicted_delta_gate)
        assert mine.score_share == pytest.approx(row.score_share)
        for metric, value in row.on_aj.items():
            assert mine.on_a[metric] == pytest.approx(value), (kind, row.k, metric)
        for metric, value in row.on_global.items():
            assert mine.on_global[metric] == pytest.approx(value), (kind, row.k, metric)


def test_global_setup_ranks_differently_from_cond():
    """The two setups are the same estimand on two token sets -- and they do disagree."""
    target, model, _, sae, plan = _fixture(seed=5)
    density, support = _eligible_stats(target, sae, plan)
    sample = draw_sample(density=density, support=support, site=MODULE, fingerprint="fp",
                         n_features=4, seed=42, max_density=0.2, min_support=1,
                         n_tokens=plan.n_tokens_total)
    pool = draw_pool(density=density, support=support, site=MODULE, fingerprint="fp",
                     seed_features=sample.feature_ids, pool_size=6, seed=42,
                     max_density=0.2, min_support=1, n_tokens=plan.n_tokens_total)
    reports = {}
    for setup in ("cond", "global"):
        cfg = MultiConfig(m=2, setup=setup, top_k=(1, 2), n_combinations=3, n_control_combos=3,
                          n_control_reps=1, n_tokens_global=2 * SEQ_LEN, batch_size=2, seed=42)
        combos = draw_combinations(pool, 2, n_combinations=3, seed=42)
        mbase = prepare_multi(target, MODULE, sae, plan, pool, combos, cfg,
                              device="cpu", cache_device="cpu")
        reports[setup] = run_step_multi(target, model, MODULE, sae, mbase, cfg, step=0,
                                        run_dir="/tmp/run", sae_dir="/tmp/sae",
                                        site=MODULE, device="cpu")
    assert reports["global"].global_skipped is None
    overlaps = [
        entry["rank_overlap_cond_vs_global"]
        for entry in reports["cond"].structure.values()
        if "rank_overlap_cond_vs_global" in entry
    ]
    assert overlaps, "the cond arm must still report the cond-vs-global overlap"
    assert min(overlaps) < 1.0, "global and cond produced identical rankings at every k"

    def ranked(r):
        return {(x.combo, x.k): tuple(x.selection)
                for x in r.edits if x.kind == "ranked" and x.block == "union"}

    assert ranked(reports["cond"]) != ranked(reports["global"])


# ------------------------------------------------------------------ 2. the selection


def test_selection_is_a_nested_union_bounded_by_m_times_k():
    target, model, _, sae, plan = _fixture(seed=2)
    density, support = _eligible_stats(target, sae, plan)
    sample = draw_sample(density=density, support=support, site=MODULE, fingerprint="fp",
                         n_features=3, seed=42, max_density=0.2, min_support=1,
                         n_tokens=plan.n_tokens_total)
    pool = draw_pool(density=density, support=support, site=MODULE, fingerprint="fp",
                     seed_features=sample.feature_ids, pool_size=6, seed=42,
                     max_density=0.2, min_support=1, n_tokens=plan.n_tokens_total)
    m, top_k = 3, (1, 2, 3)
    cfg = MultiConfig(m=m, setup="cond", top_k=top_k, n_combinations=3, n_control_combos=3,
                      n_control_reps=1, n_tokens_global=2 * SEQ_LEN, batch_size=2, seed=42)
    combos = draw_combinations(pool, m, n_combinations=3, seed=42)
    mbase = prepare_multi(target, MODULE, sae, plan, pool, combos, cfg,
                          device="cpu", cache_device="cpu")
    report = run_step_multi(target, model, MODULE, sae, mbase, cfg, step=0,
                            run_dir="/tmp/run", sae_dir="/tmp/sae", site=MODULE, device="cpu")

    by_combo: dict[int, dict[int, set[int]]] = {}
    for row in report.edits:
        if row.kind != "ranked" or row.block != "union":
            continue
        assert row.selection_size <= m * row.k
        assert row.sharing_rate == pytest.approx(1.0 - row.selection_size / (m * row.k))
        by_combo.setdefault(row.combo, {})[row.k] = set(row.selection)
    for per_k in by_combo.values():
        for small, large in zip(top_k[:-1], top_k[1:], strict=True):
            assert per_k[small] <= per_k[large], "the k sweep must ADD components, not swap them"

    # Every combination emits its union row plus one row per member.
    targets = [r for r in report.edits if r.block == "target" and r.kind == "ranked"]
    assert len(targets) == m * len(combos) * len(top_k)
    assert {r.feature_id for r in targets} == {j for combo in combos for j in combo}


# ------------------------------------------------------------------ 3. collateral and 4. weights


def test_collateral_excludes_the_whole_combination():
    """`summarize_group` drops every target's row; `summarize(exclude=...)` drops the same set."""
    targets = torch.tensor([0, 1])
    acc = DeltaAccumulator(F, targets)
    f_base = torch.zeros(2, F)
    f_edit = torch.zeros(2, F)
    f_edit[:, 0] = 1.0     # target 0 moved by 1 per position
    f_edit[:, 1] = 2.0     # target 1 moved by 2 per position
    f_edit[:, 5] = 0.5     # a true bystander
    pre = torch.zeros(2, 2)
    acc.add(f_base, f_edit, pre, pre)

    union = acc.summarize_group([0, 1], [0, 1])
    assert union["delta_abs"] == pytest.approx(3.0)          # summed over the combination
    assert union["collateral_abs"] == pytest.approx(0.5)     # the bystander alone

    own = acc.summarize(0, 0, exclude=[0, 1])
    assert own["delta_abs"] == pytest.approx(1.0)
    assert own["collateral_abs"] == pytest.approx(0.5), "the sibling must not count as damage"
    assert acc.summarize(0, 0)["collateral_abs"] == pytest.approx(2.5)


def test_union_prediction_is_support_weighted():
    """A member active on half of `A_J` must contribute half its own-`A_j` mean to the union."""
    target, model, _, sae, plan = _fixture(seed=4)
    density, support = _eligible_stats(target, sae, plan)
    sample = draw_sample(density=density, support=support, site=MODULE, fingerprint="fp",
                         n_features=2, seed=42, max_density=0.2, min_support=1,
                         n_tokens=plan.n_tokens_total)
    pool = draw_pool(density=density, support=support, site=MODULE, fingerprint="fp",
                     seed_features=sample.feature_ids, pool_size=5, seed=42,
                     max_density=0.2, min_support=1, n_tokens=plan.n_tokens_total)
    cfg = MultiConfig(m=2, setup="cond", top_k=(2,), n_combinations=2, n_control_combos=2,
                      n_control_reps=1, n_tokens_global=2 * SEQ_LEN, batch_size=2, seed=42)
    combos = draw_combinations(pool, 2, n_combinations=2, seed=42)
    mbase = prepare_multi(target, MODULE, sae, plan, pool, combos, cfg,
                          device="cpu", cache_device="cpu")
    report = run_step_multi(target, model, MODULE, sae, mbase, cfg, step=0,
                            run_dir="/tmp/run", sae_dir="/tmp/sae", site=MODULE, device="cpu")

    for union in [r for r in report.edits if r.kind == "ranked" and r.block == "union"]:
        members = [r for r in report.edits
                   if r.kind == "ranked" and r.block == "target"
                   and r.combo == union.combo and r.k == union.k]
        assert len(members) == 2
        n_union = float(mbase.combinations[union.combo].positions.numel())
        expected = sum(
            float(mbase.positions[r.feature_id].numel()) / n_union * r.predicted_delta_unit
            for r in members
        )
        assert union.predicted_delta_unit == pytest.approx(expected)


# ------------------------------------------------------------------ 5. the pool file


def test_pool_leads_with_the_sample_and_is_seed_deterministic(tmp_path):
    density = torch.full((30,), 1e-3, dtype=torch.float64)
    density[29] = 0.0                       # dead
    support = torch.full((30,), 500, dtype=torch.int64)
    kwargs = dict(density=density, support=support, site="s", fingerprint="fp",
                  seed_features=[3, 7, 11], pool_size=10, min_support=100,
                  max_density=0.2, n_tokens=1000)
    pool = draw_pool(seed=1, **kwargs)
    assert pool.feature_ids[:3] == [3, 7, 11]
    assert pool.n_seed_features == 3
    assert len(set(pool.feature_ids)) == 10
    assert 29 not in pool.feature_ids
    assert pool.feature_ids == draw_pool(seed=1, **kwargs).feature_ids
    assert pool.feature_ids != draw_pool(seed=2, **kwargs).feature_ids

    assert draw_combinations(pool, 1, n_combinations=3) == [[3], [7], [11]]
    combos = draw_combinations(pool, 4, n_combinations=5, seed=1)
    assert len(combos) == 5
    assert all(len(c) == 4 and sorted(c) == c and set(c) <= set(pool.feature_ids) for c in combos)
    assert combos == draw_combinations(pool, 4, n_combinations=5, seed=1)

    path = write_pool(pool, tmp_path / "feature_pool.json")
    assert isinstance(load_pool(path, fingerprint="fp", site="s"), FeaturePool)
    with pytest.raises(AssertionError, match="Feature ids are meaningless"):
        load_pool(path, fingerprint="other")


def test_pool_rejects_a_seed_feature_the_band_would_have_dropped():
    density = torch.full((30,), 1e-3, dtype=torch.float64)
    support = torch.full((30,), 500, dtype=torch.int64)
    support[4] = 2
    with pytest.raises(AssertionError, match="not eligible"):
        draw_pool(density=density, support=support, site="s", fingerprint="fp",
                  seed_features=[4], pool_size=5, min_support=100, max_density=0.2,
                  n_tokens=1000)


def test_defaults_follow_the_spec_grid():
    assert default_top_k(1) == (1, 5, 10, 20, 50)
    assert all(default_top_k(m) == (1, 5, 10) for m in (5, 10, 20, 50))
    assert default_control_combos(1, 50) == 50
    assert default_control_combos(10, 50) == 50
    assert default_control_combos(20, 50) == 25
    assert default_control_combos(50, 50) == 10


def test_shared_baseline_cache_addresses_the_same_rows_as_a_dense_one():
    """`cache_rows` is a memory optimization only -- it must not move a single baseline row."""
    cache = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    positions = torch.tensor([2, 5, 9])
    rows = torch.tensor([2, 5, 9])
    shared = MeasureGroup("s", positions, torch.tensor([0]), cache, cache_rows=rows)
    dense = MeasureGroup("d", positions, torch.tensor([0]), cache[rows])
    sel = torch.tensor([0, 2])
    torch.testing.assert_close(shared.baseline(sel), dense.baseline(sel))
    assert shared.n_positions == dense.n_positions == 3


# ------------------------------------------------------------------ 6. the figures


def _metrics(scale: float) -> dict[str, float]:
    """One token-set block, with every key `plots/attr_edit_multi.py` and `aggregate.py` read."""
    return {
        "delta_abs": 0.4 * scale, "delta_signed": -0.4 * scale, "delta_abs_preact": 0.5 * scale,
        "baseline_act": 1.0, "delta_relative": 0.4 * scale, "frac_active_baseline": 0.1,
        "frac_active_edited": 0.09, "death_rate": 0.02 * scale, "birth_rate": 0.001,
        "collateral_abs": 0.2 * scale, "collateral_l0": 30.0, "collateral_pr": 12.0,
        "localization": 0.6, "selectivity": 1.5, "n_positions": 1000.0,
    }


def _arm_report(m: int, setup: str, step: int) -> object:
    """A schema-correct report built through the REAL dataclasses and the real aggregator."""
    from aspd.eval.editing.multi import _aggregate, _faithfulness
    from aspd.eval.editing.report import MultiAttrEditReport, MultiEditRow

    rows = []
    for k in (1, 5, 10):
        for kind, reps in (("ranked", 1), ("random", 2), ("norm_matched", 2)):
            for rep in range(reps):
                for combo in range(3):
                    size = min(m * k, m * k - combo)
                    common = dict(
                        kind=kind, k=k, rep=rep, combo=combo, m=m,
                        selection_size=size, sharing_rate=1.0 - size / (m * k),
                        edit_norm=0.1 * k, edit_norm_rel=0.001 * k,
                        predicted_delta_unit=-0.4 * k / m, predicted_delta_gate=-0.3 * k / m,
                        score_share=0.5,
                    )
                    scale = k if kind == "ranked" else 0.1 * k
                    rows.append(MultiEditRow(
                        **common, block="union", feature_id=None,
                        on_a=_metrics(scale), on_global=_metrics(0.1 * scale),
                        selection=list(range(size)),
                    ))
                    if m > 1:
                        rows.append(MultiEditRow(
                            **common, block="target", feature_id=combo * m,
                            on_a=_metrics(scale / m), on_global=_metrics(0.1 * scale / m),
                        ))
    return MultiAttrEditReport(
        site="proj", run_dir="/tmp/run", sae_dir="/tmp/sae", step=step, module="proj",
        m=m, setup=setup, n_combinations=3, n_control_combos=3, n_components=64, n_latents=128,
        estimator="analytic", n_tokens=1000, n_tokens_global=200,
        combinations=[list(range(c * m, (c + 1) * m)) for c in range(3)],
        edits=rows, by_k=_aggregate(rows), faithfulness=_faithfulness(rows),
        structure={str(k): {"selection_size": float(m * k), "sharing_rate": 0.1,
                            "selection_overlap": 0.05, "rank_overlap_gate_vs_unit": 0.8,
                            "rank_overlap_cond_vs_global": 0.4} for k in (1, 5, 10)},
        meta={"config": {"n_control_reps": 2}},
    )


def test_writing_one_arm_over_another_fails_rather_than_clobbering(tmp_path):
    """`--out-dir` replaces the arm directory, so two arms can be aimed at one path."""
    from aspd.eval.editing.report import write_multi_report

    path = tmp_path / "attr_edit_multi_step100.json"
    write_multi_report(_arm_report(5, "cond", 100), path)
    with pytest.raises(AssertionError, match="m=5 cond"):
        write_multi_report(_arm_report(10, "cond", 100), path)
    with pytest.raises(AssertionError, match="m=5 cond"):
        write_multi_report(_arm_report(5, "global", 100), path)
    # Re-running the SAME arm is the ordinary overwrite and must stay allowed.
    write_multi_report(_arm_report(5, "cond", 100), path)
    assert json.loads(path.read_text())["m"] == 5


def test_arm_and_cross_arm_figures_render_every_arm(tmp_path):
    from aspd.eval.editing.report import write_multi_report
    from aspd.eval.plots import plot_attr_edit_multi_dir
    from aspd.eval.plots.editing_multi import headline, load_arms

    multi = tmp_path / "attr_edit_multi"
    arms = [(1, "cond"), (5, "cond"), (5, "global"), (50, "cond")]
    for m, setup in arms:
        for step in (100, 200):
            write_multi_report(
                _arm_report(m, setup, step),
                multi / f"m{m}_{setup}" / f"attr_edit_multi_step{step}.json",
            )
    # A stray directory must not be mistaken for an arm.
    (multi / "notanarm").mkdir()

    assert set(load_arms(multi)) == set(arms)
    written = [p.name for p in plot_attr_edit_multi_dir(multi)]
    assert written == [f"attr_edit_multi_m{m}_{s}.png" for m, s in arms] + \
        ["sweep_multi.json", "attr_edit_multi.png"]
    # The cross-arm join must carry every arm, at its LAST checkpoint.
    sweep = json.loads((multi / "sweep_multi.json").read_text())
    assert {(a["m"], a["setup"]) for a in sweep["arms"]} == set(arms)
    assert all(a["step"] == 200 for a in sweep["arms"])

    head = headline(multi)
    assert set(head) == {f"m{m}_{s}" for m, s in arms}
    assert head["m5_cond"]["k"] == 10.0
    assert "on_a_delta_abs" in head["m5_cond"] and "sharing_rate" in head["m5_cond"]


def test_aggregate_picks_up_the_multi_stage(tmp_path):
    from aspd.eval.editing.report import write_multi_report
    from aspd.eval.plots.aggregate import collect, plot_aggregate

    write_multi_report(
        _arm_report(5, "cond", 100),
        tmp_path / "attr_edit_multi" / "m5_cond" / "attr_edit_multi_step100.json",
    )
    stages = collect(tmp_path, None)
    assert "reason" not in stages["attr_edit_multi"]
    assert "m5_cond" in stages["attr_edit_multi"]["headline"]

    plot_aggregate(tmp_path, None, tmp_path / "eval_summary")
    summary = json.loads((tmp_path / "eval_summary" / "aggregate.json").read_text())
    assert "attr_edit_multi" not in summary["missing"]
    assert any(r["stage"] == "attr_edit_multi" and r["source"] == "m5_cond"
               for r in summary["rows"])


def test_an_arm_whose_edit_sweep_was_skipped_still_draws(tmp_path):
    """A rank-r arm reports an attribution table and no edits; the figure must say so, not crash."""
    from aspd.eval.editing.report import write_multi_report
    from aspd.eval.plots.editing_multi import plot_arm_dir

    report = _arm_report(5, "cond", 100)
    report.edits, report.by_k, report.faithfulness = [], {}, {}
    report.edit_skipped = "components have no rank-1 component weight"
    arm = tmp_path / "m5_cond"
    write_multi_report(report, arm / "attr_edit_multi_step100.json")
    assert [p.name for p in plot_arm_dir(arm)] == ["attr_edit_multi_m5_cond.png"]


# ------------------------------------------------------- 6. the cached module-input fast path


def _multi_fixture(seed: int, **cfg_kw):
    """Pool, combinations and baseline for a small `m = 2` arm."""
    target, model, comp, sae, plan = _fixture(seed=seed)
    density, support = _eligible_stats(target, sae, plan)
    pool = draw_pool(
        density=density, support=support, site=MODULE, fingerprint="fp",
        seed_features=[], pool_size=5, seed=42, max_density=0.2, min_support=1,
        n_tokens=plan.n_tokens_total,
    )
    cfg = MultiConfig(m=2, setup="cond", top_k=(1, 2), n_combinations=3, n_control_combos=3,
                      n_control_reps=2, n_tokens_global=2 * SEQ_LEN, batch_size=2, seed=42,
                      **cfg_kw)
    combos = draw_combinations(pool, 2, n_combinations=3, seed=42)
    base = prepare_multi(target, MODULE, sae, plan, pool, combos, cfg,
                         device="cpu", cache_device="cpu")
    return target, model, comp, sae, plan, cfg, base


def test_the_cached_path_is_bitwise_identical_to_the_hooked_forward():
    """The whole justification for the fast path, on real accumulators."""
    target, _, comp, sae, plan, cfg, base = _multi_fixture(seed=7)
    assert base.x_cache is not None
    assert tuple(base.x_cache.shape) == (N_SEQ, SEQ_LEN, D_IN)

    combo = base.combinations[0]
    delta_w = component_delta_weight(comp, torch.tensor([0, 1]))
    groups = [combo.group, *(base.feature_groups[j] for j in combo.feature_ids),
              base.global_group]

    common = dict(batch_size=cfg.batch_size, device="cpu")
    slow = measure_edit(target, MODULE, sae, plan, groups, delta_w, x_cache=None, **common)
    fast = measure_edit(target, MODULE, sae, plan, groups, delta_w,
                        x_cache=base.x_cache, **common)

    assert set(slow) == set(fast)
    checked = 0
    for name in slow:
        a, b = slow[name], fast[name]
        assert sorted(vars(a)) == sorted(vars(b)) and vars(a)
        for stat in sorted(vars(a)):
            x, y = getattr(a, stat), getattr(b, stat)
            if isinstance(x, Tensor):
                assert torch.equal(x, y), f"{name}.{stat} differs"
            else:
                assert x == y, f"{name}.{stat}: {x} != {y}"
            checked += 1
    assert checked >= 3 * len(slow), checked
    # Not vacuous: the edit has to actually move something, or equality is free.
    assert float(slow[combo.group.name].abs_sum.sum()) > 0


def test_the_cache_is_absent_when_the_flag_is_off_and_the_slow_path_still_runs():
    """`cache_module_inputs=False` is the escape hatch the parity failure message names."""
    target, model, _, sae, plan, cfg, base = _multi_fixture(seed=7, cache_module_inputs=False)
    assert base.x_cache is None
    report = run_step_multi(target, model, MODULE, sae, base, cfg, step=0,
                            run_dir="/tmp/run", sae_dir="/tmp/sae", site=MODULE, device="cpu")
    assert report.edits


def test_a_corrupted_input_cache_is_caught_by_the_per_run_parity_check():
    """The guard has to FAIL when the two paths disagree, or its passing means nothing."""
    target, model, _, sae, plan, cfg, base = _multi_fixture(seed=7)
    assert cfg.verify_cached_forward, "the check must be on by default or this proves nothing"
    base.x_cache = base.x_cache + 1.0

    with pytest.raises(AssertionError, match="cached-forward parity FAILED"):
        run_step_multi(target, model, MODULE, sae, base, cfg, step=0,
                       run_dir="/tmp/run", sae_dir="/tmp/sae", site=MODULE, device="cpu")


def test_the_defaults_turn_the_fast_path_and_its_check_on():
    """A cell submitted with no extra flags must get both. `verify_cached_forward` costs one
    forward out of ~180, which is the price of the equivalence being checked rather than argued.
    """
    cfg = MultiConfig(m=5)
    assert cfg.cache_module_inputs
    assert cfg.verify_cached_forward


def test_baseline_act_is_chunk_independent_to_fp32_precision():
    from aspd.eval.editing import multi as multi_mod

    target, _, _, sae, plan = _fixture(seed=3)
    density, support = _eligible_stats(target, sae, plan)
    pool = draw_pool(
        density=density, support=support, site=MODULE, fingerprint="fp",
        seed_features=[], pool_size=5, seed=42, max_density=0.2, min_support=1,
        n_tokens=plan.n_tokens_total,
    )
    cfg = MultiConfig(m=2, setup="cond", top_k=(1,), n_combinations=3, n_control_combos=3,
                      n_control_reps=1, n_tokens_global=2 * SEQ_LEN, batch_size=2, seed=42)
    combos = draw_combinations(pool, 2, n_combinations=3, seed=42)

    original = multi_mod._BASELINE_ACT_CHUNK
    acts = {}
    try:
        for chunk in (original, 7, 1):
            multi_mod._BASELINE_ACT_CHUNK = chunk
            base = prepare_multi(target, MODULE, sae, plan, pool, combos, cfg,
                                 device="cpu", cache_device="cpu")
            acts[chunk] = base.baseline_act
    finally:
        multi_mod._BASELINE_ACT_CHUNK = original

    assert acts[original], "fixture produced no baseline activations"
    for chunk in (7, 1):
        for j, want in acts[original].items():
            assert acts[chunk][j] == pytest.approx(want, rel=1e-6, abs=1e-9), (
                f"feature {j}: chunk {chunk} gave {acts[chunk][j]}, one block gave {want}"
            )
    # Not vacuous: at least one feature must have more positions than the smallest chunk.
    assert max(int(v.numel()) for v in prepare_multi(
        target, MODULE, sae, plan, pool, combos, cfg, device="cpu", cache_device="cpu"
    ).positions.values()) > 7


def test_attr_edit_cached_and_hooked_paths_give_the_same_report():
    from dataclasses import asdict

    target, model, _, sae, plan = _fixture(seed=11)
    density, support = _eligible_stats(target, sae, plan)
    sample = draw_sample(
        density=density, support=support, site=MODULE, fingerprint="fp", n_features=3,
        seed=42, max_density=0.2, min_support=1, n_tokens=plan.n_tokens_total,
    )
    reports = {}
    for cached in (True, False):
        cfg = AttrEditConfig(top_k=(1, 2), n_control_reps=2, n_tokens_global=2 * SEQ_LEN,
                             batch_size=2, seed=42, cache_module_inputs=cached)
        base = prepare(target, MODULE, sae, plan, sample, cfg, device="cpu", cache_device="cpu")
        assert (base.x_cache is not None) is cached
        reports[cached] = asdict(run_step(
            target, model, MODULE, sae, base, cfg, step=0, run_dir="/tmp/run",
            sae_dir="/tmp/sae", site=MODULE, device="cpu",
        ))
    # `meta.config` records the knob itself, so it is expected to differ and nothing else is.
    for r in reports.values():
        r.get("meta", {}).pop("config", None)
    assert reports[True]["edits"], "fixture produced no edit rows"
    assert reports[True] == reports[False]


def test_attr_edit_catches_a_corrupted_input_cache():
    """The guard must FAIL when the paths disagree, or its passing says nothing."""
    target, model, _, sae, plan = _fixture(seed=11)
    density, support = _eligible_stats(target, sae, plan)
    sample = draw_sample(
        density=density, support=support, site=MODULE, fingerprint="fp", n_features=3,
        seed=42, max_density=0.2, min_support=1, n_tokens=plan.n_tokens_total,
    )
    cfg = AttrEditConfig(top_k=(1,), n_control_reps=1, n_tokens_global=2 * SEQ_LEN,
                         batch_size=2, seed=42)
    assert cfg.verify_cached_forward, "the check must be on by default or this proves nothing"
    base = prepare(target, MODULE, sae, plan, sample, cfg, device="cpu", cache_device="cpu")
    base.x_cache = base.x_cache + 1.0
    with pytest.raises(AssertionError, match="cached-forward parity FAILED"):
        run_step(target, model, MODULE, sae, base, cfg, step=0, run_dir="/tmp/run",
                 sae_dir="/tmp/sae", site=MODULE, device="cpu")

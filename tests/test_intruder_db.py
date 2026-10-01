"""Interp (intruder score): component selection, donor pool, summaries and resume."""

import json

import pytest

pytest.importorskip("param_decomp_lab")

from param_decomp_lab.autointerp.providers import ChatResponse, LLMProvider
from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

from aspd.eval.intruder_db import (
    CI_STATS_FILE,
    STATS_FILE,
    ThresholdedHarvestDB,
    anchor_ci,
    anchor_index,
    anchor_value,
    ci_threshold_stats,
    component_scale,
    crop_to_window,
    eligible_density_entries,
    intruder_harvest_db,
    surviving_mask,
    threshold_stats,
)
from aspd.eval.judge import JudgeConfig

N_REAL = 2
N_TRIALS = 2


class FakeProvider(LLMProvider):
    """Always names example 1 as the intruder; counts calls."""

    def __init__(self) -> None:
        self.n_calls = 0

    async def chat(self, prompt, max_tokens, response_schema, timeout_ms) -> ChatResponse:
        self.n_calls += 1
        content = json.dumps({"intruder": 1, "reasoning": "fake"})
        return ChatResponse(content=content, input_tokens=1, output_tokens=1)

    async def get_pricing(self) -> tuple[float, float]:
        return (0.0, 0.0)

    async def close(self) -> None:
        pass


class StubTok:
    """`get_spans` is the only AppTokenizer method the prompt builder touches."""

    def get_spans(self, token_ids: list[int]) -> list[str]:
        return [f" t{i}" for i in token_ids]


def _component(key: str, layer: str, idx: int, density: float) -> ComponentData:
    pmi = ComponentTokenPMI(top=[(1, 1.0)], bottom=[(2, -1.0)])
    examples = [
        ActivationExample(
            token_ids=[10 + j, 20 + j, 30 + j],
            firings=[True, False, True],
            activations={"activation": [1.0, 0.0, 0.5]},
        )
        for j in range(N_REAL + 2)  # eligible: >= n_real + 1 examples
    ]
    return ComponentData(
        component_key=key,
        layer=layer,
        component_idx=idx,
        mean_activations={"activation": 0.5},
        firing_density=density,
        activation_examples=examples,
        input_token_pmi=pmi,
        output_token_pmi=pmi,
    )


def _make_db(tmp_path, n_a: int = 8, n_b: int = 6) -> HarvestDB:
    """Two key prefixes (sites), all densities within the default 0.05 tolerance."""
    db = HarvestDB(tmp_path / "harvest.db")
    for i in range(n_a):
        db.save_component(_component(f"a:{i}", "a", i, 0.50 + 0.001 * i))
    for i in range(n_b):
        db.save_component(_component(f"b:{i}", "b", i, 0.50 + 0.001 * i))
    # one ineligible component: too few examples
    starved = _component("a:99", "a", 99, 0.5)
    starved.activation_examples = starved.activation_examples[:1]
    db.save_component(starved)
    return db


def _run(tmp_path, **kw):
    provider = FakeProvider()
    defaults = dict(
        llm_config=JudgeConfig(),
        n_real=N_REAL,
        n_trials=N_TRIALS,
        n_subsample=5,
        seed=0,
        app_tok=StubTok(),
        provider=provider,
    )
    defaults.update(kw)
    path = intruder_harvest_db(tmp_path, "unused-tokenizer", **defaults)
    return json.loads(path.read_text()), provider


def test_eligibility_and_prefix_filter(tmp_path):
    db = _make_db(tmp_path)
    entries = eligible_density_entries(db, min_examples=N_REAL + 1)
    assert len(entries) == 14  # a:99 starved out
    a_only = eligible_density_entries(db, min_examples=N_REAL + 1, key_prefix="a:")
    assert {k for k, _ in a_only} == {f"a:{i}" for i in range(8)}
    db.close()


def test_seeded_subsample_is_deterministic_and_scored(tmp_path):
    _make_db(tmp_path).close()
    summary, provider = _run(tmp_path)
    assert len(summary["sampled_keys"]) == 5
    assert summary["sampled_keys"] == sorted(summary["sampled_keys"])
    assert provider.n_calls == 5 * N_TRIALS
    assert summary["n_scored"] == 5 and summary["missing_keys"] == []
    assert set(summary["scores"]) == set(summary["sampled_keys"])
    assert all(0.0 <= s <= 1.0 for s in summary["scores"].values())
    assert summary["mean"] is not None and summary["std"] is not None
    assert summary["config"]["n_trials"] == N_TRIALS and summary["config"]["seed"] == 0

    # a second, independent selection pass picks the identical keys
    other = tmp_path / "other"
    other.mkdir()
    _make_db(other).close()
    summary2, _ = _run(other)
    assert summary2["sampled_keys"] == summary["sampled_keys"]


def test_resume_makes_no_new_calls(tmp_path):
    _make_db(tmp_path).close()
    _run(tmp_path)
    summary, provider = _run(tmp_path)  # same seed/subsample -> everything already scored
    assert provider.n_calls == 0
    assert summary["n_scored"] == 5


def test_explicit_keys_override_sampling_and_validate(tmp_path):
    _make_db(tmp_path).close()
    summary, _ = _run(tmp_path, keys=["a:0", "a:1"])
    assert summary["sampled_keys"] == ["a:0", "a:1"]
    assert summary["config"]["keys_explicit"] is True
    with pytest.raises(AssertionError, match="not eligible"):
        _run(tmp_path, keys=["a:99"])  # starved component


def test_prefix_restricts_scored_set_and_donor_pool(tmp_path):
    _make_db(tmp_path).close()
    summary, _ = _run(tmp_path, key_prefix="a:")
    assert all(k.startswith("a:") for k in summary["sampled_keys"])
    assert summary["n_eligible"] == 8
    assert summary["n_donor_pool"] == 8  # same-site donors by default
    assert summary["config"]["key_prefix"] == "a:"

    wide, _ = _run(tmp_path, key_prefix="b:", restrict_donors_to_prefix=False)
    assert wide["n_donor_pool"] == 14  # full eligible population


def test_per_prefix_summaries_get_distinct_paths(tmp_path):
    _make_db(tmp_path).close()
    _run(tmp_path, key_prefix="a:")
    _run(tmp_path, key_prefix="b:")
    assert (tmp_path / "intruder_summary_a.json").exists()
    assert (tmp_path / "intruder_summary_b.json").exists()


def _long_example(peak: int, n: int = 100) -> ActivationExample:
    """`n` tokens, firing every 10th, with the strongest activation at position `peak`."""
    firings = [i % 10 == 0 for i in range(n)]
    assert firings[peak], "peak must be a firing position"
    acts = [1.0 if f else 0.0 for f in firings]
    acts[peak] = 5.0
    return ActivationExample(token_ids=list(range(n)), firings=firings, activations={"activation": acts})


def test_crop_centres_on_the_peak_firing_token():
    cropped = crop_to_window(_long_example(peak=50), 3)
    assert cropped.token_ids == [47, 48, 49, 50, 51, 52, 53]
    assert cropped.activations["activation"][3] == 5.0
    assert len(cropped.firings) == 7


def test_crop_clamps_at_both_edges_and_keeps_full_width():
    lo = crop_to_window(_long_example(peak=0), 3)
    assert lo.token_ids == [0, 1, 2, 3, 4, 5, 6]  # slid right, still 2*w+1 wide
    hi = crop_to_window(_long_example(peak=90, n=93), 3)
    assert hi.token_ids == [86, 87, 88, 89, 90, 91, 92]


def test_crop_is_a_noop_on_an_already_windowed_example():
    ex = _long_example(peak=50)
    assert crop_to_window(ex, 60) is ex  # 121-token window over a 100-token example


def test_window_scores_separately_from_the_uncropped_pass(tmp_path):
    db = HarvestDB(tmp_path / "harvest.db")
    for i in range(6):
        comp = _component(f"a:{i}", "a", i, 0.50 + 0.001 * i)
        comp.activation_examples = [_long_example(peak=10 * (j % 9)) for j in range(N_REAL + 2)]
        db.save_component(comp)
    db.close()

    full, prov_full = _run(tmp_path)
    windowed, prov_win = _run(tmp_path, window_tokens_per_side=3)
    assert prov_win.n_calls == prov_full.n_calls  # not resumed off the uncropped scores
    assert windowed["config"]["score_type"] == "intruder_w3"
    assert windowed["config"]["window_tokens_per_side"] == 3
    assert full["config"]["score_type"] == "intruder"
    assert (tmp_path / "intruder_summary.json").exists()
    assert (tmp_path / "intruder_summary_w3.json").exists()

    db = HarvestDB(tmp_path / "harvest.db", readonly=True)
    assert set(db.get_scores("intruder")) == set(db.get_scores("intruder_w3"))
    db.close()


# ---- CI-thresholded scoring -------------------------------------------------------------------

HALF = 2


def _ci_example(cis: list[float], *, tau0: float = 0.0) -> ActivationExample:
    """A window whose `firings` are `ci > tau0`, as `ParamDecompHarvestFn` writes them."""
    return ActivationExample(
        token_ids=list(range(len(cis))),
        firings=[v > tau0 for v in cis],
        activations={"causal_importance": list(cis), "component_activation": list(cis)},
    )


def _ci_component(key: str, examples: list[ActivationExample], density: float) -> ComponentData:
    pmi = ComponentTokenPMI(top=[(1, 1.0)], bottom=[(2, -1.0)])
    return ComponentData(
        component_key=key,
        layer="a",
        component_idx=int(key.split(":")[1]),
        mean_activations={"causal_importance": 0.5},
        firing_density=density,
        activation_examples=examples,
        input_token_pmi=pmi,
        output_token_pmi=pmi,
    )


def _ci_db(tmp_path, anchors_by_key: dict[str, list[float]], density: float = 0.5) -> HarvestDB:
    """One component per key, one 5-token window per given anchor CI."""
    from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig

    db = HarvestDB(tmp_path / "harvest.db")
    db.save_config(HarvestConfig(
        method_config=ParamDecompHarvestConfig(wandb_path="p-deadbeef", activation_threshold=0.0),
        activation_context_tokens_per_side=HALF,
    ))
    for key, anchors in anchors_by_key.items():
        db.save_component(_ci_component(
            key, [_ci_example([0.005, 0.005, a, 0.005, 0.005]) for a in anchors], density))
    return db


def test_anchor_index_finds_the_centre_of_a_full_window():
    assert anchor_index([False, False, True, False, False], HALF) == HALF


def test_anchor_index_follows_a_left_clip_to_the_front():
    # p = 1 < HALF, so the window is [0, 3]: 4 tokens, anchor at n - HALF - 1 = 1.
    assert anchor_index([False, True, False, False], HALF) == 1


def test_anchor_index_stays_at_the_centre_under_a_right_clip():
    assert anchor_index([False, False, True, False], HALF) == HALF


def test_anchor_index_gives_up_when_both_clips_are_consistent():
    """Two firing candidates, so window length cannot say which clip produced it."""
    assert anchor_index([False, True, True, False], HALF) is None


def test_anchor_index_gives_up_on_a_window_clipped_at_both_ends():
    assert anchor_index([True, False], HALF) is None


def test_ambiguous_anchor_falls_back_to_the_peak_and_says_so():
    example = _ci_example([0.9, 0.4])  # both-clipped: peak is 0.9, and it is a guess
    assert anchor_ci(example, HALF) == (0.9, True)
    assert anchor_ci(_ci_example([0.0, 0.0, 0.3, 0.0, 0.0]), HALF) == (0.3, False)


def test_threshold_keeps_the_examples_whose_anchor_survives(tmp_path):
    _ci_db(tmp_path, {"a:0": [0.9, 0.5, 0.05, 0.005, 0.002]}).close()
    stats = ci_threshold_stats(tmp_path, [0.01, 0.1], min_examples=1)
    db = ThresholdedHarvestDB(tmp_path / "harvest.db", 0.1, stats, readonly=True)
    anchors = [e.activations["causal_importance"][HALF]
               for e in db.get_component("a:0").activation_examples]
    assert anchors == [0.9, 0.5]
    db.close()


def test_threshold_rederives_the_mask_the_judge_is_shown(tmp_path):
    """The `[[[...]]]` brackets come from `firings`, so a threshold must move them too."""
    _ci_db(tmp_path, {"a:0": [0.9]}).close()
    stats = ci_threshold_stats(tmp_path, [0.01], min_examples=1)
    base = HarvestDB(tmp_path / "harvest.db", readonly=True)
    assert base.get_component("a:0").activation_examples[0].firings == [True] * 5
    base.close()

    db = ThresholdedHarvestDB(tmp_path / "harvest.db", 0.01, stats, readonly=True)
    # The 0.005 neighbours fired at tau = 0 and do not at 0.01; the anchor still does.
    assert db.get_component("a:0").activation_examples[0].firings == [
        False, False, True, False, False]
    db.close()


def test_threshold_scales_density_by_the_surviving_anchor_fraction(tmp_path):
    _ci_db(tmp_path, {"a:0": [0.9, 0.9, 0.005, 0.005]}, density=0.4).close()
    stats = ci_threshold_stats(tmp_path, [0.01], min_examples=1)
    assert stats["components"]["a:0"]["0.01"] == [2, pytest.approx(0.2)]
    db = ThresholdedHarvestDB(tmp_path / "harvest.db", 0.01, stats, readonly=True)
    assert db.get_component("a:0").firing_density == pytest.approx(0.2)
    assert db.get_component_densities(min_examples=1) == [("a:0", pytest.approx(0.2))]
    # and the component drops out of the donor pool once it cannot field enough examples
    assert db.get_component_densities(min_examples=3) == []
    db.close()


def test_stats_cache_is_reused_and_counts_guessed_anchors(tmp_path):
    _ci_db(tmp_path, {"a:0": [0.9], "a:1": [0.5]}).close()
    first = ci_threshold_stats(tmp_path, [0.01], min_examples=1)
    assert (tmp_path / CI_STATS_FILE).exists()
    assert first["n_examples"] == 2 and first["n_anchors_guessed"] == 0
    # A second call for a threshold already held returns the cache rather than re-reading blobs.
    (tmp_path / "harvest.db").rename(tmp_path / "moved.db")
    assert ci_threshold_stats(tmp_path, [0.01], min_examples=1)["created"] == first["created"]
    (tmp_path / "moved.db").rename(tmp_path / "harvest.db")
    # An unheld threshold forces the pass again.
    assert "0.2" in ci_threshold_stats(tmp_path, [0.2], min_examples=1)["thresholds"]


def _ci_run(tmp_path, **kw):
    """`_run` with a CI-harvest default: `min_examples` here is `n_real + 1 = 3`."""
    return _run(tmp_path, **kw)


def test_threshold_scores_under_its_own_score_type_and_leaves_tau0_alone(tmp_path):
    anchors = {f"a:{i}": [0.9, 0.9, 0.9, 0.005] for i in range(4)}
    _ci_db(tmp_path, anchors).close()

    plain, plain_provider = _ci_run(tmp_path)
    assert plain["config"]["score_type"] == "intruder" and plain["config"]["ci_threshold"] is None
    assert plain["n_scored"] == 4 and plain_provider.n_calls == 4 * N_TRIALS

    hi, hi_provider = _ci_run(tmp_path, ci_threshold=0.01)
    assert hi["config"]["score_type"] == "intruder_ci0.01"
    assert hi["config"]["ci_threshold"] == 0.01
    assert hi_provider.n_calls == 4 * N_TRIALS, "a threshold must not resume off tau = 0's scores"
    assert (tmp_path / "intruder_summary.json").exists()
    assert (tmp_path / "intruder_summary_ci0.01.json").exists()

    # tau = 0's file and scores are untouched by the thresholded pass.
    assert json.loads((tmp_path / "intruder_summary.json").read_text()) == plain
    db = HarvestDB(tmp_path / "harvest.db", readonly=True)
    assert set(db.get_scores("intruder")) == set(anchors)
    assert set(db.get_scores("intruder_ci0.01")) == set(anchors)
    db.close()


def test_a_zero_threshold_is_the_unthresholded_measurement(tmp_path):
    """`0.0` names the harvests' own threshold, so it must not become a second score_type."""
    _ci_db(tmp_path, {f"a:{i}": [0.9] * 4 for i in range(4)}).close()
    summary, _ = _ci_run(tmp_path, ci_threshold=0.0)
    assert summary["config"]["score_type"] == "intruder"
    assert summary["config"]["ci_threshold"] is None
    assert not (tmp_path / "intruder_summary_ci0.json").exists()


def test_a_threshold_that_empties_the_population_writes_a_null_mean_and_calls_no_judge(tmp_path):
    """The case that must be NaN rather than an exception: no component clears the threshold."""
    keys = [f"a:{i}" for i in range(4)]
    _ci_db(tmp_path, {k: [0.005] * 4 for k in keys}).close()
    summary, provider = _ci_run(tmp_path, keys=keys, ci_threshold=0.1)

    assert provider.n_calls == 0
    assert summary["mean"] is None and summary["std"] is None
    assert summary["n_scored"] == 0 and summary["sampled_keys"] == []
    assert summary["n_keys_requested"] == 4 and summary["n_keys_dropped"] == 4
    assert summary["n_eligible"] == 0 and summary["n_donor_pool"] == 0


def test_ineligible_keys_are_dropped_under_a_threshold_and_asserted_without_one(tmp_path):
    keys = ["a:0", "a:1"]
    # a:1 keeps one surviving example, below `n_real + 1 = 3`.
    _ci_db(tmp_path, {"a:0": [0.9] * 4, "a:1": [0.9, 0.005, 0.005, 0.005]}).close()
    summary, _ = _ci_run(tmp_path, keys=keys, ci_threshold=0.01)
    assert summary["sampled_keys"] == ["a:0"]
    assert summary["n_keys_requested"] == 2 and summary["n_keys_dropped"] == 1

    # Without a threshold the same situation is a caller error, and stays one.
    with pytest.raises(AssertionError, match="not eligible"):
        _ci_run(tmp_path, keys=[*keys, "a:99"])


def test_a_threshold_cannot_be_combined_with_window_cropping(tmp_path):
    _ci_db(tmp_path, {f"a:{i}": [0.9] * 4 for i in range(4)}).close()
    with pytest.raises(AssertionError, match="same axis"):
        _ci_run(tmp_path, ci_threshold=0.01, window_tokens_per_side=1)


# ---- the `act` criterion (eval_spec.md 7.8) ---------------------------------------------------


def _act_example(acts: list[float], firings: list[bool]) -> ActivationExample:
    """A window with its own gate mask and an ungated activation, as an ASPD harvest stores them."""
    return ActivationExample(
        token_ids=list(range(len(acts))),
        firings=firings,
        activations={"causal_importance": [1.0 if f else 0.0 for f in firings],
                     "component_activation": list(acts)},
    )


def test_act_scale_is_the_peak_magnitude_over_firing_positions_only():
    comp = _ci_component("a:0", [
        _act_example([0.1, 50.0, -2.0, 0.3, 0.1], [False, False, True, True, False]),
        _act_example([0.1, 0.2, 1.5, 0.1, 0.1], [False, False, True, False, False]),
    ], density=0.5)
    assert component_scale(comp, "act") == 2.0
    assert component_scale(comp, "ci") == 1.0


def test_act_never_switches_on_a_position_the_gate_left_off():
    """The `and f` in `surviving_mask`: activation and gate are different tensors on a gate arm."""
    ex = _act_example([9.0, 0.5, 4.0, 0.5, 9.0], [False, False, True, True, False])
    mask = surviving_mask(ex, "act", 0.1, scale=4.0)  # cut = 0.4
    assert mask == [False, False, True, True, False]  # the 9.0s stay off
    assert surviving_mask(ex, "act", 0.2, scale=4.0) == [False, False, True, False, False]


def test_act_thresholds_the_magnitude_so_negative_activations_count():
    """~half the firing positions on an ASPD run are negative; a signed cut would drop all of them."""
    ex = _act_example([0.0, 0.0, -3.0, 0.0, 0.0], [False, False, True, False, False])
    assert surviving_mask(ex, "act", 0.1, scale=3.0) == [False, False, True, False, False]
    assert anchor_value(ex, "act", HALF) == (3.0, False)


def test_act_threshold_is_relative_to_each_components_own_peak(tmp_path):
    """Same shape at 100x the scale must survive identically -- the point of a relative cut."""
    from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig

    db = HarvestDB(tmp_path / "harvest.db")
    db.save_config(HarvestConfig(
        method_config=ParamDecompHarvestConfig(wandb_path="p-deadbeef", activation_threshold=0.0),
        activation_context_tokens_per_side=HALF,
    ))
    shape = [1.0, 0.5, 0.05, 0.005]  # anchors, relative to a peak of 1.0
    for key, k in (("a:0", 1.0), ("a:1", 100.0)):
        db.save_component(_ci_component(key, [
            _act_example([0.0, 0.0, a * k, 0.0, 0.0], [False, False, True, False, False])
            for a in shape], density=0.4))
    db.close()
    stats = threshold_stats(tmp_path, [0.01, 0.1], criterion="act", min_examples=1)
    assert stats["criterion"] == "act"
    for key in ("a:0", "a:1"):
        assert stats["components"][key]["0.01"][0] == 3   # 1.0, 0.5, 0.05 survive; 0.005 does not
        assert stats["components"][key]["0.1"][0] == 2    # 1.0, 0.5
        assert stats["components"][key]["0.1"][1] == pytest.approx(0.4 * 2 / 4)
    assert stats["components"]["a:0"]["scale"] == 1.0
    assert stats["components"]["a:1"]["scale"] == 100.0


def test_act_and_ci_caches_are_separate_files(tmp_path):
    _ci_db(tmp_path, {"a:0": [0.9, 0.5]}).close()
    ci = threshold_stats(tmp_path, [0.1], criterion="ci", min_examples=1)
    act = threshold_stats(tmp_path, [0.1], criterion="act", min_examples=1)
    assert (tmp_path / STATS_FILE["ci"]).exists() and (tmp_path / STATS_FILE["act"]).exists()
    assert ci["criterion"] == "ci" and act["criterion"] == "act"
    # A `ci` cache must never be handed back for an `act` request, or vice versa.
    assert threshold_stats(tmp_path, [0.1], criterion="act", min_examples=1)["criterion"] == "act"


def test_a_zero_peak_component_survives_no_threshold(tmp_path):
    from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig

    db = HarvestDB(tmp_path / "harvest.db")
    db.save_config(HarvestConfig(
        method_config=ParamDecompHarvestConfig(wandb_path="p-deadbeef", activation_threshold=0.0),
        activation_context_tokens_per_side=HALF,
    ))
    db.save_component(_ci_component("a:0", [
        _act_example([0.0] * 5, [False, False, True, False, False]) for _ in range(3)], density=0.3))
    db.close()
    stats = threshold_stats(tmp_path, [0.01], criterion="act", min_examples=1)
    assert stats["n_zero_scale"] == 1
    assert stats["components"]["a:0"]["0.01"] == [0, 0.0]
    thr = ThresholdedHarvestDB(tmp_path / "harvest.db", 0.01, stats, readonly=True)
    assert thr.get_component("a:0").activation_examples == []
    thr.close()


def test_act_scores_under_its_own_score_type(tmp_path):
    _ci_db(tmp_path, {f"a:{i}": [0.9, 0.9, 0.9, 0.9] for i in range(4)}).close()
    summary, _ = _ci_run(tmp_path, ci_threshold=0.1, criterion="act")
    assert summary["config"]["score_type"] == "intruder_act0.1"
    assert summary["config"]["ci_criterion"] == "act"
    assert (tmp_path / "intruder_summary_act0.1.json").exists()
    assert not (tmp_path / "intruder_summary_ci0.1.json").exists()

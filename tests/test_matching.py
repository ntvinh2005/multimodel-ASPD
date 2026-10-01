"""Matching: the alignment estimands, pair selection and the judge's answer parsing."""

import torch

from aspd.eval.matching import mode_of, report_name
from aspd.eval.matching.alignment import (
    AlignmentAccumulator,
    InputAlignmentAccumulator,
    raw_footprint_matrix,
    raw_input_footprint_matrix,
)
from aspd.eval.matching.examples import ACT_THRESHOLD, Example, format_feature_examples
from aspd.eval.matching.judge import get_generation_prompts, parse_matching_predictions
from aspd.eval.matching.match import (
    chained_top1,
    eligible,
    random_control,
    subsample_components,
    top1,
)

F, C, D, T = 6, 4, 5, 40
F_IN = 7


class _FakeSae:
    def __init__(self, w_enc=None, w_dec=None):
        self.W_enc = w_enc
        self.W_dec = w_dec


def _in_acc() -> InputAlignmentAccumulator:
    return InputAlignmentAccumulator(
        n_input_features=F_IN, n_components=C, device=torch.device("cpu")
    )


def test_footprint_uses_the_raw_encoder_not_a_normalized_one():
    """Rescaling an encoder column must move that feature's row -- normalizing would hide it."""
    torch.manual_seed(0)
    w_enc = torch.randn(D, F, dtype=torch.float64)
    u = torch.randn(C, D, dtype=torch.float64)

    base = raw_footprint_matrix(_FakeSae(w_enc), u)
    rescaled = w_enc.clone()
    rescaled[:, 2] *= 7.0
    moved = raw_footprint_matrix(_FakeSae(rescaled), u)

    torch.testing.assert_close(moved[2], base[2] * 7.0)
    torch.testing.assert_close(moved[torch.arange(F) != 2], base[torch.arange(F) != 2])


def test_accumulator_matches_a_dense_reference():
    """`E_t[|ζ|]` and `E_{t:Λ_j}[|ζ|]` streamed in chunks == computed in one shot."""
    torch.manual_seed(1)
    zeta = torch.randn(T, C, dtype=torch.float32)
    active = torch.rand(T, F) > 0.5
    active[:, 3] = False  # a dead output feature

    acc = AlignmentAccumulator(n_features=F, n_components=C, device=torch.device("cpu"))
    for start in range(0, T, 7):
        acc.update(zeta[start : start + 7], active[start : start + 7])

    footprint = torch.randn(F, C)
    glob, cond = acc.alignments(footprint)

    expected_glob = zeta.abs().mean(dim=0)[:, None] * footprint.t()
    torch.testing.assert_close(glob, expected_glob, atol=1e-5, rtol=1e-4)

    for j in range(F):
        if not active[:, j].any():
            continue
        expected = zeta[active[:, j]].abs().mean(dim=0) * footprint[j]
        torch.testing.assert_close(cond[:, j], expected, atol=1e-5, rtol=1e-4)


def test_dead_output_feature_scores_zero_rather_than_a_fallback():
    """A feature that never fired has no conditional mean; it must not win an argmax."""
    torch.manual_seed(2)
    acc = AlignmentAccumulator(n_features=F, n_components=C, device=torch.device("cpu"))
    active = torch.ones(T, F, dtype=torch.bool)
    active[:, 3] = False
    acc.update(torch.randn(T, C), active)

    _, cond = acc.alignments(torch.full((F, C), 99.0))
    assert bool(acc.dead_output_features[3])
    torch.testing.assert_close(cond[:, 3], torch.zeros(C))


def test_input_footprint_is_v_against_the_decoder_rows():
    """`M_in[c, i] = <V[:, c], W_dec^in[i, :]>` -- the DECODER, and not transposed."""
    torch.manual_seed(0)
    v = torch.randn(D, C, dtype=torch.float64)
    w_dec = torch.randn(F_IN, D, dtype=torch.float64)

    footprint = raw_input_footprint_matrix(_FakeSae(w_dec=w_dec), v)

    assert footprint.shape == (C, F_IN)
    torch.testing.assert_close(footprint, v.t() @ w_dec.t())


def test_input_accumulator_matches_a_dense_reference():
    """`E_t[f_i]` and `E_{t:ζ_c≠0}[f_i]` streamed in chunks == computed in one shot."""
    torch.manual_seed(1)
    feats = torch.rand(T, F_IN) * (torch.rand(T, F_IN) > 0.5)
    fires = torch.rand(T, C) > 0.5
    fires[:, 2] = False  # a component that never fires

    acc = _in_acc()
    for start in range(0, T, 7):
        acc.update(feats[start : start + 7], fires[start : start + 7])

    footprint = torch.randn(C, F_IN)
    glob, cond = acc.alignments(footprint)

    torch.testing.assert_close(glob, feats.mean(dim=0)[None, :] * footprint, atol=1e-5, rtol=1e-4)
    for c in range(C):
        if not fires[:, c].any():
            torch.testing.assert_close(cond[c], torch.zeros(F_IN))
            continue
        torch.testing.assert_close(
            cond[c], feats[fires[:, c]].mean(dim=0) * footprint[c], atol=1e-5, rtol=1e-4
        )


def test_dead_input_feature_scores_zero_rather_than_a_fallback():
    """A feature that never fired must not win an argmax on a fabricated score."""
    acc = _in_acc()
    feats = torch.rand(T, F_IN)
    feats[:, 4] = 0.0
    acc.update(feats, torch.ones(T, C, dtype=torch.bool))

    glob, cond = acc.alignments(torch.full((C, F_IN), 99.0))
    assert bool(acc.dead_input_features[4])
    torch.testing.assert_close(cond[:, 4], torch.zeros(C))
    torch.testing.assert_close(glob[:, 4], torch.zeros(C))


def test_chained_top1_picks_the_argmax_on_each_leg_over_eligible_features_only():
    a_in = torch.tensor([[0.0, 4.0, 1.0], [7.0, 0.0, 0.0]])
    a_out = torch.tensor([[0.0, 9.0, 1.0, 2.0], [5.0, 0.0, 0.0, 0.0]])
    chosen = torch.tensor([0, 1])
    in_features = torch.tensor([0, 2])       # input feature 1 (c=0's max) is NOT eligible
    features = torch.tensor([0, 2, 3])       # output feature 1 (c=0's max) is NOT eligible

    pairing = chained_top1(a_in, a_out, chosen, in_features, features, name="t")

    assert pairing.pairs == [(2, 3), (0, 0)]
    assert pairing.components == [0, 1]
    assert pairing.input_scores == [1.0, 7.0]
    assert pairing.scores == [2.0, 5.0]


def test_identity_rate_counts_components_matched_to_their_own_input_latent():
    eye = torch.eye(C)
    pairing = chained_top1(
        eye, torch.randn(C, F), torch.arange(C), torch.arange(C), torch.arange(F), name="t"
    )
    assert pairing.identity_rate == 1.0


def test_eligibility_requires_all_three_sides_alive_and_judgeable():
    dead_c = torch.tensor([False, False, True])
    dead_f = torch.tensor([False, True, False])
    dead_in = torch.tensor([True, False, False])
    comps, feats, in_feats = eligible(dead_c, dead_f, {0, 2}, {0, 1, 2}, dead_in, {0, 1, 2})
    assert comps.tolist() == [0]  # 1 is not judgeable, 2 is dead
    assert feats.tolist() == [0, 2]
    assert in_feats.tolist() == [1, 2]


def test_c2o_eligibility_never_touches_the_input_side():
    """`c2o` does not read the input dictionary, so it must not require one to be alive."""
    comps, feats, in_feats = eligible(
        torch.tensor([False, True]), torch.tensor([False, False]), {0, 1}, {0, 1}
    )
    assert comps.tolist() == [0] and feats.tolist() == [0, 1] and in_feats is None


def test_c2o_top1_judges_the_component_against_its_matched_feature():
    alignment = torch.tensor([[0.0, 9.0, 1.0, 2.0], [5.0, 0.0, 0.0, 0.0]])
    chosen = torch.tensor([0, 1])
    features = torch.tensor([0, 2, 3])  # feature 1 (the global max for c=0) is NOT eligible

    pairing = top1(alignment, chosen, features, name="t")

    assert pairing.pairs == [(0, 3), (1, 0)]      # (component, output feature)
    assert pairing.components == [0, 1]
    assert pairing.input_scores == []              # no input leg in this mode
    assert pairing.identity_rate != pairing.identity_rate     # nan, not a tautological 1.0


def test_the_control_holds_feature_one_in_either_mode():
    torch.manual_seed(0)
    comps, feats, in_feats = torch.arange(40), torch.arange(20), torch.arange(15)
    chosen = subsample_components(comps, n_subsample=8, seed=0)

    c2o = top1(torch.randn(40, 20), chosen, feats, name="A_cond")
    i2o = chained_top1(torch.randn(40, 15), torch.randn(40, 20), chosen, in_feats, feats,
                       name="A_cond")

    for matched in (c2o, i2o):
        control = random_control(matched, feats, seed=0)
        assert [a for a, _ in control.pairs] == [a for a, _ in matched.pairs]
        assert control.components == matched.components


def test_report_names_and_scheme_round_trip():
    """The mode lives in the filename AND in `meta.scheme`; a pre-split report reads as c2o."""
    assert report_name("c2o", 250000) == "matching_c2o_step250000.json"
    assert report_name("i2o", 250000) == "matching_i2o_step250000.json"
    assert mode_of({}) == "c2o"
    assert mode_of({"scheme": "in_to_out"}) == "i2o"
    assert mode_of({"scheme": "i2o"}) == "i2o"


def test_all_three_pairings_are_judged_on_the_same_components():
    torch.manual_seed(0)
    comps, feats, in_feats = torch.arange(50), torch.arange(20), torch.arange(15)
    chosen = subsample_components(comps, n_subsample=8, seed=0)

    a = chained_top1(torch.randn(50, 15), torch.randn(50, 20), chosen, in_feats, feats,
                     name="A_glob")
    b = chained_top1(torch.randn(50, 15), torch.randn(50, 20), chosen, in_feats, feats,
                     name="A_cond")
    r = random_control(b, feats, seed=0)

    assert a.components == b.components == r.components == chosen.tolist()
    assert [i for i, _ in r.pairs] == [i for i, _ in b.pairs]  # the input side is HELD
    assert all(j in feats.tolist() for _, j in r.pairs)
    assert [j for _, j in r.pairs] != [j for _, j in b.pairs]


def test_subsample_is_deterministic_given_the_seed():
    comps = torch.arange(100)
    a = subsample_components(comps, n_subsample=10, seed=3)
    b = subsample_components(comps, n_subsample=10, seed=3)
    assert a.tolist() == b.tolist()
    assert subsample_components(comps, n_subsample=10, seed=4).tolist() != a.tolist()


def test_judge_parse_table():
    assert parse_matching_predictions("...\nANSWER: SIMILAR") == 3
    assert parse_matching_predictions("...\nANSWER: MAYBE") == 2
    assert parse_matching_predictions("...\nANSWER: DIFFERENT") == 1
    assert parse_matching_predictions('ANSWER: "similar"') == 3
    assert parse_matching_predictions("no answer block at all") == 1
    assert parse_matching_predictions("") == 1
    assert parse_matching_predictions("ANSWER: banana") == 1


def test_empty_side_renders_the_string_the_prompt_keys_on():
    """The rubric says: if either feature shows `NO EXAMPLE.`, answer DIFFERENT."""
    assert format_feature_examples([]) == "NO EXAMPLE."
    messages = get_generation_prompts([], [Example([" a"], [1.0])])
    assert "NO EXAMPLE." in messages[1]["content"]
    assert "you must output DIFFERENT" in messages[0]["content"]


def test_example_marks_only_tokens_above_threshold():
    ex = Example([" the", " cat", " sat"], [0.0, 5.0, ACT_THRESHOLD])
    assert ex.to_str(mark_toks=True) == " the<< cat>> sat"
    assert ex.to_str(mark_toks=False) == " the cat sat"


def test_a_cond_collapses_onto_a_glob_when_every_latent_reads_active():
    """Why `accumulate_alignment` asserts a nonzero JumpReLU threshold."""
    torch.manual_seed(3)
    zeta = torch.randn(T, C)
    acc = AlignmentAccumulator(n_features=F, n_components=C, device=torch.device("cpu"))
    acc.update(zeta, torch.ones(T, F, dtype=torch.bool))
    glob, cond = acc.alignments(torch.randn(F, C))
    torch.testing.assert_close(glob, cond, atol=1e-5, rtol=1e-4)


def test_row_blocking_changes_nothing_about_the_result():
    """The `[F, C]` traversals are row-blocked for memory; that must be invisible in the numbers."""
    torch.manual_seed(3)
    zeta = torch.randn(T, C)
    active = torch.rand(T, F) > 0.5
    active[:, 2] = False
    footprint = torch.randn(F, C)

    whole = AlignmentAccumulator(n_features=F, n_components=C, device=torch.device("cpu"))
    split = AlignmentAccumulator(n_features=F, n_components=C, device=torch.device("cpu"))
    assert whole.row_block >= F, "this test is vacuous unless the unsplit path is the default"
    split.row_block = 2
    for start in range(0, T, 7):
        whole.update(zeta[start : start + 7], active[start : start + 7])
        split.update(zeta[start : start + 7], active[start : start + 7])

    torch.testing.assert_close(split.cond_sum, whole.cond_sum, atol=1e-6, rtol=1e-6)
    for a, b in zip(split.alignments(footprint), whole.alignments(footprint), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_input_row_blocking_changes_nothing_about_the_result():
    """The `[C, F_in]` traversal is row-blocked for memory; that must be invisible in the numbers."""
    torch.manual_seed(3)
    feats = torch.rand(T, F_IN)
    fires = torch.rand(T, C) > 0.5
    footprint = torch.randn(C, F_IN)

    whole, split = _in_acc(), _in_acc()
    assert whole.row_block >= C, "this test is vacuous unless the unsplit path is the default"
    split.row_block = 1
    for start in range(0, T, 7):
        whole.update(feats[start : start + 7], fires[start : start + 7])
        split.update(feats[start : start + 7], fires[start : start + 7])

    torch.testing.assert_close(split.cond_sum, whole.cond_sum, atol=1e-6, rtol=1e-6)
    for a, b in zip(split.alignments(footprint), whole.alignments(footprint), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_alignments_returns_fp32_and_no_fp64_result():
    acc = AlignmentAccumulator(n_features=F, n_components=C, device=torch.device("cpu"))
    acc.update(torch.randn(T, C), torch.rand(T, F) > 0.5)
    glob, cond = acc.alignments(torch.randn(F, C))
    assert glob.dtype == torch.float32 and cond.dtype == torch.float32
    assert glob.shape == cond.shape == (C, F)


def test_the_two_modes_write_beside_each_other(tmp_path):
    """One file per mode per step: neither can overwrite the other's record."""
    import json

    from aspd.eval.matching.run import MatchingResult, write_report

    result = MatchingResult(
        name="A_cond", n_pairs=1, mean_score=2.0, score_histogram={1: 0, 2: 1, 3: 0},
        pairs=[(3, 7)], components=[5], identity_rate=0.0, scores=[2],
    )
    for mode in ("c2o", "i2o"):
        write_report([result], {"step": 50}, tmp_path, mode=mode)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "matching_c2o_step50.json", "matching_i2o_step50.json"
    ]
    for mode in ("c2o", "i2o"):
        data = json.loads((tmp_path / f"matching_{mode}_step50.json").read_text())
        assert data["meta"]["scheme"] == mode

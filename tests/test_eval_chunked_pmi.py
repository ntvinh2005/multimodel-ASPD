"""The chunked token-PMI ranker equals the dense computation at every chunk width."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from param_decomp_lab.harvest.sampling import compute_pmi, top_k_pmi

from aspd.eval.chunked_pmi import ChunkedTokenPmiRanker


def _reference(cooc, marginals, firings, total, top_k):
    return [
        top_k_pmi(
            cooccurrence_counts=cooc[c],
            marginal_counts=marginals,
            target_count=float(firings[c]),
            total_count=total,
            top_k=top_k,
        )
        for c in range(cooc.shape[0])
    ]


def _chunked(cooc, marginals, firings, total, top_k, chunk):
    c_n, vocab = cooc.shape
    ranker = ChunkedTokenPmiRanker(c_n, vocab, top_k, device="cpu")
    for start in range(0, vocab, chunk):
        stop = min(start + chunk, vocab)
        ranker.add_chunk(
            cooccurrence=cooc[:, start:stop],
            marginals=marginals[start:stop],
            firing_counts=firings,
            total_tokens=total,
            vocab_offset=start,
        )
    return [ranker.finalize(c) for c in range(c_n)]


def _assert_matches(got, want, cooc, marginals, firings, total, label):
    for c, ((g_top, g_bot), (w_top, w_bot)) in enumerate(zip(got, want, strict=True)):
        dense = compute_pmi(cooc[c], marginals, float(firings[c]), total)
        for side, g, w in (("top", g_top, w_top), ("bottom", g_bot, w_bot)):
            assert len(g) == len(w), f"{label} c={c} {side}: len {len(g)} != {len(w)}"
            gv = [v for _, v in g]
            wv = [v for _, v in w]
            assert gv == pytest.approx(wv, abs=0.0, rel=0.0), (
                f"{label} c={c} {side}: values differ\n got={gv}\nwant={wv}"
            )
            for idx, val in g:
                assert float(dense[idx]) == pytest.approx(val, abs=0.0, rel=0.0), (
                    f"{label} c={c} {side}: index {idx} carries {float(dense[idx])}, returned {val}"
                )


def _case(seed, n_components, vocab, density, top_k, zero_marginals=0):
    g = torch.Generator().manual_seed(seed)
    cooc = torch.poisson(torch.full((n_components, vocab), density), generator=g)
    marginals = torch.randint(1, 500, (vocab,), generator=g).float()
    if zero_marginals:
        marginals[torch.randperm(vocab, generator=g)[:zero_marginals]] = 0.0
    firings = cooc.sum(dim=1).clamp(min=1.0)
    total = int(marginals.sum().item())
    return cooc, marginals, firings, total, top_k


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 16, 64, 1000])
def test_matches_dense_at_every_chunk_width(chunk):
    """The pass multiplier must not change the answer -- including chunk=1 and chunk>vocab."""
    cooc, marginals, firings, total, top_k = _case(0, n_components=6, vocab=64, density=0.8, top_k=5)
    _assert_matches(
        _chunked(cooc, marginals, firings, total, top_k, chunk),
        _reference(cooc, marginals, firings, total, top_k),
        cooc, marginals, firings, total, f"chunk={chunk}",
    )


def test_sparse_components_have_empty_bottom_lists():
    """The subtle one. With most tokens never co-occurring, `n_invalid >= k`, and the dense path
    spends its whole `k` budget on `-inf` entries that it then filters -- so `bottom` is []. An
    implementation that returned the k smallest FINITE values here would look more useful and be
    wrong.
    """
    cooc, marginals, firings, total, top_k = _case(
        1, n_components=8, vocab=512, density=0.02, top_k=20
    )
    want = _reference(cooc, marginals, firings, total, top_k)
    assert any(len(b) == 0 for _, b in want), "fixture does not exercise the empty-bottom rule"
    _assert_matches(
        _chunked(cooc, marginals, firings, total, top_k, 37),
        want, cooc, marginals, firings, total, "sparse",
    )


def test_dense_components_have_nonempty_bottom_lists():
    """The complementary case: when nearly every token co-occurs, `n_invalid < k` and the bottom
    list holds exactly `k - n_invalid` entries.
    """
    cooc, marginals, firings, total, top_k = _case(
        2, n_components=4, vocab=48, density=6.0, top_k=10
    )
    want = _reference(cooc, marginals, firings, total, top_k)
    assert any(len(b) > 0 for _, b in want), "fixture does not exercise the nonempty-bottom rule"
    _assert_matches(
        _chunked(cooc, marginals, firings, total, top_k, 5),
        want, cooc, marginals, firings, total, "dense",
    )


def test_zero_marginal_tokens_are_invalid():
    """`valid = (cooc > 0) & (marginal > 0)`. A token the corpus never contained must never rank,
    even where `cooc` is somehow positive.
    """
    cooc, marginals, firings, total, top_k = _case(
        3, n_components=5, vocab=96, density=1.5, top_k=8, zero_marginals=30
    )
    _assert_matches(
        _chunked(cooc, marginals, firings, total, top_k, 11),
        _reference(cooc, marginals, firings, total, top_k),
        cooc, marginals, firings, total, "zero-marginals",
    )


def test_never_firing_component_returns_nothing():
    """`n_valid == 0` -> `k == 0` -> both lists empty, with no division by a zero firing count."""
    cooc, marginals, firings, total, top_k = _case(4, n_components=3, vocab=32, density=1.0, top_k=4)
    cooc[1] = 0.0
    firings = cooc.sum(dim=1).clamp(min=1.0)
    got = _chunked(cooc, marginals, firings, total, top_k, 8)
    assert got[1] == ([], [])
    _assert_matches(
        got, _reference(cooc, marginals, firings, total, top_k),
        cooc, marginals, firings, total, "dead-component",
    )


def test_top_k_larger_than_valid_set_is_clamped():
    """`k = min(top_k, n_valid)`: asking for more than exists returns what exists, not padding."""
    cooc, marginals, firings, total, _ = _case(5, n_components=4, vocab=40, density=0.05, top_k=0)
    top_k = 100
    _assert_matches(
        _chunked(cooc, marginals, firings, total, top_k, 9),
        _reference(cooc, marginals, firings, total, top_k),
        cooc, marginals, firings, total, "clamped-k",
    )


def test_indices_are_mapped_back_to_global_token_ids():
    """The offset test, on a TIE-FREE fixture so index equality is meaningful."""
    g = torch.Generator().manual_seed(9)
    n_components, vocab, top_k, chunk = 3, 50, 7, 6
    # Distinct counts against distinct marginals -> distinct ratios -> a unique correct ranking.
    cooc = torch.arange(1.0, n_components * vocab + 1.0).reshape(n_components, vocab)
    cooc = cooc[:, torch.randperm(vocab, generator=g)]
    marginals = torch.arange(1.0, vocab + 1.0)
    firings = cooc.sum(dim=1)
    total = int(marginals.sum().item())

    want = _reference(cooc, marginals, firings, total, top_k)
    # Self-validating: if the fixture ever ties, the assertion below would be testing nothing.
    for c, (w_top, _) in enumerate(want):
        vals = [v for _, v in w_top]
        assert len(set(vals)) == len(vals), f"fixture tied for component {c}; indices are ambiguous"

    got = _chunked(cooc, marginals, firings, total, top_k, chunk)
    for c, ((g_top, g_bot), (w_top, w_bot)) in enumerate(zip(got, want, strict=True)):
        assert [i for i, _ in g_top] == [i for i, _ in w_top], f"c={c} top indices"
        assert [i for i, _ in g_bot] == [i for i, _ in w_bot], f"c={c} bottom indices"

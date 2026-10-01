"""SAE reconstruction statistics, and that an identity splice leaves the model unchanged."""

import torch
from jaxtyping import Float
from torch import Tensor

from aspd.eval.dictionary import (
    DictionaryAdapter,
    sae_dictionaries_from_pair,
    transcoder_dictionary,
)
from aspd.eval.dictionary_report import reconstruction_stats, splice_ce_kl
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig
from aspd.sae.transcoder import (
    MatryoshkaBatchTopKTranscoder,
    MatryoshkaTranscoderConfig,
    evaluate_transcoder,
)

from ._toy_lm import toy_model_and_sites


def _warmed_sae(d_in: int, seed: int) -> MatryoshkaBatchTopKSAE:
    torch.manual_seed(seed)
    sae = MatryoshkaBatchTopKSAE(MatryoshkaSAEConfig(d_in=d_in, n_features=4 * d_in, top_k=4))
    for _ in range(15):
        sae.loss(torch.randn(64, d_in))
    return sae.freeze()


def _batches(model, seed: int = 5):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, model.vocab, (2, 8), generator=g)


def test_reconstruction_stats_are_well_formed():
    model, sites = toy_model_and_sites()
    saes = {sites.input_site: _warmed_sae(16, 1), sites.output_site: _warmed_sae(24, 2)}
    dicts = list(sae_dictionaries_from_pair(saes, sites).values())

    stats = reconstruction_stats(model, dicts, _batches(model), n_batches=4, device="cpu")
    assert {s.role for s in stats} == {"in", "out"}
    for s in stats:
        assert s.fvu >= 0.0 and 0.0 <= s.dead_frac <= 1.0 and s.mean_l0 >= 0.0
        # Matryoshka prefixes partition the dictionary and their active shares sum to 1.
        assert s.per_prefix and s.per_prefix[0].lo == 0 and s.per_prefix[-1].hi == s.n_features
        assert abs(sum(p.active_share for p in s.per_prefix) - 1.0) < 1e-4


class _IdentityDict(DictionaryAdapter):
    """encode = identity, decode = identity, so reconstruct(x) == x exactly."""

    def __init__(self, site: str, d: int) -> None:
        self.site_path = site
        self.role = "out"
        self._d = d

    @property
    def n_features(self) -> int:
        return self._d

    def encode(self, acts: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        return acts

    def decode(self, features: Float[Tensor, "... f"]) -> Float[Tensor, "... d"]:
        return features

    def decoder_rows(self) -> Float[Tensor, "f d"]:
        return torch.eye(self._d)

    def group_boundaries(self) -> None:
        return None


def test_identity_splice_is_behavior_neutral():
    model, sites = toy_model_and_sites()
    d_ff = model.h[0].mlp.c_fc.out_features
    ident = _IdentityDict(sites.output_site, d_ff)

    stats = splice_ce_kl(
        model, ident, _batches(model), logits_fn=model,
        n_batches=3, pad_id=0, device="cpu",
    )
    assert abs(stats.ce_clean - stats.ce_spliced) < 1e-5
    assert abs(stats.kl_spliced_vs_clean) < 1e-6


# ---- cross-site: a transcoder reads one site and reconstructs another --------------------------


def _warmed_transcoder(d_in: int, d_out: int, seed: int = 3) -> MatryoshkaBatchTopKTranscoder:
    torch.manual_seed(seed)
    tc = MatryoshkaBatchTopKTranscoder(
        MatryoshkaTranscoderConfig(d_in=d_in, n_features=4 * d_in, top_k=4, d_out=d_out)
    )
    for _ in range(15):
        tc.loss(torch.randn(64, d_in), torch.randn(64, d_out))
    return tc.freeze()


def _toy_transcoder_dictionary(seed: int = 3):
    model, sites = toy_model_and_sites()
    d_in = model.h[0].mlp.c_fc.in_features
    d_out = model.h[0].mlp.c_fc.out_features
    tc = _warmed_transcoder(d_in, d_out, seed)
    return model, sites, tc, transcoder_dictionary(tc, sites)


def test_a_transcoder_reads_and_writes_different_sites():
    """The adapter must SAY the two ends differ; every metric below branches on it."""
    _, sites, _, d = _toy_transcoder_dictionary()
    assert d.is_cross_site
    assert (d.hook, d.take) == (sites.input_site, sites.input_take)
    assert (d.write_hook, d.write_take) == (sites.output_site, "output")


def test_cross_site_fvu_is_scored_against_the_reconstructed_site():
    """Ties `reconstruction_stats` to `evaluate_transcoder`, which is the number of record."""
    model, sites, tc, d = _toy_transcoder_dictionary()

    stats = reconstruction_stats(model, [d], _batches(model), n_batches=4, device="cpu")[0]
    reference = evaluate_transcoder(
        model, sites, _batches(model), tc, n_batches=4, device="cpu"
    )

    assert stats.site == sites.output_site
    assert abs(stats.fvu - reference["fvu"]) < 1e-4
    assert abs(stats.mean_l0 - reference["mean_l0"]) < 1e-6
    assert abs(stats.dead_frac - reference["dead_frac"]) < 1e-9
    assert abs(sum(p.active_share for p in stats.per_prefix) - 1.0) < 1e-4


def test_cross_site_splice_replaces_the_written_site():
    """A transcoder splice must move the model, and by the amount its own reconstruction implies."""
    model, _, _, d = _toy_transcoder_dictionary()

    stats = splice_ce_kl(
        model, d, _batches(model), logits_fn=model,
        n_batches=3, pad_id=-1, device="cpu",
    )
    assert stats.n_tokens > 0
    assert stats.kl_spliced_vs_clean > 1e-6
    assert abs(stats.ce_spliced - stats.ce_clean) > 1e-4

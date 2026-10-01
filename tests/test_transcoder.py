"""The cross-site transcoder: an SAE dictionary reconstructing the other site."""

import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig
from aspd.sae.sites import gpt2_mlp_c_fc, gpt2_mlp_c_fc_resid
from aspd.sae.train import SITES_STAMP
from aspd.sae.transcoder import (
    MatryoshkaBatchTopKTranscoder,
    MatryoshkaTranscoderConfig,
    build_transcoder,
    checkpoint_name,
    evaluate_transcoder,
    load_transcoder,
    save_transcoder,
    stamped_input_take,
    train_transcoder,
    transcoder_steps,
)

N_EMBD, VOCAB = 32, 64
D_IN, D_OUT, N_FEAT = 8, 20, 32


@pytest.fixture
def model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    cfg = GPT2Config(
        n_embd=N_EMBD, n_layer=2, n_head=2, vocab_size=VOCAB, n_positions=32, n_ctx=32
    )
    m = GPT2LMHeadModel(cfg)
    m.eval()
    return m


def _tokens(n: int = 4, seq: int = 16):
    g = torch.Generator().manual_seed(1)
    while True:
        yield torch.randint(0, VOCAB, (n, seq), generator=g)


def _wrapped(m: GPT2LMHeadModel):

    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = m

        def forward(self, ids):
            return self.inner(input_ids=ids).logits

        def get_submodule(self, path):
            return self.inner.get_submodule(path)

    return _Wrapped()


def _cfg(**over) -> MatryoshkaTranscoderConfig:
    base = dict(
        d_in=D_IN,
        d_out=D_OUT,
        n_features=N_FEAT,
        group_fracs=(0.25, 0.25, 0.5),
        top_k=3,
        top_k_aux=4,
        n_batches_to_dead=2,
    )
    return MatryoshkaTranscoderConfig(**{**base, **over})


def _tc(**over) -> MatryoshkaBatchTopKTranscoder:
    torch.manual_seed(0)
    return MatryoshkaBatchTopKTranscoder(_cfg(**over))


# ---- shapes and construction -------------------------------------------------------------------


def test_shapes_are_asymmetric_and_the_two_biases_live_in_different_spaces():
    tc = _tc()
    assert tc.W_enc.shape == (D_IN, N_FEAT)
    assert tc.W_dec.shape == (N_FEAT, D_OUT)
    # `b_dec` keeps its name AND its job -- centring the encoder input. `b_out` is new.
    assert tc.b_dec.shape == (D_IN,)
    assert tc.b_out.shape == (D_OUT,)
    assert tc.d_out == D_OUT


def test_decoder_rows_start_unit_norm():
    """The tie `W_dec = normalize(W_enc.T)` cannot exist at d_in != d_out; the normalization is
    what survives it, and `normalize_decoder_` maintains it every step thereafter.
    """
    tc = _tc()
    torch.testing.assert_close(tc.W_dec.norm(dim=-1), torch.ones(N_FEAT), atol=1e-6, rtol=0)


def test_normalize_decoder_keeps_rows_unit_in_the_output_space():
    tc = _tc()
    with torch.no_grad():
        tc.W_dec.mul_(3.7)
    tc.normalize_decoder_()
    torch.testing.assert_close(tc.W_dec.norm(dim=-1), torch.ones(N_FEAT), atol=1e-6, rtol=0)


def test_decode_adds_b_out_not_b_dec():
    tc = _tc()
    with torch.no_grad():
        tc.b_out.fill_(2.0)
    f = torch.zeros(5, N_FEAT)
    torch.testing.assert_close(tc.decode(f), torch.full((5, D_OUT), 2.0))


# ---- the parent's objective, reused -------------------------------------------------------------


def test_degenerates_to_the_autoencoder_in_loss_and_gradients():
    """`d_out=None` must reproduce `MatryoshkaBatchTopKSAE` EXACTLY."""
    torch.manual_seed(0)
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(
            d_in=D_IN, n_features=N_FEAT, group_fracs=(0.25, 0.25, 0.5), top_k=3, top_k_aux=4,
            n_batches_to_dead=2,
        )
    )
    tc = MatryoshkaBatchTopKTranscoder(_cfg(d_out=None))
    tc.load_state_dict({**sae.state_dict(), "b_out": sae.b_dec.detach().clone()}, strict=True)

    x = torch.randn(16, D_IN)
    ref = sae.loss(x)
    got = tc.loss(x, x)  # the autoencoder IS the transcoder whose target is its own input
    for key in ("loss", "l2_loss", "aux_loss", "l1_norm", "l0_norm", "fvu", "n_dead"):
        torch.testing.assert_close(got[key], ref[key], msg=lambda m, k=key: f"{k}: {m}")

    ref["loss"].backward()
    got["loss"].backward()
    torch.testing.assert_close(tc.W_enc.grad, sae.W_enc.grad)
    torch.testing.assert_close(tc.W_dec.grad, sae.W_dec.grad)


def test_decoder_bias_indirection_leaves_the_autoencoder_untouched():
    """`_decoder_bias` is `b_dec` on an SAE. Pins the three lines added to `matryoshka.py`."""
    sae = MatryoshkaBatchTopKSAE(MatryoshkaSAEConfig(d_in=D_IN, n_features=N_FEAT, top_k=3))
    assert sae._decoder_bias is sae.b_dec
    f = torch.randn(4, N_FEAT)
    torch.testing.assert_close(sae.decode(f), f @ sae.W_dec + sae.b_dec)


def test_loss_refuses_a_shifted_pairing():
    tc = _tc()
    with pytest.raises(AssertionError, match="ONE forward"):
        tc.loss(torch.randn(8, D_IN), torch.randn(7, D_OUT))


def test_loss_refuses_the_wrong_output_width():
    tc = _tc()
    with pytest.raises(AssertionError):
        tc.loss(torch.randn(8, D_IN), torch.randn(8, D_OUT + 1))


def test_encode_and_loss_advances_the_threshold_ema_once():
    """Two independent encodes would double the EMA and the dead clock -- the parent's reason for
    `encode_and_loss` existing at all, which the transcoder inherits.
    """
    tc = _tc()
    x, y = torch.randn(16, D_IN), torch.randn(16, D_OUT)
    tc.loss(x, y)
    after_one = tc.threshold.clone()
    dead_after_one = tc.n_batches_not_active.clone()

    tc2 = _tc()
    acts, _ = tc2.encode_and_loss(x, y)
    torch.testing.assert_close(tc2.threshold, after_one)
    torch.testing.assert_close(tc2.n_batches_not_active, dead_after_one)
    assert acts.shape == (16, N_FEAT)


# ---- the identities the eval suite is built on --------------------------------------------------


def test_reconstruction_is_a_sum_of_per_latent_write_vectors():
    """`y_hat - b_out = sum_j f_j W_dec[j]`, which is what makes `W_dec` the write-vector matrix
    `scr_tpp`'s ablation subtracts (`a' = a - sum_{j in T} f_j W_dec[j]`).
    """
    tc = _tc()
    with torch.no_grad():
        tc.threshold.fill_(0.0)
    x = torch.randn(6, D_IN)
    f = tc.features(x)
    torch.testing.assert_close(tc.reconstruct(x) - tc.b_out, f @ tc.W_dec)


def test_weight_edit_removes_exactly_one_latents_contribution():
    tc = _tc()
    with torch.no_grad():
        tc.threshold.fill_(0.0)
    x = torch.randn(32, D_IN)
    f = tc.features(x)
    j = int(f.sum(0).argmax())
    active = f[:, j] > 0
    assert active.any(), "no token activates the busiest latent; the fixture is degenerate"

    dW_j = torch.outer(tc.W_enc[:, j], tc.W_dec[j])  # [d_in, d_out], the `x @ W` convention
    removed = (x - tc.b_dec) @ dW_j
    torch.testing.assert_close(removed[active], (f[active, j, None] * tc.W_dec[j]), atol=1e-5, rtol=1e-4)


def test_implied_weight_is_the_ungated_sum_of_those_outer_products():
    """`W_enc @ W_dec == sum_j W_enc[:, j] (x) W_dec[j]` -- the object compared to `W` in the
    report, and the transcoder's counterpart of VPD's `(VU)^T`.
    """
    tc = _tc()
    stacked = sum(torch.outer(tc.W_enc[:, j], tc.W_dec[j]) for j in range(N_FEAT))
    torch.testing.assert_close(tc.implied_weight(), stacked.float(), atol=1e-5, rtol=1e-4)


def test_reconstruct_is_a_function_of_the_token_alone():
    """`reconstruct` must take the JumpReLU path, never BatchTopK: every FVU in this project
    assumes `y_hat(x)` does not move with batch composition.
    """
    tc = _tc()
    with torch.no_grad():
        tc.threshold.fill_(0.05)
    x = torch.randn(12, D_IN)
    whole = tc.reconstruct(x)
    halves = torch.cat([tc.reconstruct(x[:5]), tc.reconstruct(x[5:])])
    torch.testing.assert_close(whole, halves)


# ---- build / train / persist --------------------------------------------------------------------


def test_build_reads_both_widths_off_a_real_forward(model):
    """`d_in`/`d_out` are discovered, never declared, and `F = multiplier * reference_width`."""
    wrapped = _wrapped(model)
    tc = build_transcoder(
        wrapped, gpt2_mlp_c_fc(0), next(_tokens()), feature_multiplier=4, device="cpu",
        top_k=3, group_fracs=(0.5, 0.5),
    )
    assert (tc.cfg.d_in, tc.d_out) == (N_EMBD, 4 * N_EMBD)
    assert tc.cfg.n_features == 4 * min(N_EMBD, 4 * N_EMBD) == 4 * N_EMBD


def test_build_refuses_a_dictionary_narrower_than_its_output_space(model):
    with pytest.raises(AssertionError, match="undercomplete"):
        build_transcoder(
            _wrapped(model), gpt2_mlp_c_fc(0), next(_tokens()), feature_multiplier=1,
            device="cpu", top_k=3, group_fracs=(0.5, 0.5),
        )


def test_training_refuses_zero_tokens(model):
    wrapped = _wrapped(model)
    sites = gpt2_mlp_c_fc(0)
    tc = build_transcoder(
        wrapped, sites, next(_tokens()), feature_multiplier=4, device="cpu", top_k=3,
        group_fracs=(0.5, 0.5),
    )
    with pytest.raises(AssertionError, match="n_tokens=0"):
        train_transcoder(wrapped, sites, _tokens(), tc, n_tokens=0, device="cpu")


def test_training_reduces_fvu_and_fires_the_checkpoint_hook(model, tmp_path):
    wrapped = _wrapped(model)
    sites = gpt2_mlp_c_fc(0)
    stream = _tokens()
    tc = build_transcoder(
        wrapped, sites, next(stream), feature_multiplier=4, device="cpu", top_k=4,
        group_fracs=(0.5, 0.5), n_batches_to_dead=1000,
    )
    before = evaluate_transcoder(wrapped, sites, _tokens(), tc, n_batches=2, device="cpu")["fvu"]

    seen: list[int] = []
    history = train_transcoder(
        wrapped, sites, stream, tc, n_tokens=4096, sae_batch_tokens=256, lr=3e-3,
        device="cpu", log_every=1, checkpoint_every_tokens=1024, on_checkpoint=seen.append,
    )
    after = evaluate_transcoder(wrapped, sites, _tokens(), tc, n_batches=2, device="cpu")["fvu"]

    assert after < before, (before, after)
    assert history and all(row["tokens"] > 0 for row in history)
    assert seen == [1024, 2048, 3072, 4096], seen


def test_evaluate_reports_the_centred_fvu_convention(model):
    """The denominator is `sum (y - mean_t y)^2` -- `evaluate_sae_pair`'s and
    `site_recon_stats`' convention, so the number lines up with the column it is read against.
    """
    wrapped = _wrapped(model)
    sites = gpt2_mlp_c_fc(0)
    tc = build_transcoder(
        wrapped, sites, next(_tokens()), feature_multiplier=4, device="cpu", top_k=4,
        group_fracs=(0.5, 0.5),
    )
    report = evaluate_transcoder(wrapped, sites, _tokens(), tc, n_batches=2, device="cpu")
    assert report["fvu"] > 0 and report["n_tokens"] == 2 * 4 * 16
    assert 0.0 <= report["dead_frac"] <= 1.0
    assert report["d_in"] == N_EMBD and report["d_out"] == 4 * N_EMBD


def test_save_load_round_trip_and_the_site_stamp(model, tmp_path):
    wrapped = _wrapped(model)
    sites = gpt2_mlp_c_fc_resid(0)
    tc = build_transcoder(
        wrapped, sites, next(_tokens()), feature_multiplier=4, device="cpu", top_k=3,
        group_fracs=(0.5, 0.5),
    )
    with torch.no_grad():
        tc.threshold.fill_(0.123)
    save_transcoder(tc, tmp_path, 50000, sites=sites)
    save_transcoder(tc, tmp_path, 250000, sites=sites)

    assert transcoder_steps(tmp_path) == [50000, 250000]
    assert (tmp_path / checkpoint_name(250000)).exists()
    assert json.loads((tmp_path / SITES_STAMP).read_text())["input_take"] == "input"
    assert stamped_input_take(tmp_path) == "input"

    loaded = load_transcoder(tmp_path, device="cpu")  # None => the LAST checkpoint
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    torch.testing.assert_close(loaded.W_dec, tc.W_dec)
    torch.testing.assert_close(loaded.b_out, tc.b_out)
    torch.testing.assert_close(loaded.threshold, tc.threshold)
    x = torch.randn(5, tc.cfg.d_in)
    torch.testing.assert_close(loaded.reconstruct(x), tc.reconstruct(x))


def test_an_unstamped_directory_is_refused_rather_than_assumed_adjacent(model, tmp_path):
    tc = _tc()
    save_transcoder(tc, tmp_path, 100)  # no sites => no stamp
    with pytest.raises(AssertionError, match="unrecoverable"):
        stamped_input_take(tmp_path)

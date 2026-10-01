"""The logit lens: the MLP-hidden lens continues the module's forward, the residual lens does not."""

import torch

from aspd.eval.dictionary import SAEDictionary
from aspd.eval.logit_lens import LogitLensRefs, decoder_logits
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

from ._toy_lm import toy_model_and_sites


def _warmed_sae(d_in: int, seed: int) -> MatryoshkaBatchTopKSAE:
    torch.manual_seed(seed)
    sae = MatryoshkaBatchTopKSAE(MatryoshkaSAEConfig(d_in=d_in, n_features=6 * d_in, top_k=4))
    for _ in range(10):
        sae.loss(torch.randn(64, d_in))
    return sae.freeze()


def _refs(model) -> LogitLensRefs:
    return LogitLensRefs(
        final_norm=model.final_norm,
        unembed=model.unembed.weight.t(),
        mlp_module=model.h[0].mlp,
        c_fc_module=model.h[0].mlp.c_fc,
        mlp_in_width=model.d_model,
    )


def test_output_role_continues_the_real_mlp_forward():
    """`decoder_logits` for an MLP-hidden dict == final_norm(c_proj(gelu(W_dec[i]))) @ W_U."""
    model, sites = toy_model_and_sites()
    d_ff = model.h[0].mlp.c_fc.out_features
    sae = _warmed_sae(d_ff, seed=3)
    d = SAEDictionary(sae, sites.output_site, "out")
    refs = _refs(model)

    got = decoder_logits(d, refs)

    mlp = model.h[0].mlp
    expected_resid = mlp.c_proj(mlp.act(sae.W_dec))  # continue the forward from the c_fc output
    expected = model.final_norm(expected_resid) @ model.unembed.weight.t()
    assert torch.allclose(got, expected, atol=1e-5)


def test_input_role_is_direct_unembed_no_mlp():
    model, sites = toy_model_and_sites()
    d_model = model.d_model
    sae = _warmed_sae(d_model, seed=4)
    d = SAEDictionary(sae, sites.input_site, "in")
    refs = _refs(model)

    got = decoder_logits(d, refs)
    expected = model.final_norm(sae.W_dec) @ model.unembed.weight.t()
    assert torch.allclose(got, expected, atol=1e-6)

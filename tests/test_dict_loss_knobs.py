"""The terms of the evaluation SAE's Matryoshka objective."""

import pytest
import torch

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

D_IN, F, N_TOKENS = 8, 64, 8
AUX_PENALTY = 1.0 / 32


@pytest.fixture
def sae() -> MatryoshkaBatchTopKSAE:
    """Configured so the AuxK term is LIVE, which takes arranging."""
    torch.manual_seed(0)
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(
            d_in=D_IN,
            n_features=F,
            group_fracs=(0.25, 0.75),
            top_k=2,
            top_k_aux=4,
            aux_penalty=AUX_PENALTY,
            n_batches_to_dead=1,
        )
    )
    sae.n_batches_not_active.fill_(999.0)
    return sae


@pytest.fixture
def stats(sae) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    out = sae.loss(torch.randn(N_TOKENS, D_IN))
    assert out["_aux_live"].detach().abs() > 0, "fixture must produce a LIVE AuxK term"
    return out


def test_the_two_live_halves_sum_to_the_fused_objective(stats):
    assert stats["_l2_live"].requires_grad and stats["_aux_live"].requires_grad
    torch.testing.assert_close(stats["_l2_live"] + stats["_aux_live"], stats["loss"])


def test_the_detached_halves_still_report_the_same_numbers(stats):
    """`l2_loss` / `aux_loss` are what every logger reads. The live pair is a gradient handle, not
    a second set of metrics, and the two must not drift apart.
    """
    torch.testing.assert_close(stats["_aux_live"].detach(), stats["aux_loss"])
    # `_l2_live` folds in the (inert, l1_coeff=0) L1 term, so it equals `l2_loss` here by design.
    torch.testing.assert_close(stats["_l2_live"].detach(), stats["l2_loss"])


def test_the_raw_auxk_is_the_scaled_one_divided_by_aux_penalty(stats):
    """**This is the identity that keeps `AuxKLoss(coeff=aux_penalty)` exact.**"""
    assert stats["_aux_raw_live"].requires_grad
    torch.testing.assert_close(AUX_PENALTY * stats["_aux_raw_live"], stats["_aux_live"])


def test_the_fused_objective_is_reconstructible_from_the_raw_pair(stats):
    """What the split config computes: `1.0*(l2 + l1) + aux_penalty*raw` must equal the fused
    scalar the un-split term returned.
    """
    torch.testing.assert_close(
        stats["_l2_live"] + AUX_PENALTY * stats["_aux_raw_live"], stats["loss"]
    )


def test_auxk_alone_still_reaches_the_decoder(sae, stats):
    """Dead-latent revival must survive being charged on its own, with no reconstruction term."""
    (grad,) = torch.autograd.grad(stats["_aux_raw_live"], sae.W_dec, retain_graph=True)
    assert grad.abs().sum() > 0, "an AuxK term that moves no gradient would log identically"


def test_auxk_reaches_the_ENCODER_through_the_pre_sparse_activations(sae, stats):
    """**The mechanism the TopK-SAE formulation depends on.**"""
    (grad,) = torch.autograd.grad(stats["_aux_raw_live"], sae.W_enc, retain_graph=True)
    assert grad.abs().sum() > 0, (
        "AuxK moved no gradient into the encoder -- it is being ranked by a post-sparse-function "
        "activation, which is ~0 on every dead latent"
    )


def test_the_l2_is_still_computed_when_it_is_not_charged(stats):
    """AuxK's residual is `x - x_hat`, so dropping the reconstruction from the OBJECTIVE must not
    drop it from the forward -- and `fvu` stays readable on an arm that is not optimizing it.
    """
    assert torch.isfinite(stats["fvu"]) and stats["fvu"] > 0
    assert stats["_l2_live"].detach() > 0

"""Activation pooling and masking of the evaluation sources."""

import einops
import torch

from aspd.eval.adapters.capture import keep_token_mask
from aspd.eval.adapters.source import pooled


class _Tok:
    def __init__(self, pad=None, eos=None, bos=None):
        self.pad_token_id, self.eos_token_id, self.bos_token_id = pad, eos, bos


def test_keep_mask_matches_upstream_bos_pad_eos_expression():
    tokens = torch.tensor([[50256, 5, 6, 7], [1, 2, 50256, 0]])
    tok = _Tok(pad=0, eos=50256, bos=50256)

    upstream_drop = (
        (tokens == tok.pad_token_id)
        | (tokens == tok.eos_token_id)
        | (tokens == tok.bos_token_id)
    )
    torch.testing.assert_close(keep_token_mask(tokens, tok), ~upstream_drop)


def test_keep_mask_ignores_undefined_ids():
    """Plain GPT-2 has no pad id; upstream would compare against `None`."""
    tokens = torch.tensor([[50256, 5, 6]])
    mask = keep_token_mask(tokens, _Tok(pad=None, eos=50256, bos=50256))
    torch.testing.assert_close(mask, torch.tensor([[False, True, True]]))


def test_pooled_matches_upstream_create_meaned_model_activations():
    """Upstream infers `N_b` from `Σ_d a == 0` after zeroing; we carry the mask. Same answer."""
    torch.manual_seed(0)
    acts = torch.randn(4, 7, 5, dtype=torch.float64)
    mask = torch.rand(4, 7) > 0.3
    mask[:, 0] = True  # never leave a row fully masked
    zeroed = acts * mask[:, :, None]

    activations_bl = einops.reduce(zeroed, "b l d -> b l", "sum")
    nonzero_bl = (activations_bl != 0.0).to(zeroed.dtype)
    nonzero_b = einops.reduce(nonzero_bl, "b l -> b", "sum")
    upstream = einops.reduce(zeroed, "b l d -> b d", "sum") / nonzero_b[:, None]

    torch.testing.assert_close(pooled(zeroed, mask), upstream)


def test_pooled_handles_a_latent_axis():
    """Used for `ᾱ_{b,c}` in the node effects, not just `[b, d]` activations."""
    torch.manual_seed(0)
    latents = torch.randn(3, 6, 11, dtype=torch.float64)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[1, 4:] = False
    out = pooled(latents, mask)
    assert out.shape == (3, 11)
    torch.testing.assert_close(out[1], latents[1, :4].mean(dim=0))

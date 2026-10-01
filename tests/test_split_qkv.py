"""Splitting GPT-2's fused Q/K/V matrix into three is exact."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.targets import SplitQKV, split_gpt2_qkv


def _tiny_gpt2(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=64, n_positions=32, n_embd=16, n_layer=3, n_head=4, resid_pdrop=0.0,
        embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


def test_split_reproduces_the_fused_projection_exactly() -> None:
    """`cat([q,k,v])(x)` equals `c_attn(x)` -- the substitution `GPT2Attention.forward` relies on."""
    model = _tiny_gpt2()
    x = torch.randn(2, 5, model.config.n_embd)
    fused_outs = [block.attn.c_attn(x) for block in model.transformer.h]

    split_gpt2_qkv(model)

    for i, (block, want) in enumerate(zip(model.transformer.h, fused_outs, strict=True)):
        got = block.attn.c_attn(x)
        assert got.shape == want.shape, f"block {i}: {got.shape} != {want.shape}"
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, msg=f"block {i}")


def test_split_model_logits_match_the_stock_model() -> None:
    """The end-to-end claim: splitting changes the module tree, never the function."""
    stock, split = _tiny_gpt2(), _tiny_gpt2()
    ids = torch.randint(0, stock.config.vocab_size, (2, 9))

    with torch.no_grad():
        want = stock(ids).logits
    split_gpt2_qkv(split)
    with torch.no_grad():
        got = split(ids).logits

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_split_order_is_q_k_v() -> None:
    """Order is `.split()`'s, not ours. Reversed q/v trains a decomposition whose "query"
    components are values, with nothing anywhere to indicate it.
    """
    model = _tiny_gpt2()
    d = model.config.n_embd
    fused = model.transformer.h[0].attn.c_attn
    want_q, want_k, want_v = (fused.weight[:, s * d : (s + 1) * d].clone() for s in range(3))

    split_gpt2_qkv(model)
    got = model.transformer.h[0].attn.c_attn

    torch.testing.assert_close(got.q_proj.weight, want_q)
    torch.testing.assert_close(got.k_proj.weight, want_k)
    torch.testing.assert_close(got.v_proj.weight, want_v)


def test_decomposition_target_paths_exist_and_are_conv1d_shaped() -> None:
    model = _tiny_gpt2()
    d = model.config.n_embd
    split_gpt2_qkv(model)

    names = dict(model.named_modules())
    for layer in range(model.config.n_layer):
        for proj in ("q_proj", "k_proj", "v_proj"):
            path = f"transformer.h.{layer}.attn.c_attn.{proj}"
            assert path in names, f"{path} missing; a config globbing it would match nothing"
            mod = names[path]
            assert (mod.nx, mod.nf) == (d, d), f"{path}: Conv1D({mod.nf}, {mod.nx}), want ({d}, {d})"
        assert isinstance(names[f"transformer.h.{layer}.attn.c_attn"], SplitQKV)


def test_parameter_count_is_preserved() -> None:
    """No parameters invented or dropped: the split is a re-layout, not a re-parameterization."""
    stock, split = _tiny_gpt2(), _tiny_gpt2()
    before = sum(p.numel() for p in stock.parameters())
    split_gpt2_qkv(split)
    assert sum(p.numel() for p in split.parameters()) == before


def test_refuses_a_head_pruned_block() -> None:
    """`prune_conv1d_layer` narrows `nf` and invalidates `split_size`; slicing anyway is silent."""
    model = _tiny_gpt2()
    model.transformer.h[1].attn.pruned_heads = {0}
    try:
        split_gpt2_qkv(model)
    except AssertionError as e:
        assert "pruned heads" in str(e)
    else:
        raise AssertionError("split_gpt2_qkv accepted a head-pruned block")

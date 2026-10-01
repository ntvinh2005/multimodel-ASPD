"""`share_encoders: false`: one encoder per matrix, each reading its matrix's residual site."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.ci import (
    ASPDCiConfig,
    ASPDCiFnSet,
)
from aspd.ci.aspd import make_aspd_ci_fn
from aspd.sites import group_modules_by_resid_site

C, N_EMBD, SEQ, VOCAB = 16, 8, 6, 32

SHARED = ["transformer.h.0.attn.c_proj", "transformer.h.0.mlp.c_fc"]
OTHER = ["transformer.h.1.attn.c_attn"]
MODULES = SHARED + OTHER
SITE_A, SITE_B = "transformer.h.0.ln_2", "transformer.h.1.ln_1"


def _model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB, n_positions=SEQ, n_embd=N_EMBD, n_layer=2, n_head=2)
    ).eval()
    model.requires_grad_(False)
    return model


def _sites() -> dict[str, str]:
    return {m: s for s, ms in group_modules_by_resid_site(MODULES).items() for m in ms}


def _cfg(**over) -> ASPDCiConfig:
    """Defaults that `over` may REPLACE rather than collide with -- `n_features` is overridden."""
    base = dict(
        n_features=C,
        top_k=2,
        pool_tokens=64,
        n_batches_to_dead=3,
        d_act=N_EMBD,
        resid_sites=_sites(),
    )
    return ASPDCiConfig(**{**base, **over})


class _Comp(torch.nn.Module):
    def __init__(self, d_in: int, c: int = C):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(d_in, c))
        self.U = torch.nn.Parameter(torch.randn(c, d_in))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))


def _d_in(model: GPT2LMHeadModel, module: str) -> int:
    from param_decomp.components import get_module_input_dim

    return get_module_input_dim(model.get_submodule(module))


def _built(model: GPT2LMHeadModel, **over) -> ASPDCiFnSet:
    fn = make_aspd_ci_fn(
        target_model=model, module_to_c={m: C for m in MODULES}, ci_config=_cfg(**over)
    )
    assert isinstance(fn, ASPDCiFnSet)
    return fn


@pytest.fixture()
def separated():
    model = _model()
    fn = _built(model, share_encoders=False)
    fn.attach_components({m: _Comp(_d_in(model, m)) for m in MODULES})
    yield model, fn
    fn.detach_resid_site()


def _acts(model: GPT2LMHeadModel, seed: int = 0):
    captured: dict[str, torch.Tensor] = {}
    handles = [
        model.get_submodule(m).register_forward_hook(
            lambda _mod, args, _out, m=m: captured.__setitem__(m, args[0].detach())
        )
        for m in MODULES
    ]
    try:
        with torch.no_grad():
            model(torch.randint(0, VOCAB, (2, SEQ), generator=torch.Generator().manual_seed(seed)))
    finally:
        for h in handles:
            h.remove()
    return captured


# ---- the default is untouched -------------------------------------------------------------------


def test_sharing_is_the_default():
    """A config that never mentions the field is the arm as shipped."""
    assert _cfg().share_encoders is True


def test_the_default_build_is_unchanged(request):
    """The shipped path, re-pinned HERE as well as in `test_aspd_ci_set.py`."""
    model = _model()
    fn = _built(model)
    request.addfinalizer(fn.detach_resid_site)
    assert sorted(fn.encoders()) == [SITE_A, SITE_B], "the default must key on the SITE"
    assert fn.fns()[SHARED[0]] is fn.fns()[SHARED[1]]
    assert fn.shared is True
    keys = set(fn.state_dict())
    assert "_encoders.transformer-h-0-ln_2.W_enc" in keys
    assert "_encoders.transformer-h-1-ln_1.W_dec" in keys


def test_an_existing_checkpoints_keys_still_load(request):
    model = _model()
    old = _built(model)
    request.addfinalizer(old.detach_resid_site)
    saved = {k: v.clone() for k, v in old.state_dict().items()}

    model2 = _model()
    fresh = _built(model2)
    request.addfinalizer(fresh.detach_resid_site)
    missing, unexpected = fresh.load_state_dict(saved, strict=True)
    assert not missing and not unexpected
    assert torch.equal(
        fresh.encoders()[SITE_A].W_enc, saved["_encoders.transformer-h-0-ln_2.W_enc"]
    )


# ---- the ablation --------------------------------------------------------------------------------


def test_separated_gives_one_encoder_per_matrix(separated):
    _, fn = separated
    assert sorted(fn.encoders()) == sorted(MODULES), "separated encoders key on the MODULE"
    assert fn.shared is False
    fns = fn.fns()
    assert fns[SHARED[0]] is not fns[SHARED[1]], (
        "the two matrices at one site must hold DIFFERENT encoders under share_encoders=False"
    )
    assert len({id(r) for r in fns.values()}) == len(MODULES)


def test_separated_encoders_still_read_their_own_residual_site(separated):
    """The flag moves parameters, never what the gate is a function of. Both encoders at `SITE_A`
    hook `SITE_A`; that is what keeps this an `r`-gated arm rather than an `x`-gated one.
    """
    _, fn = separated
    assert fn.sites() == {**{m: SITE_A for m in SHARED}, OTHER[0]: SITE_B}
    for module in SHARED:
        assert fn.fns()[module].site == SITE_A


def test_separated_state_dict_is_keyed_on_the_module(separated):
    _, fn = separated
    keys = set(fn.state_dict())
    assert "_encoders.transformer-h-0-attn-c_proj.W_enc" in keys
    assert "_encoders.transformer-h-0-mlp-c_fc.W_enc" in keys
    assert not any(k.startswith("_encoders.transformer-h-0-ln_2.") for k in keys), (
        "a site-keyed entry under share_encoders=False means the grouping did not change"
    )


def test_each_encoder_is_gated_once_per_forward(separated):
    model, fn = separated
    acts = _acts(model)
    fn.train()
    fn({m: acts[m] for m in MODULES})
    for module, encoder in fn.fns().items():
        assert encoder.n_batches_not_active.max() == 1, (
            f"{module}'s clock advanced {encoder.n_batches_not_active.max()} times in one forward"
        )


def test_matrices_at_one_site_no_longer_share_a_gate(separated):
    """The inverse of `test_matrices_at_one_site_get_the_same_gate`, and the reason `k` counts
    matrix-components on a separated run: component `c` is a different feature at each matrix.
    """
    model, fn = separated
    acts = _acts(model)
    out = fn({m: acts[m] for m in MODULES})
    assert not torch.equal(out[SHARED[0]], out[SHARED[1]]), (
        "two independently initialised encoders produced the identical gate; they are still shared"
    )


def test_separated_encoders_do_not_couple(separated):
    """Perturbing one encoder must move only its own matrix's gate, including at a site whose other
    matrix reads the identical tensor.
    """
    model, fn = separated
    acts = _acts(model)
    before = fn({m: acts[m] for m in MODULES})
    with torch.no_grad():
        fn.fns()[SHARED[0]].W_enc.mul_(-3.0)
    after = fn({m: acts[m] for m in MODULES})
    assert not torch.equal(before[SHARED[0]], after[SHARED[0]])
    assert torch.equal(before[SHARED[1]], after[SHARED[1]])
    assert torch.equal(before[OTHER[0]], after[OTHER[0]])


def test_separated_needs_no_width_agreement_within_a_site():
    """`C` must agree across a SHARED encoder's matrices -- declare different C, bound one-for-one. With
    one encoder per matrix the constraint is vacuous, and refusing it would block a legitimate run.
    """
    model = _model()
    fn = make_aspd_ci_fn(
        target_model=model,
        module_to_c={SHARED[0]: C, SHARED[1]: C * 2, OTHER[0]: C},
        ci_config=_cfg(share_encoders=False, n_features=None),
    )
    assert isinstance(fn, ASPDCiFnSet)
    assert fn.encoders()[SHARED[1]].n_features == C * 2
    assert fn.encoders()[SHARED[0]].n_features == C, "two encoders at one site, two widths"
    fn.detach_resid_site()


def test_shared_still_refuses_a_width_disagreement():
    model = _model()
    with pytest.raises(AssertionError, match="declare different C"):
        make_aspd_ci_fn(
            target_model=model,
            module_to_c={SHARED[0]: C, SHARED[1]: C * 2, OTHER[0]: C},
            ci_config=_cfg(),
        )

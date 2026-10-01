"""ASPD's shared encoder g^s: one encoder per residual site, shared by the matrices reading it."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.ci import (
    ASPDCiConfig,
    ASPDCiFn,
    ASPDCiFnSet,
    SharedEncoder,
    gate_for,
)
from aspd.ci.aspd import make_aspd_ci_fn
from aspd.sites import group_modules_by_resid_site

C, N_EMBD, SEQ, VOCAB = 16, 8, 6, 32

# Two sites, one of them shared by two matrices -- the shape the whole-model run has 24 of.
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


def _cfg(**over) -> ASPDCiConfig:
    return ASPDCiConfig(
        n_features=C,
        top_k=2,
        pool_tokens=64,
        n_batches_to_dead=3,
        d_act=N_EMBD,
        resid_sites=group_modules_by_site(),
        **over,
    )


def group_modules_by_site() -> dict[str, str]:
    """`{module: site}` from the shared registry -- the same map the config generator writes."""
    return {m: s for s, ms in group_modules_by_resid_site(MODULES).items() for m in ms}


class _Comp(torch.nn.Module):
    """Minimal stand-in for `TranscoderLinearComponents`: `attach_components` reads `V` and `b_dec`."""

    def __init__(self, d_in: int, c: int = C):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(d_in, c))
        self.U = torch.nn.Parameter(torch.randn(c, d_in))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))


def _d_in(model: GPT2LMHeadModel, module: str) -> int:
    from param_decomp.components import get_module_input_dim

    return get_module_input_dim(model.get_submodule(module))


def _built(model: GPT2LMHeadModel, cfg=None) -> ASPDCiFnSet:
    fn = make_aspd_ci_fn(
        target_model=model,
        module_to_c={m: C for m in MODULES},
        ci_config=cfg or _cfg(),
    )
    assert isinstance(fn, ASPDCiFnSet)
    return fn


@pytest.fixture()
def wired():
    model = _model()
    fn = _built(model)
    fn.attach_components({m: _Comp(_d_in(model, m)) for m in MODULES})
    yield model, fn
    fn.detach_resid_site()


def _acts(model: GPT2LMHeadModel, fn: ASPDCiFnSet, seed: int = 0):
    """Run the target so every encoder captures, and return each module's own input activation."""
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


# ---- shape ------------------------------------------------------------------------------------


def test_one_target_still_builds_the_bare_ci_fn():
    model = _model()
    fn = make_aspd_ci_fn(
        target_model=model,
        module_to_c={MODULES[1]: C},
        ci_config=ASPDCiConfig(
            n_features=C, top_k=2, pool_tokens=64, d_act=N_EMBD, resid_site=SITE_A
        ),
    )
    assert isinstance(fn, ASPDCiFn)
    assert not isinstance(fn, ASPDCiFnSet)
    # The keys three runs on disk carry, at the top level.
    assert {"W_enc", "b_dec", "W_dec"} <= set(fn.state_dict())
    fn.detach_resid_site()


def test_one_encoder_per_site_not_per_matrix(wired):
    _, fn = wired
    assert sorted(fn.encoders()) == [SITE_A, SITE_B]
    encoders = fn.fns()
    assert encoders[SHARED[0]] is encoders[SHARED[1]], "the two matrices at one site share a encoder"
    assert encoders[OTHER[0]] is not encoders[SHARED[0]]


def test_state_dict_is_keyed_on_the_site(wired):
    """`aspd.assemble` and `aspd.analysis.pairs.weights` both parse checkpoint keys by hand."""
    _, fn = wired
    keys = set(fn.state_dict())
    assert "_encoders.transformer-h-0-ln_2.W_enc" in keys
    assert "_encoders.transformer-h-1-ln_1.W_dec" in keys
    assert not any(k.startswith("_ci_fns.") for k in keys), (
        "no per-matrix gate state exists on this arm; a `_ci_fns.*` key means a encoder was "
        "duplicated per module"
    )


def test_gate_for_resolves_a_module_to_its_site_encoder(wired):
    """The accessor `metric.py`'s four lookups go through. Getting `None` back there would hand
    `_gate_implies_positive_preact` its `True` fall-through on the one arm where it is false.
    """
    _, fn = wired
    for module in MODULES:
        encoder = gate_for(fn, module)
        assert isinstance(encoder, SharedEncoder)
        assert encoder.gate_implies_positive_preact is False


# ---- what sharing must not break ---------------------------------------------------------------


def test_a_site_is_gated_once_per_forward(wired):
    """`gate` advances the threshold EMA and the dead clock. Two matrices at one site must not
    advance them twice: `n_batches_to_dead` is stated in POOLS, and a clock running at 2x revives
    latents that are not dead.
    """
    model, fn = wired
    acts = _acts(model, fn)
    fn.train()
    shared, other = fn.fns()[SHARED[0]], fn.fns()[OTHER[0]]
    assert shared.n_batches_not_active.max() == 0

    fn({m: acts[m] for m in MODULES})

    assert shared.n_batches_not_active.max() == 1, (
        f"the shared encoder's clock advanced {shared.n_batches_not_active.max()} times in one "
        "forward -- once per matrix instead of once per site"
    )
    assert other.n_batches_not_active.max() == 1


def test_matrices_at_one_site_get_the_same_gate(wired):
    """The point of sharing: component `c` names one feature at a site, not one per matrix."""
    model, fn = wired
    acts = _acts(model, fn)
    out = fn({m: acts[m] for m in MODULES})
    assert torch.equal(out[SHARED[0]], out[SHARED[1]])


def test_sites_do_not_couple(wired):
    """Sharing is per site and stops there -- BatchTopK's pool is one encoder's own latents."""
    model, fn = wired
    acts = _acts(model, fn)
    both = fn({m: acts[m] for m in MODULES})
    alone = fn({OTHER[0]: acts[OTHER[0]]})
    assert torch.equal(both[OTHER[0]], alone[OTHER[0]])


def test_a_partial_call_gates_only_what_it_was_asked_for(wired):
    model, fn = wired
    acts = _acts(model, fn)
    out = fn({SHARED[0]: acts[SHARED[0]]})
    assert set(out) == {SHARED[0]}


def test_an_unknown_module_is_refused(wired):
    model, fn = wired
    acts = _acts(model, fn)
    with pytest.raises(AssertionError, match="no encoder for"):
        fn({"transformer.h.0.mlp.c_proj": acts[SHARED[1]]})


# ---- what the builder refuses -------------------------------------------------------------------


def test_several_targets_without_a_site_map_are_refused():
    """The scalar `resid_site` would point every encoder at one block's norm -- shape-legal on a
    uniform model, and wrong in a way no logged number shows.
    """
    with pytest.raises(AssertionError, match="need `resid_sites"):
        make_aspd_ci_fn(
            target_model=_model(),
            module_to_c={m: C for m in MODULES},
            ci_config=ASPDCiConfig(
                n_features=C, top_k=2, pool_tokens=64, d_act=N_EMBD, resid_site=SITE_A
            ),
        )


def test_mismatched_widths_within_a_site_are_refused():
    """One encoder has ONE latent axis and every matrix there binds its components to it 1:1."""
    module_to_c = {m: C for m in MODULES}
    module_to_c[SHARED[1]] = C * 2
    with pytest.raises(AssertionError, match="declare different C"):
        make_aspd_ci_fn(
            target_model=_model(),
            module_to_c=module_to_c,
            ci_config=ASPDCiConfig(
                n_features=None,
                top_k=2,
                pool_tokens=64,
                d_act=N_EMBD,
                resid_sites=group_modules_by_site(),
            ),
        )


def test_a_site_map_that_misses_a_target_is_refused():
    sites = group_modules_by_site()
    del sites[OTHER[0]]
    with pytest.raises(AssertionError, match="must cover exactly"):
        make_aspd_ci_fn(
            target_model=_model(),
            module_to_c={m: C for m in MODULES},
            ci_config=ASPDCiConfig(
                n_features=C, top_k=2, pool_tokens=64, d_act=N_EMBD, resid_sites=sites
            ),
        )


def test_attach_components_checks_the_width_binding():
    model = _model()
    fn = _built(model)
    comps = {m: _Comp(_d_in(model, m)) for m in MODULES}
    comps[SHARED[0]] = _Comp(_d_in(model, SHARED[0]), c=C + 1)
    with pytest.raises(AssertionError, match="expected"):
        fn.attach_components(comps)
    fn.detach_resid_site()


# ---- lifetime -----------------------------------------------------------------------------------


def test_detach_removes_every_hook():
    """A leaked whole-model set is 24 hooks and 24 captures per stale checkpoint, not one."""
    model = _model()
    fn = _built(model)
    assert all(r._resid_handle is not None for r in fn.encoders().values())
    fn.detach_resid_site()
    assert all(r._resid_handle is None for r in fn.encoders().values())
    fn.detach_resid_site()  # idempotent


def test_the_set_is_registered_for_post_build_attach_and_for_detach():
    from aspd.ci.setup import _BY_CI_FN_TYPE, READ_SITE_DETACHERS

    assert ASPDCiFnSet in _BY_CI_FN_TYPE
    assert any(
        isinstance_check in (SharedEncoder, ASPDCiFnSet)
        for isinstance_check in READ_SITE_DETACHERS
    )
    assert ASPDCiFnSet in READ_SITE_DETACHERS

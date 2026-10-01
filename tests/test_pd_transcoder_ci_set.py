"""Model-wide PD Transcoder: one independent BatchTopK gate per matrix."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.ci import PDTranscoderCiConfig, PDTranscoderCiFn, PDTranscoderCiFnSet
from aspd.ci.pd_transcoder import make_transcoder_ci_fn

C = 16
MODULES = ["transformer.h.0.mlp.c_fc", "transformer.h.0.attn.c_proj"]


def _model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=1, n_head=2)
    ).eval()


def _cfg() -> PDTranscoderCiConfig:
    return PDTranscoderCiConfig(n_features=C, top_k=4, pool_tokens=8, n_batches_to_dead=3)


class _Comp(torch.nn.Module):
    """Minimal stand-in for `TranscoderLinearComponents`: the gate reads `V`, `b_dec` and the raw
    preact. `get_component_acts` is `(x - b_dec) @ V`, matching the real class exactly -- the
    centring is what makes `g_c * z_c == f_c` hold, so a stub that dropped it would test a
    different object.
    """

    def __init__(self, d_in: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = torch.nn.Parameter(torch.randn(d_in, C, generator=g))
        self.b_dec = torch.nn.Parameter(torch.randn(d_in, generator=g) * 0.1)

    def get_component_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.V


def test_one_target_still_builds_the_bare_ci_fn():
    fn = make_transcoder_ci_fn(
        target_model=_model(), module_to_c={MODULES[0]: C}, ci_config=_cfg()
    )
    assert isinstance(fn, PDTranscoderCiFn) and not isinstance(fn, PDTranscoderCiFnSet)


def test_several_targets_build_a_set_with_per_module_state():
    fns = make_transcoder_ci_fn(
        target_model=_model(), module_to_c=dict.fromkeys(MODULES, C), ci_config=_cfg()
    )
    assert isinstance(fns, PDTranscoderCiFnSet)
    assert sorted(fns.module_names) == sorted(MODULES)
    members = fns.fns()
    a, b = (members[m] for m in MODULES)
    assert a.n_batches_not_active is not b.n_batches_not_active
    assert a.threshold is not b.threshold


def test_the_set_matches_running_each_gate_alone():
    """THE equivalence the single-process plan rests on: identical `g`, module by module."""
    model = _model()
    d_in = {m: model.get_submodule(m).weight.shape[0] for m in MODULES}
    comps = {m: _Comp(d_in[m], seed=i) for i, m in enumerate(MODULES)}

    torch.manual_seed(0)
    acts = {m: torch.randn(2, 8, d_in[m]) for m in MODULES}

    joint = make_transcoder_ci_fn(
        target_model=model, module_to_c=dict.fromkeys(MODULES, C), ci_config=_cfg()
    )
    assert isinstance(joint, PDTranscoderCiFnSet)
    joint.attach_components(comps)
    joint_out = joint(acts)

    for m in MODULES:
        alone = make_transcoder_ci_fn(
            target_model=model, module_to_c={m: C}, ci_config=_cfg()
        )
        assert isinstance(alone, PDTranscoderCiFn)
        alone.attach_components(comps[m])
        assert torch.equal(joint_out[m], alone({m: acts[m]})[m]), m


def test_attach_refuses_a_module_mismatch():
    """A target in the model but not the CI fn would train UNGATED, which no metric would show."""
    model = _model()
    fns = make_transcoder_ci_fn(
        target_model=model, module_to_c=dict.fromkeys(MODULES, C), ci_config=_cfg()
    )
    assert isinstance(fns, PDTranscoderCiFnSet)
    d_in = {m: model.get_submodule(m).weight.shape[0] for m in MODULES}
    with pytest.raises(AssertionError, match="mismatch"):
        fns.attach_components({MODULES[0]: _Comp(d_in[MODULES[0]], seed=0)})


def test_the_set_is_registered_for_post_build_attach():
    from aspd.ci.setup import _BY_CI_FN_TYPE

    assert PDTranscoderCiFnSet in _BY_CI_FN_TYPE
    assert _BY_CI_FN_TYPE[PDTranscoderCiFnSet].attach is not None

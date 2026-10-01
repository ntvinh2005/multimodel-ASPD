"""The shared encoder reads the frozen target's residual stream on every forward of a step."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from param_decomp.component_model import ComponentModel
from param_decomp.configs import DecompositionTargetConfig
from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp.masks import make_mask_infos
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.capture_guard import (
    capture_armed,
    disarmed,
    install_clean_capture_guard,
)
from aspd.ci import ASPDCiConfig
from aspd.ci.aspd import make_aspd_ci_fn

C, N_EMBD, SEQ, VOCAB = 8, 8, 6, 32
MODULE = "transformer.h.0.mlp.c_fc"
GATE_SITE = "transformer.h.0.ln_2"


def _target() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB, n_positions=SEQ, n_embd=N_EMBD, n_layer=1, n_head=2)
    ).eval()
    model.requires_grad_(False)
    return model


def _observer(target: GPT2LMHeadModel):
    """A routed CI fn hooked on the target purely to watch what the capture sees."""
    cfg = ASPDCiConfig(
        n_features=C, top_k=2, pool_tokens=8, resid_site=GATE_SITE, d_act=N_EMBD
    )
    return make_aspd_ci_fn(
        target_model=target, module_to_c={MODULE: C}, ci_config=cfg
    )


def _component_model(target: GPT2LMHeadModel) -> ComponentModel:
    from param_decomp.ci_fns import LayerwiseCiConfig

    return ComponentModel(
        target_model=target,
        run_batch=lambda m, b: m(b).logits,
        decomposition_targets=resolve_decomposition_targets(
            target, [DecompositionTargetConfig(module_pattern=MODULE, C=C)]
        ),
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
    )


@pytest.fixture()
def wired():
    install_clean_capture_guard()
    target = _target()
    observer = _observer(target)
    yield target, observer, _component_model(target)
    observer.detach_resid_site()


def _batch(seed: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (2, SEQ), generator=torch.Generator().manual_seed(seed))


def _masks(model: ComponentModel, batch: torch.Tensor):
    with torch.no_grad():
        clean = model(batch, cache_type="input")
        ci = model.calc_causal_importances(pre_weight_acts=clean.cache, sampling="continuous")
    return make_mask_infos(ci.lower_leaky, weight_deltas_and_masks=None)


def test_armed_by_default():
    """Several entry points call the target directly and must capture --
    `calibrate_aspd_encoder` runs before a `ComponentModel` exists at all.
    """
    assert capture_armed()


def test_disarmed_nests_and_restores():
    with disarmed():
        assert not capture_armed()
        with disarmed():
            assert not capture_armed()
        assert not capture_armed()
    assert capture_armed()


def test_a_plain_forward_refreshes_the_capture(wired):
    target, observer, model = wired
    with torch.no_grad():
        model(_batch(1))
    first = observer._resid.clone()
    with torch.no_grad():
        model(_batch(2))
    assert not torch.allclose(first, observer._resid), (
        "a frozen-target forward on different tokens must update the capture; if it does not, the "
        "guard is disarming the one forward the encoder is supposed to read"
    )


def test_a_masked_forward_does_not_touch_the_capture(wired):
    """The whole point. `ComponentModel` runs the target several times per step with components
    substituted; every one fires the gate-site hook, and none of them may win.
    """
    target, observer, model = wired
    masks = _masks(model, _batch(2))
    with torch.no_grad():
        model(_batch(1))
    clean = observer._resid.clone()
    with torch.no_grad():
        model(_batch(2), mask_infos=masks)
    assert torch.equal(clean, observer._resid), (
        "a masked forward overwrote the capture -- the encoder would gate on the decomposition's "
        "own output rather than on the frozen model's residual stream"
    )


def test_the_capture_survives_a_masked_forward_on_the_same_tokens(wired):
    """The realistic shape: the masked forwards of a step run on the SAME batch, so a corrupted
    capture is close to the clean one and no shape or finiteness check would catch it.
    """
    target, observer, model = wired
    batch = _batch(3)
    with torch.no_grad():
        model(batch)
    clean = observer._resid.clone()
    with torch.no_grad():
        model(batch, mask_infos=_masks(model, batch))
    assert torch.equal(clean, observer._resid)


def test_the_guard_is_armed_again_after_the_masked_forward(wired):
    target, observer, model = wired
    batch = _batch(4)
    with torch.no_grad():
        model(batch, mask_infos=_masks(model, batch))
    assert capture_armed()


def test_the_installer_wraps_once(wired):
    """Every entry point installs at the top of `main` or at import, and several install twice via
    two different modules; a wrapper that stacked would nest `forward` several frames deep.
    """
    install_clean_capture_guard()
    first = ComponentModel.forward
    install_clean_capture_guard()
    install_clean_capture_guard()
    assert ComponentModel.forward is first


DEEP_LAYERS = 3


def _deep_target(n_layer: int = DEEP_LAYERS) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB, n_positions=SEQ, n_embd=N_EMBD, n_layer=n_layer, n_head=2)
    ).eval()
    model.requires_grad_(False)
    return model


def _observer_at(target: GPT2LMHeadModel, module: str, site: str):
    cfg = ASPDCiConfig(
        n_features=C, top_k=2, pool_tokens=8, resid_site=site, d_act=N_EMBD
    )
    return make_aspd_ci_fn(
        target_model=target, module_to_c={module: C}, ci_config=cfg
    )


def _component_model_for(target: GPT2LMHeadModel, module: str) -> ComponentModel:
    from param_decomp.ci_fns import LayerwiseCiConfig

    return ComponentModel(
        target_model=target,
        run_batch=lambda m, b: m(b).logits,
        decomposition_targets=resolve_decomposition_targets(
            target, [DecompositionTargetConfig(module_pattern=module, C=C)]
        ),
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
    )


def _probe(target: GPT2LMHeadModel, site: str):
    """Record `site`'s INPUT on every forward, guard or no guard -- the ground truth."""
    seen: dict[str, torch.Tensor] = {}

    def hook(_m, inputs, _o):
        seen["r"] = inputs[0].detach().clone()

    return seen, target.get_submodule(site).register_forward_hook(hook)


def _assert_encoder_ignores_the_perturbed_stream(module: str, site: str, n_layer: int) -> None:
    install_clean_capture_guard()
    target = _deep_target(n_layer)
    observer = _observer_at(target, module, site)
    model = _component_model_for(target, module)
    seen, probe_handle = _probe(target, site)
    try:
        batch = _batch(7)
        with torch.no_grad():
            clean = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=clean.cache, sampling="continuous"
            )
        clean_stream = seen["r"].clone()
        clean_capture = observer._resid.clone()

        zeroed = make_mask_infos({k: torch.zeros_like(v) for k, v in ci.lower_leaky.items()})
        with torch.no_grad():
            model(batch, mask_infos=zeroed)

        # NON-VACUITY: without this, the assertion below is about a tensor nothing could change.
        assert not torch.allclose(seen["r"], clean_stream, atol=1e-5), (
            f"{site} did not move when {module} was ablated, so this test cannot detect a "
            "corrupted capture. Pick a site genuinely downstream of the decomposed matrix."
        )
        assert torch.equal(observer._resid, clean_capture), (
            f"the encoder at {site} followed the decomposition's perturbed residual stream. Its "
            "gate would then drift with the decomposition's own error, and every loss is defined "
            "against the frozen model's gate."
        )
    finally:
        probe_handle.remove()
        observer.detach_resid_site()


def test_a_site_downstream_of_the_decomposed_block_stays_clean():
    """Across blocks: a layer-2 site under a layer-0 decomposition. The whole-model geometry."""
    _assert_encoder_ignores_the_perturbed_stream(
        module="transformer.h.0.mlp.c_fc", site="transformer.h.2.ln_2", n_layer=DEEP_LAYERS
    )


def test_the_attention_output_matrix_site_stays_clean():
    """Within one block: `attn.c_proj` writes INTO the `resid_mid` its own encoder reads, so its
    site is downstream of it even in a single-block model. This is the one entry in `RESID_SITES`
    with `runs_after=True`, and the only place the corruption is same-block.
    """
    _assert_encoder_ignores_the_perturbed_stream(
        module="transformer.h.0.attn.c_proj", site="transformer.h.0.ln_2", n_layer=1
    )

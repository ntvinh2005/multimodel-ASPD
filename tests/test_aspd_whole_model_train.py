"""Model-wide ASPD end to end on CPU: build, initialize, gate, backward, reload."""

import copy
from pathlib import Path

import pytest

pytest.importorskip("param_decomp_lab")

import torch
import yaml
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import resolve_decomposition_targets
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.arms import aspd_heads, resid_site_map
from aspd.ci import ASPDCiFnSet, SharedEncoder
from aspd.ci.setup import (
    attach_ci_fn,
    calibrate_ci_scale,
    detach_read_site,
    install_ci_fns,
    seed_ci_fn,
)
from aspd.component_setup import (
    _restore_stock_factory,
    install_component_parameterization,
)
from aspd.config import LMInterpExperimentConfig
from aspd.losses import (
    ActivationReconLoss,
    ActivationReconLossConfig,
    InternalReconLoss,
    InternalReconLossConfig,
)
from aspd.sites import group_modules_by_resid_site

BASE = (
    Path(__file__).resolve().parents[1] / "configs/gpt2/aspd.yaml"
)

C, N_EMBD, SEQ, VOCAB, LAYERS = 16, 8, 6, 32, 2
MATRICES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
MODULES = [f"transformer.h.{layer}.{m}" for layer in range(LAYERS) for m in MATRICES]
SITES = group_modules_by_resid_site(MODULES)


def _slug(module: str) -> str:
    return module.removeprefix("transformer.").replace(".", "_").replace("h_", "h", 1)


def _cfg() -> LMInterpExperimentConfig:
    """The shipped single-matrix ASPD config, widened and shrunk to this synthetic model."""
    raw = copy.deepcopy(yaml.safe_load(BASE.read_text()))
    pd = raw["pd"]
    pd["decomposition_targets"] = [{"module_pattern": m, "C": C} for m in MODULES]
    gate = pd["ci_config"]
    gate.update(
        n_features=C,
        top_k=2,
        pool_tokens=64,
        d_act=N_EMBD,
        resid_site="",
        resid_sites={m: s for s, ms in SITES.items() for m in ms},
    )

    entries = [
        e
        for e in pd["loss_metrics"]
        if e["type"] not in ("InternalReconLoss", "ActivationReconLoss", "AuxKLoss")
    ]
    head_s = next(
        e for e in pd["loss_metrics"] if e["type"] == "InternalReconLoss" and e["mode"] == "fvu"
    )
    head_r = next(e for e in pd["loss_metrics"] if e["type"] == "ActivationReconLoss")
    auxk = next(e for e in pd["loss_metrics"] if e["type"] == "AuxKLoss")
    for module in MODULES:
        entries.append({**copy.deepcopy(head_s), "name": f"ci_recon_{_slug(module)}",
                        "module": module})
    for members in SITES.values():
        rep = members[0]
        entries.append({**copy.deepcopy(head_r), "name": f"gate_recon_{_slug(rep)}",
                        "module": rep})
        entries.append({**copy.deepcopy(auxk), "name": f"auxk_{_slug(rep)}", "module": rep,
                        "recon": f"gate_recon_{_slug(rep)}"})
    pd["loss_metrics"] = entries
    return LMInterpExperimentConfig.model_validate(raw)


def _target() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB, n_positions=SEQ, n_embd=N_EMBD, n_layer=LAYERS, n_head=2)
    ).eval()
    model.requires_grad_(False)
    return model


def _batches(n: int = 16):
    g = torch.Generator().manual_seed(7)
    for _ in range(n):
        yield torch.randint(0, VOCAB, (2, SEQ), generator=g)


def _build(cfg, target) -> ComponentModel:
    """`aspd.run`'s construction order, with `Trainer` reduced to what it does to the model."""
    install_ci_fns()
    install_component_parameterization(cfg)
    return ComponentModel(
        target_model=target,
        run_batch=lambda m, b: m(b).logits,
        decomposition_targets=resolve_decomposition_targets(target, cfg.pd.decomposition_targets),
        ci_config=cfg.pd.ci_config,
        sigmoid_type=cfg.pd.sigmoid_type,
    )


@pytest.fixture()
def wired():
    cfg = _cfg()
    target = _target()
    model = _build(cfg, target)
    scale, _stats = calibrate_ci_scale(cfg, target, MODULES[0], _batches(), "cpu")
    seed_ci_fn(cfg, model, scale)
    yield cfg, target, model, scale
    detach_read_site(model)
    _restore_stock_factory()


def _clean(model: ComponentModel, batch: torch.Tensor):
    with torch.no_grad():
        out = model(batch, cache_type="input")
        ci = model.calc_causal_importances(pre_weight_acts=out.cache, sampling="continuous")
    return out.cache, ci


# ---- build and wiring --------------------------------------------------------------------------


def test_the_config_describes_four_encoders_over_eight_matrices():
    cfg = _cfg()
    assert len(resid_site_map(cfg)) == 8
    heads_r, heads_s = aspd_heads(cfg)
    assert len(heads_r) == 4 and len(heads_s) == 8


def test_the_registry_builds_the_shared_set_and_ties_it(wired):
    _cfg_, _target_, model, _scale = wired
    assert isinstance(model.ci_fn, ASPDCiFnSet)
    assert len(model.ci_fn.encoders()) == 4
    # The tie is what a state dict cannot carry, and what raises on the first forward if missed.
    assert set(model.ci_fn.components) == set(MODULES)


def test_initialization_centres_each_encoder_on_its_own_stream(wired):
    """One `E[r]` per site, not one shared. `resid_pre` at block 0 and `resid_mid` at block 1 are
    not remotely the same tensor, and a shared mean would leave three encoders selecting on an
    uncentred stream from step 0.
    """
    _cfg_, _target_, model, scale = wired
    assert set(scale) == set(SITES)
    for site, encoder in model.ci_fn.encoders().items():
        assert torch.equal(encoder.b_dec, scale[site])
    means = torch.stack([scale[s] for s in sorted(scale)])
    assert not torch.allclose(means[0], means[-1]), "two sites got the same mean"


def test_the_encoder_is_seeded_unit_norm_and_tied(wired):
    _cfg_, _target_, model, _scale = wired
    for encoder in model.ci_fn.encoders().values():
        assert torch.allclose(
            encoder.W_enc.norm(dim=0), torch.ones(C), atol=1e-5
        ), "W_enc columns are not unit-norm after seeding"
        assert torch.allclose(encoder.W_dec.norm(dim=-1), torch.ones(C), atol=1e-5)


# ---- the forward -------------------------------------------------------------------------------


def test_causal_importances_cover_every_matrix_and_are_shared_within_a_site(wired):
    _cfg_, _target_, model, _scale = wired
    _cache, ci = _clean(model, next(_batches(1)))
    assert set(ci.lower_leaky) == set(MODULES)
    for members in SITES.values():
        first = ci.lower_leaky[members[0]]
        for other in members[1:]:
            assert torch.equal(first, ci.lower_leaky[other]), (
                "matrices at one site must be gated by one encoder; component `c` names one feature "
                "there, not one per matrix"
            )


def test_the_gate_is_binary_and_at_the_configured_l0(wired):
    _cfg_, _target_, model, _scale = wired
    _cache, ci = _clean(model, next(_batches(1)))
    g = ci.lower_leaky[MODULES[0]]
    assert set(g.unique().tolist()) <= {0.0, 1.0}


# ---- the two heads -----------------------------------------------------------------------------


def _bind(metric_cls, cfg_cls, model, **fields):
    metric = metric_cls(cfg_cls(train_diag_every=1, **fields))
    metric.bind(model=model, device="cpu")
    metric.reset()
    return metric


def _context(model: ComponentModel, batch: torch.Tensor):
    from param_decomp.metrics.context import MetricContext

    cache, ci = _clean(model, batch)
    return MetricContext(
        model=model,
        batch=batch,
        target_out=model(batch),
        pre_weight_acts=cache,
        ci=ci,
        weight_deltas=model.calc_weight_deltas(),
        step=1,
        total_steps=4,
        use_delta_component=False,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=lambda p, t: (p.sub(t).pow(2).sum(), p.numel()),
        is_eval=False,
    )


def test_l_act_reaches_its_own_encoder_and_nothing_else(wired):
    """The arm's whole claim, at whole-model scale: L_act's gradient must reach `W_enc`/`b_dec`/
    `W_dec` of ONE encoder, and neither the components nor any other encoder.
    """
    cfg, _target_, model, _scale = wired
    site = sorted(SITES)[0]
    rep = SITES[site][0]
    metric = _bind(ActivationReconLoss, ActivationReconLossConfig, model, name="head_r",
                   module=rep, coeff=1.0, warmup=False)
    assert isinstance(metric.ci_fn, SharedEncoder)

    model.zero_grad(set_to_none=True)
    metric.update(_context(model, next(_batches(1)))).backward()

    mine = model.ci_fn.encoders()[site]
    assert mine.W_enc.grad is not None and mine.W_enc.grad.abs().sum() > 0
    assert mine.W_dec.grad is not None and mine.W_dec.grad.abs().sum() > 0
    for other_site, other in model.ci_fn.encoders().items():
        if other_site != site:
            assert other.W_enc.grad is None, f"L_act at {site} reached the encoder at {other_site}"
    for module in MODULES:
        comp = model.components[module]
        assert comp.V.grad is None and comp.U.grad is None, (
            f"L_act reached {module}'s components -- the two decoders must never meet"
        )


def test_l_internal_reaches_its_own_matrix_and_not_the_encoder(wired):
    cfg, _target_, model, _scale = wired
    module = MODULES[2]
    metric = _bind(InternalReconLoss, InternalReconLossConfig, model, name="head_s", module=module,
                   coeff=1.0, mode="fvu", mask="ci", allow_single_mask=True,
                   warmup_start_frac=0.0, warmup_ramp_frac=0.0)

    model.zero_grad(set_to_none=True)
    metric.update(_context(model, next(_batches(1)))).backward()

    assert model.components[module].V.grad.abs().sum() > 0
    for encoder in model.ci_fn.encoders().values():
        assert encoder.W_enc.grad is None, (
            "L_internal reached the encoder. The gate is a step function of `z'`, so a gradient there "
            "means something other than the gate carried it"
        )


def test_every_encoder_is_trained_by_exactly_one_l_act(wired):
    """Bind the config's real head-R set and check the four gradients land on four encoders."""
    cfg, _target_, model, _scale = wired
    heads_r, _ = aspd_heads(cfg)
    metrics = {
        site: _bind(ActivationReconLoss, ActivationReconLossConfig, model, name=e.name,
                    module=e.module, coeff=e.coeff, warmup=False)
        for site, e in heads_r.items()
    }
    model.zero_grad(set_to_none=True)
    ctx = _context(model, next(_batches(1)))
    total = sum(m.update(ctx) for m in metrics.values())
    total.backward()
    for site, encoder in model.ci_fn.encoders().items():
        assert encoder.W_enc.grad is not None and encoder.W_enc.grad.abs().sum() > 0, (
            f"the encoder at {site} got no gradient -- its dictionary would stay at its init while "
            "every matrix it gates trains against it"
        )


# ---- the round trip every offline tool makes ----------------------------------------------------


def test_a_checkpoint_round_trip_restores_the_gate(wired):
    """`load_component_model`, harvest, the app, the offline sweep and `assemble` all do this. The
    tie and the hooks are the two things a state dict cannot carry.
    """
    cfg, target, model, _scale = wired
    _cache, ci = _clean(model, next(_batches(1)))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(k.startswith("ci_fn._encoders.") for k in state)

    detach_read_site(model)
    reloaded = _build(cfg, target)
    reloaded.load_state_dict(state, strict=True)
    attach_ci_fn(reloaded)
    try:
        _cache2, ci2 = _clean(reloaded, next(_batches(1)))
        for module in MODULES:
            assert torch.equal(ci.lower_leaky[module], ci2.lower_leaky[module])
    finally:
        detach_read_site(reloaded)


def test_detach_read_site_removes_every_hook(wired):
    """The sweeps build one model per checkpoint against a shared target. At 24 encoders a leaked
    set is 24 hooks and 24 captures per stale checkpoint, and `empty_cache` reclaims none of it.
    """
    _cfg_, _target_, model, _scale = wired
    assert all(r._resid_handle is not None for r in model.ci_fn.encoders().values())
    detach_read_site(model)
    assert all(r._resid_handle is None for r in model.ci_fn.encoders().values())

"""L_internal, L_act and AuxK against their definitions."""

import pytest
import torch
from param_decomp.masks import ComponentsMaskInfo
from torch import nn

from aspd.loss_utils import (
    feature_recon_loss,
    gbar_from_ci,
    global_fvu,
    importance_weights,
    masked_module_output,
    per_feature_fvu,
    split_fvu,
    target_module_output,
    warmup_scale,
)
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

D, F = 10, 64


# ---- FVU --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [global_fvu, lambda p, t: per_feature_fvu(p, t)[0]])
def test_fvu_is_zero_for_a_perfect_reconstruction(fn):
    torch.manual_seed(1)
    target = torch.rand(32, F)
    assert fn(target.clone(), target).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("fn", [global_fvu, lambda p, t: per_feature_fvu(p, t)[0]])
def test_fvu_is_one_for_a_mean_predictor(fn):
    """Predicting the per-feature mean explains none of the variance, by definition."""
    torch.manual_seed(2)
    target = torch.rand(64, F)
    pred = target.mean(dim=0, keepdim=True).expand_as(target)
    assert fn(pred, target).item() == pytest.approx(1.0, rel=1e-4)


def test_global_fvu_is_dominated_by_high_norm_latents():
    """The finding that motivated `per_feature_fvu`."""
    torch.manual_seed(3)
    target = torch.rand(64, F) * 0.01
    target[:, 0] = torch.rand(64) * 100.0
    pred = target.clone()
    pred[:, 1:] = 0.0

    assert (pred - target).pow(2).mean().item() < 1e-3, "setup: raw MSE looks fine here"
    assert global_fvu(pred, target).item() < 1e-3, "global FVU is fooled -- this is the point"


def test_per_feature_fvu_registers_the_destroyed_latents():
    """Same input, the metric the loss actually uses. Each latent is normalized by its OWN
    variance, so destroying the quiet 63/64 is scored as destroying 63/64.
    """
    torch.manual_seed(3)
    target = torch.rand(64, F) * 0.01
    target[:, 0] = torch.rand(64) * 100.0
    pred = target.clone()
    pred[:, 1:] = 0.0

    mean_fvu, per_feature, _ = per_feature_fvu(pred, target)
    assert mean_fvu.item() > 0.9, "per-feature FVU must see the long tail"
    assert per_feature[0].item() == pytest.approx(0.0, abs=1e-6), "the loud latent is perfect"
    assert (per_feature[1:] > 0.9).all()


def test_per_feature_fvu_excludes_zero_variance_latents():
    """A latent that never fires has FVU 0/0; it must not poison the mean. Its spurious mass is
    accounted for separately in `split_fvu`.
    """
    torch.manual_seed(8)
    target = torch.rand(32, F)
    target[:, 5:] = 0.0
    mean_fvu, _, _ = per_feature_fvu(target.clone(), target)
    assert torch.isfinite(mean_fvu) and mean_fvu.item() == pytest.approx(0.0, abs=1e-6)


def test_split_reports_alive_and_spurious_separately():
    torch.manual_seed(4)
    target = torch.rand(64, F)
    target[:, F // 2 :] = 0.0

    kept_but_invented = target.clone()
    kept_but_invented[:, F // 2 :] = 0.5  # perfect on alive, invents dead ones
    stats = split_fvu(kept_but_invented, target)
    assert stats["fvu"].item() == pytest.approx(0.0, abs=1e-6)
    assert stats["spurious_mass"].item() > 0.1

    dropped = target.clone()
    dropped[:, : F // 2] = 0.0  # destroys alive, invents nothing
    stats2 = split_fvu(dropped, target)
    assert stats2["fvu"].item() > 0.5
    assert stats2["spurious_mass"].item() == pytest.approx(0.0, abs=1e-9)


def test_worst_decile_is_reported():
    """A mean over features can still hide a minority of badly-reconstructed latents; the worst
    decile is what shows whether coverage is uniform or lumpy.
    """
    torch.manual_seed(9)
    target = torch.rand(64, F)
    pred = target.clone()
    pred[:, : F // 8] = 0.0  # an eighth destroyed, comfortably over a decile
    stats = split_fvu(pred, target)
    assert stats["fvu_worst_decile"].item() > stats["fvu"].item()


def test_alive_frac_is_reported():
    target = torch.zeros(16, F)
    target[:, : F // 4] = 1.0
    assert split_fvu(target.clone(), target)["alive_frac"].item() == pytest.approx(0.25)


# ---- Eq. 10 -------------------------------------------------------------------------------------


def _sae() -> MatryoshkaBatchTopKSAE:
    torch.manual_seed(5)
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(d_in=D, n_features=F, top_k=6, n_batches_to_dead=5)
    )
    for _ in range(20):
        sae.loss(torch.randn(64, D))
    return sae.freeze()


def test_recon_loss_is_zero_when_the_mask_is_a_no_op():
    sae = _sae()
    y = torch.randn(8, D)
    loss, _ = feature_recon_loss(sae, y.clone(), y)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_recon_loss_gradient_reaches_the_masked_output_only():
    sae = _sae()
    y_target = torch.randn(8, D)
    y_masked = (y_target + 0.3 * torch.randn(8, D)).requires_grad_(True)
    loss, _ = feature_recon_loss(sae, y_masked, y_target)
    loss.backward()
    assert y_masked.grad is not None and y_masked.grad.abs().sum() > 0
    assert all(p.grad is None for p in sae.parameters()), "SAE is frozen"


def test_recon_loss_grows_with_corruption():
    sae = _sae()
    torch.manual_seed(6)
    y = torch.randn(32, D)
    mild, _ = feature_recon_loss(sae, y + 0.05 * torch.randn(32, D), y)
    severe, _ = feature_recon_loss(sae, y + 2.0 * torch.randn(32, D), y)
    assert severe.item() > mild.item()


def test_recon_diagnostics_are_detached():
    sae = _sae()
    y = torch.randn(8, D)
    _, stats = feature_recon_loss(sae, y + 0.1, y)
    assert not any(v.requires_grad for v in stats.values())


# ---- shared reductions ---------------------------------------------------------------------------


def test_gbar_is_detached_and_batch_averaged():
    ci = torch.rand(4, 7, 5, requires_grad=True)
    gbar = gbar_from_ci(ci)
    assert gbar.shape == (5,)
    assert not gbar.requires_grad
    assert torch.allclose(gbar, ci.detach().reshape(-1, 5).mean(0))


def test_weights_are_normalized_to_sum_to_one():
    """Eq. 9 is a weighted MEAN. `gbar_sum` comes back so the unnormalized value stays derivable."""
    gbar = torch.tensor([0.8, 0.2, 0.0, 0.5])
    weights, gbar_sum = importance_weights(gbar)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.isclose(gbar_sum, torch.tensor(1.5))
    assert torch.allclose(weights, gbar / 1.5)


def test_loss_is_invariant_to_a_global_rescaling_of_importance():
    """The property the normalization exists for."""
    gbar = torch.tensor([0.8, 0.6, 0.4, 0.02])
    term = torch.tensor([100.0, 50.0, 25.0, 900.0])

    def loss(g):
        w, _ = importance_weights(g)
        return (w * term).sum()

    for scale in (0.5, 0.1, 0.001):
        assert torch.isclose(loss(gbar * scale), loss(gbar), rtol=1e-5)


def test_loss_is_invariant_to_components_leaving_the_population():
    gbar = torch.tensor([0.8, 0.6, 0.4])
    term = torch.tensor([100.0, 50.0, 25.0])
    w_before, _ = importance_weights(gbar)

    dead = gbar.clone()
    dead[2] = 0.0
    w_after, _ = importance_weights(dead)

    assert torch.isclose((w_after * term).sum(), (w_before[:2] * term[:2]).sum() / w_before[:2].sum())
    # ... and the indicator form it replaces is NOT invariant: it drops a whole term from the sum.
    assert (gbar > 0.01).float() @ term != pytest.approx(((dead > 0.01).float() @ term).item())


def test_all_dead_yields_zero_rather_than_nan():
    """`0/0` on a batch where nothing fires would poison the step; the eps floor makes it 0."""
    weights, gbar_sum = importance_weights(torch.zeros(5))
    assert torch.isfinite(weights).all() and float(weights.sum()) == 0.0
    assert float(gbar_sum) == 0.0


def test_importance_weights_reject_a_grad_carrying_gbar():
    """Gradient through the weight would let the optimizer cut the loss by suppressing importance.
    Asserted rather than assumed, in the one place the weighting is formed.
    """
    with pytest.raises(AssertionError):
        importance_weights(torch.rand(4, requires_grad=True))


@pytest.mark.parametrize(
    "step,expected",
    [(0, 0.0), (999, 0.0), (1000, 0.0), (1500, 0.5), (2000, 1.0), (9000, 1.0)],
)
def test_warmup_ramp(step, expected):
    assert warmup_scale(step, 10_000) == pytest.approx(expected)


# ---- y from the cached input, no model forward ---------------------------------------------------


class _Comp(nn.Module):
    """Stands in for `LinearComponents`: same call signature, trivially checkable output."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d_in, 3))
        self.U = nn.Parameter(torch.randn(3, d_out))

    def forward(self, x, mask=None, weight_delta_and_mask=None):
        acts = x @ self.V
        if mask is not None:
            acts = acts * mask
        out = acts @ self.U
        if weight_delta_and_mask is not None:
            delta, delta_mask = weight_delta_and_mask
            out = out + delta_mask[..., None] * (x @ delta.t())
        return out


def test_target_output_is_detached_and_matches_the_module():
    torch.manual_seed(0)
    target = nn.Linear(D, D)
    x = torch.randn(4, 6, D, requires_grad=True)
    y = target_module_output(target, x)
    assert not y.requires_grad, "Eq. 10's reference must never be a gradient path"
    assert torch.allclose(y, target(x).detach(), atol=1e-6)


def test_masked_output_carries_gradient_to_the_components():
    torch.manual_seed(1)
    comp, target = _Comp(D, D), nn.Linear(D, D)
    x = torch.randn(4, 6, D)
    info = ComponentsMaskInfo(component_mask=torch.rand(4, 6, 3), routing_mask="all")
    masked_module_output(comp, target, x, info).sum().backward()
    assert comp.V.grad is not None and comp.V.grad.abs().sum() > 0
    assert comp.U.grad is not None


def test_routing_mask_blend_matches_the_core_hook():
    torch.manual_seed(2)
    comp, target = _Comp(D, D), nn.Linear(D, D)
    x = torch.randn(4, 6, D)
    mask = torch.rand(4, 6, 3)
    routing = torch.rand(4, 6) > 0.5

    got = masked_module_output(
        comp, target, x, ComponentsMaskInfo(component_mask=mask, routing_mask=routing)
    )
    comp_out = comp(x, mask=mask)
    want = torch.where(routing[..., None], comp_out, target(x))
    assert torch.allclose(got, want, atol=1e-6)


def test_routing_all_is_pure_component_output():
    torch.manual_seed(3)
    comp, target = _Comp(D, D), nn.Linear(D, D)
    x = torch.randn(2, 3, D)
    mask = torch.rand(2, 3, 3)
    got = masked_module_output(
        comp, target, x, ComponentsMaskInfo(component_mask=mask, routing_mask="all")
    )
    assert torch.allclose(got, comp(x, mask=mask), atol=1e-6)


def test_weight_delta_is_applied():
    """The spillover term is a masked extra component; dropping it would make `y_masked` diverge
    from what the real forward installs whenever `use_delta_component` is on.
    """
    torch.manual_seed(4)
    comp, target = _Comp(D, D), nn.Linear(D, D)
    x = torch.randn(2, 3, D)
    mask = torch.ones(2, 3, 3)
    delta = torch.randn(D, D)
    without = masked_module_output(
        comp, target, x, ComponentsMaskInfo(component_mask=mask, routing_mask="all")
    )
    with_delta = masked_module_output(
        comp, target, x,
        ComponentsMaskInfo(
            component_mask=mask, routing_mask="all",
            weight_delta_and_mask=(delta, torch.ones(2, 3)),
        ),
    )
    assert not torch.allclose(without, with_delta)
    assert torch.allclose(with_delta - without, x @ delta.t(), atol=1e-5)

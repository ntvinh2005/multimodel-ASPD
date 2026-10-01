"""The Matryoshka BatchTopK SAE matches the reference implementation numerically."""

import pytest
import torch

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

from ._reference_matryoshka import GlobalBatchTopKMatryoshkaSAE

D_IN, N_FEATURES = 16, 64
GROUP_FRACS = (1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2)


def _make_pair() -> tuple[MatryoshkaBatchTopKSAE, GlobalBatchTopKMatryoshkaSAE]:
    """Both models, weight-for-weight identical. Reference is put in train mode so it takes the
    BatchTopK branch rather than the threshold branch.
    """
    cfg = MatryoshkaSAEConfig(d_in=D_IN, n_features=N_FEATURES, group_fracs=GROUP_FRACS)
    ours = MatryoshkaBatchTopKSAE(cfg)

    ref_cfg = {
        "seed": 0,
        "act_size": D_IN,
        "dict_size": N_FEATURES,
        "group_sizes": cfg.group_sizes(),
        "top_k": cfg.top_k,
        "top_k_aux": cfg.top_k_aux,
        "aux_penalty": cfg.aux_penalty,
        "n_batches_to_dead": cfg.n_batches_to_dead,
        "l1_coeff": cfg.l1_coeff,
        "input_unit_norm": False,
        "device": "cpu",
        "dtype": torch.float32,
    }
    ref = GlobalBatchTopKMatryoshkaSAE(ref_cfg)
    with torch.no_grad():
        ref.W_enc.copy_(ours.W_enc)
        ref.W_dec.copy_(ours.W_dec)
        ref.b_dec.copy_(ours.b_dec)
        ref.b_enc.zero_()  # unused upstream; zeroed so the parity claim is unambiguous
    ref.train()
    return ours, ref


def test_group_sizes_match_bussmann_construction():
    cfg = MatryoshkaSAEConfig(d_in=8, n_features=36864, group_fracs=GROUP_FRACS)
    assert cfg.group_sizes() == [2304, 2304, 4608, 9216, 18432]
    # Cumulative prefixes double.
    prefixes = torch.cumsum(torch.tensor(cfg.group_sizes()), 0).tolist()
    assert prefixes == [2304, 4608, 9216, 18432, 36864]


def test_group_sizes_absorb_rounding_remainder():
    cfg = MatryoshkaSAEConfig(d_in=8, n_features=1000, group_fracs=GROUP_FRACS)
    assert sum(cfg.group_sizes()) == 1000


def test_loss_matches_reference():
    torch.manual_seed(0)
    ours, ref = _make_pair()
    x = torch.randn(128, D_IN)

    got = ours.loss(x)
    want = ref(x)

    assert got["loss"].item() == pytest.approx(want["loss"].item(), rel=1e-6)
    assert got["l2_loss"].item() == pytest.approx(want["l2_loss"].item(), rel=1e-6)
    assert got["aux_loss"].item() == pytest.approx(want["aux_loss"].item(), rel=1e-6)
    assert got["l0_norm"].item() == pytest.approx(want["l0_norm"].item(), rel=1e-6)
    assert got["l1_norm"].item() == pytest.approx(want["l1_norm"].item(), rel=1e-6)


def test_threshold_ema_matches_reference_over_many_steps():
    """The EMA is path-dependent, so a single step would not catch a drift in the update rule."""
    torch.manual_seed(1)
    ours, ref = _make_pair()
    for _ in range(25):
        x = torch.randn(64, D_IN)
        ours.loss(x)
        ref(x)
    assert ours.threshold.item() == pytest.approx(float(ref.threshold), rel=1e-6)


def test_dead_feature_tracking_matches_reference():
    torch.manual_seed(2)
    ours, ref = _make_pair()
    for _ in range(30):
        x = torch.randn(32, D_IN)
        ours.loss(x)
        ref(x)
    assert torch.equal(ours.n_batches_not_active, ref.num_batches_not_active)


def test_gradients_match_reference():
    torch.manual_seed(3)
    ours, ref = _make_pair()
    x = torch.randn(96, D_IN)
    ours.loss(x)["loss"].backward()
    ref(x)["loss"].backward()
    for name in ("W_enc", "W_dec", "b_dec"):
        assert torch.allclose(
            getattr(ours, name).grad, getattr(ref, name).grad, rtol=1e-5, atol=1e-7
        ), f"{name} gradient diverges from reference"


def test_decoder_normalization_matches_reference():
    torch.manual_seed(4)
    ours, ref = _make_pair()
    x = torch.randn(64, D_IN)
    ours.loss(x)["loss"].backward()
    ref(x)["loss"].backward()
    ours.normalize_decoder_()
    ref.make_decoder_weights_and_grad_unit_norm()
    assert torch.allclose(ours.W_dec.data, ref.W_dec.data, rtol=1e-6, atol=1e-8)
    assert torch.allclose(ours.W_dec.grad, ref.W_dec.grad, rtol=1e-5, atol=1e-7)


# ---- properties our footprint math depends on (beyond reference parity) ---------------------


def test_decoder_rows_are_unit_norm_after_normalization():
    """`J_dec e_i = W_dec[i]` is used directly as a JVP tangent, so unit norm is load-bearing."""
    torch.manual_seed(5)
    ours, _ = _make_pair()
    ours.W_dec.grad = torch.randn_like(ours.W_dec)
    ours.normalize_decoder_()
    norms = ours.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_encoder_normalized_has_unit_columns():
    torch.manual_seed(6)
    ours, _ = _make_pair()
    norms = ours.encoder_normalized().norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_features_are_exactly_zero_off_the_active_mask():
    """The gated footprint is exact only if `features` and `active_mask` agree everywhere."""
    torch.manual_seed(7)
    ours, _ = _make_pair()
    for _ in range(5):  # warm the threshold off zero
        ours.loss(torch.randn(64, D_IN))
    x = torch.randn(20, D_IN)
    f, mask = ours.features(x), ours.active_mask(x)
    assert (f[~mask] == 0).all()
    assert (f[mask] > 0).all()


def test_freeze_blocks_gradient():
    ours, _ = _make_pair()
    ours.freeze()
    assert not any(p.requires_grad for p in ours.parameters())
    assert not ours.training


# ---- encoder init modes ------------------------------------------------------------------------


def test_reference_encoder_init_is_unchanged_and_shrinks_with_width():
    import math

    for n_features in (256, 1024):
        torch.manual_seed(0)
        sae = MatryoshkaBatchTopKSAE(
            MatryoshkaSAEConfig(d_in=D_IN, n_features=n_features, group_fracs=GROUP_FRACS)
        )
        assert sae.cfg.encoder_init == "reference"
        expected = math.sqrt(2 * D_IN / n_features)
        norms = sae.W_enc.norm(dim=0)
        assert torch.allclose(norms.mean(), torch.tensor(expected), rtol=0.05), (
            f"F={n_features}: mean column norm {norms.mean():.4f}, expected ~{expected:.4f}"
        )


def test_unit_norm_encoder_init_ties_encoder_to_decoder_at_every_width():
    torch.manual_seed(0)
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(
            d_in=D_IN, n_features=1024, group_fracs=GROUP_FRACS, encoder_init="unit_norm"
        )
    )
    assert torch.allclose(sae.W_enc.norm(dim=0), torch.ones(1024), atol=1e-5)
    assert torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(1024), atol=1e-5)
    # Exactly tied: the whole point is that the two are the same directions at the same scale.
    assert torch.allclose(sae.W_enc, sae.W_dec.t(), atol=1e-6)


def test_encoder_init_changes_no_forward_or_backward_math():
    """It is an INIT knob. Given identical weights the two modes must be indistinguishable --
    which is why it cannot break the vendored Bussmann parity claim.
    """
    torch.manual_seed(0)
    ref = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(d_in=D_IN, n_features=256, group_fracs=GROUP_FRACS)
    )
    unit = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(
            d_in=D_IN, n_features=256, group_fracs=GROUP_FRACS, encoder_init="unit_norm"
        )
    )
    with torch.no_grad():
        unit.W_enc.copy_(ref.W_enc)
        unit.W_dec.copy_(ref.W_dec)
        unit.b_dec.copy_(ref.b_dec)
    x = torch.randn(64, D_IN)
    a, b = ref.loss(x), unit.loss(x)
    assert torch.allclose(a["loss"], b["loss"], atol=1e-7)
    a["loss"].backward()
    b["loss"].backward()
    assert torch.allclose(ref.W_enc.grad, unit.W_enc.grad, atol=1e-7)

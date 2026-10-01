"""`component_init` for PD Transcoder components, and that the default leaves them unchanged."""

import torch
from param_decomp.components import LinearComponents

from aspd.transcoder_components import (
    TranscoderLinearComponents,
    _to_transcoder,
    install_transcoder_components,
)

C, D_IN, D_OUT = 24, 16, 40  # all distinct, so a transposed axis cannot pass


def _linear() -> LinearComponents:
    torch.manual_seed(0)
    return LinearComponents(C, d_in=D_IN, d_out=D_OUT, bias=None)


def _built(init: str) -> TranscoderLinearComponents:
    """`_linear()` reseeds, so the RNG state entering `_to_transcoder` is fully determined by it --
    which is what lets the expected draw below be reproduced by repeating the same call order.
    """
    out = _to_transcoder("m", _linear(), init)  # pyright: ignore[reportArgumentType]
    assert isinstance(out, TranscoderLinearComponents)
    return out


def test_reference_init_is_bit_identical_to_the_hardcoded_draw():
    comp = _linear()
    expected = TranscoderLinearComponents(
        C=comp.C, d_in=comp.d_in, d_out=comp.d_out, bias=None
    )
    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(expected.V)
        torch.nn.init.kaiming_uniform_(expected.U)
        expected.U.div_(expected.U.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    got = _built("reference")
    torch.testing.assert_close(got.V, expected.V, rtol=0, atol=0)
    torch.testing.assert_close(got.U, expected.U, rtol=0, atol=0)


def test_the_default_is_reference():
    """A config that says nothing must get the draw every existing run got."""
    default = _to_transcoder("m", _linear())
    assert isinstance(default, TranscoderLinearComponents)
    torch.testing.assert_close(default.V, _built("reference").V, rtol=0, atol=0)
    torch.testing.assert_close(default.U, _built("reference").U, rtol=0, atol=0)


def test_unit_norm_gives_the_encoder_unit_columns():
    """`V` is `[d_in, C]`, so a column is one component's encoder direction -- the axis `W_enc`'s
    `norm(dim=0)` normalizes in `MatryoshkaBatchTopKSAE.__init__`.
    """
    v = _built("unit_norm").V
    assert v.shape == (D_IN, C)
    torch.testing.assert_close(v.norm(dim=0), torch.ones(C), rtol=1e-6, atol=1e-6)


def test_unit_norm_changes_only_the_encoder_scale_not_its_directions():
    """It is a rescale of each column, not a redraw: the decoder and every encoder DIRECTION are
    the reference draw's. A version that redrew would change the arm in a second, unstated way.
    """
    ref, uni = _built("reference"), _built("unit_norm")
    torch.testing.assert_close(uni.U, ref.U, rtol=0, atol=0)
    cos = torch.nn.functional.cosine_similarity(uni.V, ref.V, dim=0)
    torch.testing.assert_close(cos, torch.ones(C), rtol=1e-6, atol=1e-6)


def test_unit_norm_is_the_fix_it_claims_to_be_at_a_wide_C():
    """The failure the option exists for: raw `kaiming_uniform_` on `[d_in, C]` has `fan_in = C`,
    so encoder columns carry norm `~sqrt(2*d_in/C)` against a UNIT-norm decoder, and the ratio falls
    as the dictionary widens. At Gemma's shape that is 0.707; the whole point is that it is 1.0.
    """
    torch.manual_seed(0)
    wide = LinearComponents(4 * D_IN, d_in=D_IN, d_out=D_OUT, bias=None)
    torch.manual_seed(7)
    ref = _to_transcoder("m", wide, "reference")
    torch.manual_seed(7)
    uni = _to_transcoder("m", wide, "unit_norm")
    assert isinstance(ref, TranscoderLinearComponents)
    assert isinstance(uni, TranscoderLinearComponents)
    assert ref.V.norm(dim=0).mean() < 0.85, "the reference draw should be the SHRUNK one here"
    torch.testing.assert_close(
        uni.V.norm(dim=0).mean(), torch.tensor(1.0), rtol=1e-6, atol=1e-6
    )


def test_install_passes_the_init_through_to_every_module():
    """The patch is the only path that constructs these, so an `init` it drops is an option that
    parses, logs, and does nothing.
    """
    import param_decomp.component_model as cm

    stock = cm.make_components
    try:
        install_transcoder_components("unit_norm")
        torch.manual_seed(0)

        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(D_IN, D_OUT, bias=False)

        built = cm.make_components(_M(), {"lin": C})
        for comp in built.values():
            assert isinstance(comp, TranscoderLinearComponents)
            torch.testing.assert_close(comp.V.norm(dim=0), torch.ones(C), rtol=1e-6, atol=1e-6)
    finally:
        cm.make_components = stock

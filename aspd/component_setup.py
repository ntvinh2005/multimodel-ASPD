"""Installs the component parameterization chosen by the config before the `ComponentModel` is built."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aspd.config import LMInterpExperimentConfig

_STOCK_MARKER = "_p2_stock_make_components"


def _restore_stock_factory() -> None:
    import param_decomp.component_model as cm

    stock = getattr(cm, _STOCK_MARKER, None)
    if stock is None:
        # First call in this process: remember core's own factory before anyone patches it.
        setattr(cm, _STOCK_MARKER, cm.make_components)
        return
    cm.make_components = stock


def install_component_parameterization(cfg: "LMInterpExperimentConfig") -> str:
    """Point `make_components` at whatever `cfg` asks for. Returns the parameterization's name."""
    from aspd.transcoder_components import install_transcoder_components

    from aspd.config import component_arch, component_init

    _restore_stock_factory()
    if component_arch(cfg) == "transcoder":
        install_transcoder_components(component_init(cfg))
        return "transcoder"
    return "vpd"


def assert_components_have_write_vectors(cfg: "LMInterpExperimentConfig", stage: str) -> None:
    """Reject a stage that reads `U[c]` as a token-independent write direction."""
    from aspd.config import component_arch

    arch = component_arch(cfg)
    assert arch in ("vpd", "transcoder"), (
        f"{stage} reads `U[c]` as a per-component write direction, which does not exist at "
        f"component_arch={arch!r}. Ablation-based stages (ce_kl, scr_tpp, harvest) are unaffected "
        "and can be run with `--only`."
    )

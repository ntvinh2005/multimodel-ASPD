"""The method ("arm") a config trains, derived from the config, plus the launch-time checks.

The arm is a pure function of the config -- which CI function it builds and which loss terms it
optimizes (`coeff > 0`) -- never a flag that could disagree with the YAML:

    ci_config.mode         optimized terms                                   arm
    aspd                   L_internal (fvu) + L_act + AuxK                    aspd
    pd_transcoder          L_internal (matryoshka) + AuxK                     pdtc
                           ... + L_param                                      pdtc_param
                           ... + L_ablate (stochastic)                        pdtc_ablate
    global                 L_sparse + L_param + L_ablate (stoch + adv)        vpd
                           ... with the adaptive-L0 L_sparse                  vpd_adaptive
                           ... + L_internal (fvu)                             vpd_internal
                           ... + L_internal, without L_param                  vpd_internal_noparam
                           ... + L_internal, without L_ablate                 vpd_internal_noablate

Anything else derives `custom`. Every check below fails at launch rather than hours into a run.
"""

from typing import TYPE_CHECKING

from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.metrics.stochastic_recon_subset import StochasticReconSubsetLossConfig

from aspd.adaptive_l0 import AdaptiveSparsityLossConfig
from aspd.ci import ASPDCiConfig, PDTranscoderCiConfig
from aspd.config import component_arch
from aspd.losses import (
    AuxKLossConfig,
    InternalReconLossConfig,
    ComponentAliveTrackerConfig,
    ActivationReconLossConfig,
)

if TYPE_CHECKING:
    from aspd.config import LMInterpExperimentConfig

ARMS = (
    "aspd",
    "pdtc",
    "pdtc_param",
    "pdtc_ablate",
    "vpd",
    "vpd_adaptive",
    "vpd_internal",
    "vpd_internal_noparam",
    "vpd_internal_noablate",
)

# Every loss entry `aspd.run` injects `module` / `train_diag_every` into.
INJECTED_LOSS_CONFIGS = (InternalReconLossConfig, AuxKLossConfig, ActivationReconLossConfig)


def _optimized(entry: object) -> bool:
    """A term is optimized iff its coeff is strictly positive; `coeff 0` is measured only."""
    return (getattr(entry, "coeff", 0.0) or 0.0) > 0.0


def _any_optimized(cfg: "LMInterpExperimentConfig", *types: type) -> bool:
    return any(isinstance(e, types) and _optimized(e) for e in cfg.pd.loss_metrics)


def uses_transcoder(cfg: "LMInterpExperimentConfig") -> bool:
    """PD Transcoder or ASPD: rank-1 components gated by a BatchTopK code (`PDTranscoderCiConfig`)."""
    return isinstance(cfg.pd.ci_config, PDTranscoderCiConfig)


def uses_aspd(cfg: "LMInterpExperimentConfig") -> bool:
    """ASPD: the gate is a shared sparse encoder g^s on the residual stream."""
    return isinstance(cfg.pd.ci_config, ASPDCiConfig)


def target_arch(cfg: "LMInterpExperimentConfig") -> str | None:
    """The target's HF model class name, for `aspd.sites`'s architecture-qualified lookups."""
    spec = cfg.target.spec
    dotted = getattr(spec, "model_class", None) or getattr(spec, "params", {}).get("model_class")
    return dotted.rsplit(".", 1)[-1] if dotted else None


def resid_site_map(cfg: "LMInterpExperimentConfig") -> dict[str, str]:
    """`{decomposed module: residual-stream site its gate reads}` for an ASPD config."""
    from aspd.sites import resid_site_for_module

    arch = target_arch(cfg)
    gate = cfg.pd.ci_config
    assert isinstance(gate, ASPDCiConfig)
    modules = [t.module_pattern for t in cfg.pd.decomposition_targets]

    if len(modules) == 1:
        only = modules[0]
        stated = {only: gate.resid_site or (gate.resid_sites or {}).get(only, "")}
        assert stated[only], f"no gate site for {only!r}: set `resid_site` or `resid_sites`"
    else:
        assert gate.resid_sites is not None, (
            f"{len(modules)} decomposed matrices need `resid_sites: {{module: site}}`, not the "
            "scalar `resid_site`"
        )
        stated = dict(gate.resid_sites)
        missing, extra = set(modules) - set(stated), set(stated) - set(modules)
        assert not missing and not extra, (
            f"`resid_sites` does not match the decomposition targets: missing {sorted(missing)}, "
            f"extra {sorted(extra)}"
        )

    wrong = {
        m: (stated[m], resid_site_for_module(m, arch))
        for m in modules
        if stated[m] != resid_site_for_module(m, arch)
    }
    assert not wrong, f"residual sites disagree with `aspd.sites.RESID_SITES` (stated, expected): {wrong}"
    return stated


def _entry_module(entry, modules: list[str]) -> str:
    """The module a loss entry is bound to; an empty `module` means the only target."""
    if entry.module:
        return entry.module
    assert len(modules) == 1, (
        f"entry {entry.name!r} names no `module` but the config decomposes {len(modules)} matrices"
    )
    return modules[0]


def aspd_heads(
    cfg: "LMInterpExperimentConfig",
) -> tuple[dict[str, ActivationReconLossConfig], dict[str, InternalReconLossConfig]]:
    """`({encoder: L_act entry}, {module: L_internal entry})` for an ASPD config, all optimized."""
    from aspd.ci import encoder_key_map

    sites = resid_site_map(cfg)
    modules = [t.module_pattern for t in cfg.pd.decomposition_targets]
    encoder_of = encoder_key_map(sites, getattr(cfg.pd.ci_config, "share_encoders", True))
    expected_encoders = sorted(set(encoder_of.values()))

    heads_act: dict[str, ActivationReconLossConfig] = {}
    for entry in (e for e in cfg.pd.loss_metrics if isinstance(e, ActivationReconLossConfig)):
        module = _entry_module(entry, modules)
        assert module in sites, f"L_act entry {entry.name!r} is bound to undecomposed {module!r}"
        key = encoder_of[module]
        assert key not in heads_act, f"two L_act entries train the encoder {key!r}"
        heads_act[key] = entry
    assert sorted(heads_act) == expected_encoders, (
        f"L_act covers {len(heads_act)} of {len(expected_encoders)} encoders; missing "
        f"{sorted(set(expected_encoders) - set(heads_act))[:4]}"
    )

    heads_internal: dict[str, InternalReconLossConfig] = {}
    for entry in (
        e for e in cfg.pd.loss_metrics if isinstance(e, InternalReconLossConfig) and e.mode == "fvu"
    ):
        module = _entry_module(entry, modules)
        assert module in sites, f"L_internal entry {entry.name!r} is bound to undecomposed {module!r}"
        assert module not in heads_internal, f"two L_internal entries reconstruct {module!r}"
        heads_internal[module] = entry
    assert sorted(heads_internal) == sorted(sites), (
        f"L_internal covers {len(heads_internal)} of {len(sites)} matrices; missing "
        f"{sorted(set(sites) - set(heads_internal))[:4]}"
    )

    unoptimized = [e.name for e in (*heads_act.values(), *heads_internal.values()) if not _optimized(e)]
    assert not unoptimized, f"ASPD needs L_act and L_internal optimized; at coeff 0: {unoptimized[:4]}"
    return heads_act, heads_internal


def _assert_aspd(cfg: "LMInterpExperimentConfig") -> None:
    if not uses_aspd(cfg):
        return
    gate = cfg.pd.ci_config
    assert isinstance(gate, ASPDCiConfig)
    aspd_heads(cfg)
    assert gate.d_act is not None, "ASPD needs `d_act` (the residual stream width)"
    assert cfg.pd.sampling == "continuous", f"ASPD needs `sampling: continuous`, got {cfg.pd.sampling!r}"


def _assert_transcoder(cfg: "LMInterpExperimentConfig") -> None:
    """`component_arch: transcoder` and a transcoder CI function come together or not at all."""
    arch, is_tc_gate = component_arch(cfg), uses_transcoder(cfg)
    if not (arch == "transcoder" or is_tc_gate):
        return
    assert arch == "transcoder" and is_tc_gate, (
        f"component_arch={arch!r} with ci_config.mode={cfg.pd.ci_config.mode!r}: the transcoder "
        "parameterization (b_dec, b_out) and the transcoder gate are one object"
    )
    gate = cfg.pd.ci_config
    assert isinstance(gate, PDTranscoderCiConfig)
    if gate.n_features is not None:
        mismatched = {
            t.module_pattern: t.C for t in cfg.pd.decomposition_targets if t.C != gate.n_features
        }
        assert not mismatched, (
            f"n_features={gate.n_features} but {mismatched} disagree: component c IS feature c"
        )


def _assert_auxk_bindings(cfg: "LMInterpExperimentConfig") -> None:
    """Every `AuxKLoss` names a reconstruction entry declared before it (metrics run in order)."""
    seen: set[str] = set()
    for entry in cfg.pd.loss_metrics:
        if isinstance(entry, InternalReconLossConfig | ActivationReconLossConfig):
            seen.add(entry.name or type(entry).__name__.removesuffix("Config"))
        elif isinstance(entry, AuxKLossConfig):
            assert entry.recon, "AuxKLoss needs `recon: <name of the reconstruction entry>`"
            assert entry.recon in seen, (
                f"AuxKLoss(recon={entry.recon!r}) names no entry declared before it; seen {sorted(seen)}"
            )


def _assert_internal_entries(cfg: "LMInterpExperimentConfig") -> None:
    """At most one L_internal entry per (module, mask), and named when there are several."""
    modules = [t.module_pattern for t in cfg.pd.decomposition_targets]
    entries = [e for e in cfg.pd.loss_metrics if isinstance(e, InternalReconLossConfig)]
    by_module: dict[str, list[InternalReconLossConfig]] = {}
    for entry in entries:
        by_module.setdefault(_entry_module(entry, modules), []).append(entry)
    for module, group in by_module.items():
        assert len({(e.mode, e.mask) for e in group}) == len(group), (
            f"two InternalReconLoss entries share mode and mask on {module!r}; their coefficients would sum"
        )
    if len(entries) > 1:
        names = [e.name for e in entries]
        assert all(names) and len(set(names)) == len(names), (
            f"several InternalReconLoss entries need distinct `name`s; got {names}"
        )


def assert_config(cfg: "LMInterpExperimentConfig") -> None:
    """Every launch-time check. Called by `derive_arm_name`, so no run skips it."""
    assert any(isinstance(e, ComponentAliveTrackerConfig) for e in cfg.pd.loss_metrics), (
        "config is missing `ComponentAliveTracker` (coeff 0): the alive set is a firing history "
        "built during training and cannot be recovered from a checkpoint"
    )
    _assert_auxk_bindings(cfg)
    _assert_internal_entries(cfg)
    _assert_transcoder(cfg)
    _assert_aspd(cfg)


def derive_arm_name(cfg: "LMInterpExperimentConfig") -> str:
    """The paper arm this config trains (see the module docstring), or `custom`."""
    assert_config(cfg)
    param = _any_optimized(cfg, FaithfulnessLossConfig)
    stochastic = _any_optimized(cfg, StochasticReconSubsetLossConfig)
    adversarial = _any_optimized(cfg, PersistentPGDReconLossConfig)
    internal = any(
        isinstance(e, InternalReconLossConfig) and e.mode == "fvu" and _optimized(e)
        for e in cfg.pd.loss_metrics
    )
    adaptive = _any_optimized(cfg, AdaptiveSparsityLossConfig)

    if uses_aspd(cfg):
        return "aspd" if not (param or stochastic or adversarial) else "custom"
    if uses_transcoder(cfg):
        match (param, stochastic or adversarial):
            case (False, False):
                return "pdtc"
            case (True, False):
                return "pdtc_param"
            case (False, True):
                return "pdtc_ablate" if not adversarial else "custom"
            case _:
                return "custom"
    ablate = stochastic and adversarial
    if not internal:
        if param and ablate:
            return "vpd_adaptive" if adaptive else "vpd"
        return "custom"
    match (param, ablate, stochastic or adversarial):
        case (True, True, _):
            return "vpd_internal"
        case (False, True, _):
            return "vpd_internal_noparam"
        case (True, False, False):
            return "vpd_internal_noablate"
        case _:
            return "custom"


def arm_summary(cfg: "LMInterpExperimentConfig") -> str:
    """One line per loss term for the run log: what the run optimizes, and what it only measures."""
    rows = [
        f"  ci_fn: {cfg.pd.ci_config.mode}",
        f"  component_arch: {component_arch(cfg)}",
    ]
    if uses_aspd(cfg):
        gate = cfg.pd.ci_config
        assert isinstance(gate, ASPDCiConfig)
        heads_act, heads_internal = aspd_heads(cfg)
        rows.append(
            f"  shared encoder g^s: {len(heads_act)} encoder(s) over {len(heads_internal)} matrices, "
            f"reading {sorted(set(resid_site_map(cfg).values()))[:3]}; d_act={gate.d_act} "
            f"top_k={gate.top_k}"
        )
    for entry in cfg.pd.loss_metrics:
        status = "optimized" if _optimized(entry) else "measured-only"
        rows.append(f"  {entry.type}: coeff={getattr(entry, 'coeff', None)} [{status}]")
    return "\n".join(rows)

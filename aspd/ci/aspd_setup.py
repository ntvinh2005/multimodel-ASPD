"""Build, calibrate and tie ASPD's CI function.

Calibration measures the mean of r_t at every residual site over `CALIBRATION_BATCHES` batches.
Seeding sets b_dec to that mean, rescales W_enc to unit-norm columns and ties W_dec = W_enc^T.
"""

from collections.abc import Iterator
from typing import Any

import torch
from param_decomp.component_model import ComponentModel
from param_decomp.log import logger
from torch import Tensor, nn

from aspd.ci.aspd import ASPDCiFn, ASPDCiFnSet, SharedEncoder
from aspd.ci.pd_transcoder_setup import seed_transcoder_ci_fn
from aspd.config import LMInterpExperimentConfig

CALIBRATION_BATCHES = 8


def _resid_sites(cfg: LMInterpExperimentConfig) -> list[str]:
    """The distinct residual sites the config's encoders read."""
    ci_config = cfg.pd.ci_config
    sites = getattr(ci_config, "resid_sites", None)
    if sites is None:
        site = ci_config.resid_site  # pyright: ignore[reportAttributeAccessIssue]
        assert site, "ASPD needs `resid_site` or `resid_sites`"
        return [site]
    modules = [t.module_pattern for t in cfg.pd.decomposition_targets]
    assert set(modules) <= set(sites), f"`resid_sites` does not cover {sorted(set(modules) - set(sites))}"
    return sorted({sites[m] for m in modules})


def calibrate_aspd_encoder(
    cfg: LMInterpExperimentConfig,
    target_model: Any,
    module: str,
    batches: Iterator[Tensor],
    device: str,
) -> tuple[Tensor | dict[str, Tensor], dict[str, float]]:
    """E[r_t] per residual site: a [d_act] tensor for one site, `{site: [d_act]}` for several."""
    del module
    ci_config = cfg.pd.ci_config
    d_act = ci_config.d_act  # pyright: ignore[reportAttributeAccessIssue]
    assert d_act, "ASPD needs `d_act`"
    sites = _resid_sites(cfg)
    sums = {s: torch.zeros(d_act, dtype=torch.float64, device=device) for s in sites}
    sq_sums = {s: torch.zeros(d_act, dtype=torch.float64, device=device) for s in sites}
    counts = dict.fromkeys(sites, 0)

    def make_hook(site: str):
        def hook(_m: nn.Module, args: tuple, _out: object) -> None:
            r = args[0].detach().reshape(-1, args[0].shape[-1]).double()
            assert r.shape[-1] == d_act, f"{site!r} has width {r.shape[-1]}, config says d_act={d_act}"
            sums[site] += r.sum(dim=0)
            sq_sums[site] += r.pow(2).sum(dim=0)
            counts[site] += r.shape[0]

        return hook

    handles = [target_model.get_submodule(s).register_forward_hook(make_hook(s)) for s in sites]
    try:
        with torch.no_grad():
            for i, batch in enumerate(batches):
                if i >= CALIBRATION_BATCHES:
                    break
                target_model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
    assert all(counts.values()), f"nothing captured at {[s for s in sites if not counts[s]]}"

    means: dict[str, Tensor] = {}
    stats: dict[str, float] = {}
    for site in sites:
        n = counts[site]
        mean = sums[site] / n
        centred_var = (sq_sums[site] / n - mean.pow(2)).clamp_min(0.0).sum()
        means[site] = mean.to(torch.float32).to(device)
        stats[f"{site}/resid_rms"] = float(((sq_sums[site] / n).sum() / d_act) ** 0.5)
        stats[f"{site}/resid_centred_rms"] = float((centred_var / d_act) ** 0.5)
        stats[f"{site}/resid_mean_abs"] = float(mean.abs().mean())
        stats[f"{site}/calibration_tokens"] = float(n)
        logger.info(
            f"ASPD encoder at {site!r}: E[r] over {n} tokens, "
            f"|E[r]| {stats[f'{site}/resid_mean_abs']:.4g}"
        )

    if getattr(ci_config, "resid_sites", None) is None:
        return means[sites[0]], {k.split("/", 1)[1]: v for k, v in stats.items()}
    return means, stats


def tie_aspd_ci_fn(component_model: ComponentModel) -> SharedEncoder | ASPDCiFnSet:
    """Attach the CI function to the components it gates; run after the Trainer is built."""
    ci_fn = component_model.ci_fn
    if isinstance(ci_fn, ASPDCiFnSet):
        ci_fn.attach_components(component_model.components)
        unhooked = [s for s, e in ci_fn.encoders().items() if e._resid_handle is None]
        assert not unhooked, f"encoders without a residual hook: {unhooked}"
        logger.info(
            f"ASPD: {len(ci_fn.module_names)} matrices over {len(ci_fn.encoder_names)} encoders "
            f"({'shared' if ci_fn.shared else 'one per matrix'}), top_k={ci_fn.cfg.top_k}"
        )
        return ci_fn

    ci_fn = seed_transcoder_ci_fn(component_model, torch.zeros(0))
    assert isinstance(ci_fn, ASPDCiFn), f"expected ASPDCiFn, got {type(ci_fn).__name__}"
    assert ci_fn._resid_handle is not None, f"the encoder for {ci_fn.module!r} has no residual hook"
    logger.info(
        f"ASPD: V={tuple(ci_fn.components.V.shape)} W_enc={tuple(ci_fn.W_enc.shape)} "
        f"resid_site={ci_fn.site} top_k={ci_fn.cfg.top_k}"
    )
    return ci_fn


def attach_aspd_ci_fn(component_model: ComponentModel) -> None:
    """Re-attach after `load_state_dict` (the tie is a Python reference, not a tensor). Idempotent."""
    tie_aspd_ci_fn(component_model)


def _seed_encoder(encoder: SharedEncoder, mean: Tensor) -> None:
    """b_dec <- E[r]; W_enc <- unit-norm columns; W_dec <- W_enc^T with unit-norm rows."""
    assert mean.numel() == encoder.d_in, f"calibration for {encoder.site!r} has {mean.numel()} values"
    with torch.no_grad():
        encoder.b_dec.copy_(mean.to(encoder.b_dec.dtype))
        encoder.W_enc.div_(encoder.W_enc.norm(dim=0, keepdim=True).clamp_min(1e-8))
    encoder.tie_decoder()


def seed_aspd_ci_fn(
    component_model: ComponentModel, scale: Tensor | dict[str, Tensor]
) -> SharedEncoder | ASPDCiFnSet:
    """Attach the CI function and initialize every encoder from its site's calibration."""
    ci_fn = tie_aspd_ci_fn(component_model)
    if isinstance(ci_fn, ASPDCiFnSet):
        assert isinstance(scale, dict), "several encoders need one E[r] per site"
        encoders = ci_fn.encoders()
        missing = {e.site for e in encoders.values()} - set(scale)
        assert not missing, f"no calibration for {sorted(missing)}"
        for encoder in encoders.values():
            _seed_encoder(encoder, scale[encoder.site])
        return ci_fn
    assert isinstance(scale, Tensor), "a single encoder needs one [d_act] mean"
    _seed_encoder(ci_fn, scale)
    return ci_fn

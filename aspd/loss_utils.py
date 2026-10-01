"""Shared tensor helpers for the losses: module outputs under a mask, FVU statistics, warmup."""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float
from param_decomp.masks import ComponentsMaskInfo
from torch import Tensor, nn

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE

MaskKind = Literal["stochastic", "ppgd", "unmasked"]


# ---- module outputs from the cached input, no forward -------------------------------------------


def target_module_output(target_module: nn.Module, x: Float[Tensor, "... d_in"]) -> Tensor:
    """`y` with no masking: the FROZEN target module applied to its own cached input."""
    with torch.no_grad():
        return target_module(x)


def masked_module_output(
    components: nn.Module,
    target_module: nn.Module,
    x: Float[Tensor, "... d_in"],
    mask_info: ComponentsMaskInfo,
) -> Tensor:
    """`y` under a VPD mask, computed directly from `x`."""
    components_out = components(
        x,
        mask=mask_info.component_mask,
        weight_delta_and_mask=mask_info.weight_delta_and_mask,
    )
    if mask_info.routing_mask == "all":
        return components_out
    with torch.no_grad():
        target_out = target_module(x)
    return torch.where(mask_info.routing_mask[..., None], components_out, target_out)


# ---- feature reconstruction (Eq. 10) -------------------------------------------------------------


@dataclass(frozen=True)
class FeatureReconConfig:
    mask_kinds: tuple[MaskKind, ...] = ("stochastic",)
    eps: float = 1e-8


def global_fvu(
    pred: Float[Tensor, "... f"], target: Float[Tensor, "... f"], *, eps: float = 1e-8
) -> Float[Tensor, ""]:
    """`||pred - target||^2 / ||target - mean||^2` summed over ALL features at once."""
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    resid = (pred_flat - target_flat).pow(2).sum()
    total = (target_flat - target_flat.mean(dim=0, keepdim=True)).pow(2).sum()
    return resid / total.clamp_min(eps)


def per_feature_fvu(
    pred: Float[Tensor, "... f"],
    target: Float[Tensor, "... f"],
    *,
    eps: float = 1e-8,
    min_var: float = 1e-10,
) -> tuple[Float[Tensor, ""], Float[Tensor, " f"], Bool[Tensor, " f"]]:
    """FVU per latent, averaged over live latents. Returns `(mean, per_feature, live)`."""
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    resid = (pred_flat - target_flat).pow(2).sum(dim=0)
    var = (target_flat - target_flat.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)

    live = var > min_var
    per_feature = resid / var.clamp_min(eps)
    if not live.any():
        return torch.zeros((), device=pred.device), per_feature, live
    return per_feature[live].mean(), per_feature, live


def worst_decile_fvu(
    per_feature: Float[Tensor, " f"], live: Bool[Tensor, " f"]
) -> Float[Tensor, ""]:
    """Mean FVU of the worst-reconstructed 10% of live latents."""
    values = per_feature[live]
    if values.numel() == 0:
        return torch.zeros((), device=per_feature.device)
    k = max(1, int(round(0.1 * values.numel())))
    return values.topk(k).values.mean()


def split_fvu(
    pred: Float[Tensor, "... f"], target: Float[Tensor, "... f"], *, eps: float = 1e-8
) -> dict[str, Tensor]:
    """FVU split over `{j : f_true > 0}` and `{j : f_true = 0}`."""
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    alive = target_flat > 0

    mean_fvu, per_feature, live = per_feature_fvu(pred_flat, target_flat, eps=eps)
    return {
        "fvu": mean_fvu,
        "fvu_global": global_fvu(pred_flat, target_flat, eps=eps),
        "fvu_worst_decile": worst_decile_fvu(per_feature, live),
        "spurious_mass": pred_flat[~alive].pow(2).mean()
        if (~alive).any()
        else torch.zeros((), device=pred.device),
        "alive_frac": alive.float().mean(),
    }


def feature_recon_loss(
    sae_out: MatryoshkaBatchTopKSAE,
    y_masked: Float[Tensor, "... d_out"],
    y_target: Float[Tensor, "... d_out"],
    *,
    eps: float = 1e-8,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    """Eq. 10 for one mask draw. `y_target` is detached; `y_masked` carries gradient."""
    f_true = sae_out.features(y_target.detach()).detach()
    f_masked = sae_out.features(y_masked)
    stats = split_fvu(f_masked, f_true, eps=eps)
    return stats["fvu"], {k: v.detach() for k, v in stats.items()}


def _promote(x: Tensor) -> Tensor:
    """At least fp32, never DOWN from fp64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def _exact(x: Tensor):
    """Autocast disabled for the enclosed block, keyed to `x`'s device."""
    return torch.autocast(device_type=x.device.type, enabled=False)


def site_recon_stats(
    pred: Float[Tensor, "... d"], target: Float[Tensor, "... d"], *, eps: float = 1e-8
) -> dict[str, Tensor]:
    """`fvu` / `mse` / `cossim` of a reconstruction of the SITE activation `y`."""
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    return {
        "fvu": global_fvu(pred_flat, target_flat, eps=eps),
        "mse": (pred_flat - target_flat).pow(2).mean(),
        "cossim": torch.nn.functional.cosine_similarity(
            pred_flat.float(), target_flat.float(), dim=-1
        ).mean(),
    }


# ---- reductions over the footprints (Eq. 9) ------------------------------------------------------


def gbar_from_ci(ci_lower: Float[Tensor, "... c"]) -> Float[Tensor, " c"]:
    return ci_lower.detach().reshape(-1, ci_lower.shape[-1]).mean(dim=0)


def importance_weights(
    gbar: Float[Tensor, " c"], *, eps: float = 1e-12
) -> tuple[Float[Tensor, " c"], Float[Tensor, ""]]:
    """Eq. 9's `sg[.]`: `gbar_c / sum_c gbar_c`. Returns `(weights, gbar_sum)`."""
    assert not gbar.requires_grad, "importance weights must be detached (sg[.]) -- see docstring"
    gbar_sum = gbar.sum()
    return gbar / gbar_sum.clamp_min(eps), gbar_sum


def warmup_scale(step: int, total_steps: int, *, start_frac: float = 0.1,
                 ramp_frac: float = 0.1) -> float:
    """0 until `start_frac` of training, then a linear ramp over `ramp_frac`."""
    start = int(total_steps * start_frac)
    ramp = max(int(total_steps * ramp_frac), 1)
    if step < start:
        return 0.0
    return min(1.0, (step - start) / ramp)

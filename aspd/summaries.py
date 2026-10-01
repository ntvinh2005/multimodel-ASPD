"""Per-component summary statistics logged beside the losses."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

_QUANTILES = (0.1, 0.5, 0.9)
_SUFFIXES = ("p10", "median", "p90")


def component_summary(
    name: str,
    v: Float[Tensor, " c"],
    *,
    alive: Bool[Tensor, " c"] | None = None,
    legacy_median_alive_alias: bool = False,
) -> dict[str, Tensor]:
    """`{name}_p10/_median/_p90/_mean` over all components, plus `_alive` variants."""
    assert v.ndim == 1, f"component_summary expects [C], got {tuple(v.shape)}"
    v = v.detach().float()
    out: dict[str, Tensor] = {}

    def _fill(prefix: str, values: Tensor) -> None:
        qs = torch.quantile(values, torch.tensor(_QUANTILES, device=values.device))
        for suffix, q in zip(_SUFFIXES, qs.unbind()):
            out[f"{prefix}_{suffix}"] = q
        out[f"{prefix}_mean"] = values.mean()

    _fill(name, v)
    if alive is not None and bool(alive.any()):
        _fill(f"{name}_alive", v[alive])
    else:
        zero = torch.zeros((), device=v.device)
        for suffix in (*_SUFFIXES, "mean"):
            out[f"{name}_alive_{suffix}"] = zero
    if legacy_median_alive_alias:
        out[f"{name}_median_alive"] = out[f"{name}_alive_median"]
    return out


def alive_summary(
    gbar: Float[Tensor, " c"], threshold: float
) -> tuple[Bool[Tensor, " c"], dict[str, Tensor]]:
    """The `gbar_c > threshold` mask plus the three scalars that make every `_alive` key readable."""
    alive = gbar > threshold
    return alive, {
        "alive_frac": alive.float().mean(),
        "n_alive": alive.sum().float(),
        "gbar_mean": gbar.mean(),
    }



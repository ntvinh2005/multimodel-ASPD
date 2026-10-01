"""The transcoder parameterization (`component_arch: transcoder`): rank-1 components P_c = u_c v_c^T
plus an input-centring bias b_dec and an output bias b_out, used by PD Transcoder and ASPD.
"""

from typing import Literal, override

import einops
import torch
from jaxtyping import Float
from param_decomp.component_model import ComponentModel
from param_decomp.components import Components, LinearComponents
from param_decomp.masks import WeightDeltaAndMask
from torch import Tensor, nn


class TranscoderLinearComponents(LinearComponents):
    """`LinearComponents` with the transcoder's `b_dec` (encoder centring) and `b_out` (recon bias)."""

    def __init__(self, C: int, d_in: int, d_out: int, bias: Tensor | None = None):
        super().__init__(C, d_in=d_in, d_out=d_out, bias=bias)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.b_out = nn.Parameter(torch.zeros(d_out))

    @override
    def get_component_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        """The raw preact `z_c = (x - b_dec) . W_enc[:, c]`."""
        return einops.einsum(
            x.to(self.V.dtype) - self.b_dec, self.V, "... d_in, d_in C -> ... C"
        )

    @override
    def forward(
        self,
        x: Float[Tensor, "... d_in"],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Core's forward plus `b_out`."""
        component_acts = self.get_component_acts(x)
        if component_acts_cache is not None:
            component_acts_cache["pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            component_acts_cache["post_detach"] = component_acts

        if mask is not None:
            component_acts = component_acts * mask

        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")
        out = out + self.b_out

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = einops.einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out = out + einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... d_out -> ... d_out"
            )

        if self.bias is not None:
            out = out + self.bias

        return out


def _to_transcoder(
    name: str, comp: Components, init: Literal["reference", "unit_norm"] = "reference"
) -> Components:
    assert isinstance(comp, LinearComponents), (
        f"transcoder components are implemented for LinearComponents only; {name} is "
        f"{type(comp).__name__}"
    )
    assert not isinstance(comp, TranscoderLinearComponents), f"{name} is already a transcoder"
    new = TranscoderLinearComponents(
        C=comp.C, d_in=comp.d_in, d_out=comp.d_out, bias=None
    ).to(device=comp.V.device, dtype=comp.V.dtype)
    with torch.no_grad():
        nn.init.kaiming_uniform_(new.V)
        nn.init.kaiming_uniform_(new.U)
        new.U.div_(new.U.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        if init == "unit_norm":
            new.V.div_(new.V.norm(dim=0, keepdim=True).clamp_min(1e-8))
    return new


def install_transcoder_components(
    init: Literal["reference", "unit_norm"] = "reference",
) -> None:
    """Patch `make_components` so `ComponentModel` builds transcoder blocks."""
    import param_decomp.component_model as _cm

    original = getattr(_cm, "_pre_transcoder_make_components", None) or _cm.make_components
    _cm._pre_transcoder_make_components = original  # pyright: ignore[reportAttributeAccessIssue]

    def _make(target_model: nn.Module, module_to_c: dict[str, int]) -> dict[str, Components]:
        return {
            n: _to_transcoder(n, c, init)
            for n, c in original(target_model, module_to_c).items()
        }

    _cm.make_components = _make  # pyright: ignore[reportAttributeAccessIssue]


def transcoder_param_counts(model: ComponentModel) -> dict[str, int]:
    """`{module: V + U + b_dec + b_out}` parameter counts, for the launch log and provenance."""
    return {
        n: c.V.numel() + c.U.numel() + c.b_dec.numel() + c.b_out.numel()
        for n, c in model.components.items()
        if isinstance(c, TranscoderLinearComponents)
    }


@torch.no_grad()
def normalize_decoder_(comp: TranscoderLinearComponents) -> None:
    """Project the radial gradient component out of `U`, then renormalize its rows to unit norm."""
    normed = comp.U / comp.U.norm(dim=-1, keepdim=True)
    if comp.U.grad is not None:
        radial = (comp.U.grad * normed).sum(-1, keepdim=True) * normed
        comp.U.grad -= radial
    comp.U.data = normed

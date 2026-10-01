"""A decomposed matrix's components as an activation source (per-token g_{t,c} v_c^T x_t)."""

from dataclasses import dataclass

import torch
from aspd.sae.sites import HookSite
from jaxtyping import Bool, Float, Int
from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from param_decomp.masks import SamplingType
from torch import Tensor

from aspd.eval.adapters.capture import keep_token_mask
from aspd.eval.adapters.source import SiteBatch, SiteCache


def target_module_bias(model: ComponentModel, module_path: str) -> Tensor | None:
    """The FROZEN target module's own bias `b`, or `None` if it has none."""
    module = model.target_model.get_submodule(module_path)
    return getattr(module, "bias", None)


@dataclass
class VPDComponentSource:

    model: ComponentModel
    module_path: str
    tokenizer: object
    sampling: SamplingType = "continuous"

    def __post_init__(self) -> None:
        assert self.module_path in self.model.components, (
            f"{self.module_path!r} is not decomposed; have {sorted(self.model.components)}"
        )
        assert isinstance(self.components, LinearComponents), (
            "only LinearComponents expose the U/V factorization these evals need"
        )

    @property
    def components(self) -> LinearComponents:
        return self.model.components[self.module_path]  # type: ignore[return-value]

    @property
    def n_latents(self) -> int:
        return self.components.C

    @property
    def write_vectors(self) -> Float[Tensor, "c d_out"]:
        comp = self.components
        assert comp.U.ndim == 2, (
            f"`U` is {tuple(comp.U.shape)}; a per-component write direction needs [C, d_out]."
        )
        return comp.U.detach()

    @property
    def read_vectors(self) -> Float[Tensor, "d_in c"]:
        comp = self.components
        assert comp.V.ndim == 2, f"`V` is {tuple(comp.V.shape)}; needs [d_in, C]."
        return comp.V.detach()

    @property
    def output_site(self) -> HookSite:
        return HookSite(key=f"{self.module_path}:out", module=self.module_path, take="output")

    @torch.no_grad()
    def cache_batch(self, tokens: Int[Tensor, "b l"]) -> SiteCache:
        """`(y, x, mask)`. `x` is the carrier because `ζ` is a function of the module's INPUT."""
        mask = keep_token_mask(tokens, self.tokenizer)

        out = self.model.forward(tokens, cache_type="input")
        x = out.cache[self.module_path]
        assert x.shape[:2] == tokens.shape, (x.shape, tokens.shape)

        return SiteCache(
            site_acts=self.true_site_output(x) * mask[:, :, None],
            carrier=x * mask[:, :, None],
            tokens=tokens,
            token_mask=mask,
        )

    @torch.no_grad()
    def latents(self, cache: SiteCache) -> tuple[Tensor, Tensor]:
        out = self.model.forward(cache.tokens, cache_type="input")
        x = out.cache[self.module_path]
        z = self.components.get_component_acts(x)
        g = self.model.calc_causal_importances({self.module_path: x}, sampling=self.sampling)
        g_c = g.lower_leaky[self.module_path]
        assert g_c.shape == z.shape, (g_c.shape, z.shape)

        assert (g_c >= 0).all(), (
            "causal importance went negative; 's clamp is documented as inert and the "
            "node-effect ranking would silently change meaning -- re-read it before relaxing this"
        )
        mask = cache.token_mask[:, :, None]
        return (g_c * z) * mask, (g_c.clamp_min(0.0) * z) * mask

    def encode_batch(self, tokens: Int[Tensor, "b l"]) -> SiteBatch:
        cache = self.cache_batch(tokens)
        ablation, ranking = self.latents(cache)
        return SiteBatch(
            site_acts=cache.site_acts,
            ablation=ablation,
            ranking=ranking,
            token_mask=cache.token_mask,
        )

    def true_site_output(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """`W x + b` in PD's `[d_out, d_in]` convention -- the frozen module's own output."""
        weight = self.model.target_weight(self.module_path)
        y = x.to(weight.dtype) @ weight.t()
        bias = target_module_bias(self.model, self.module_path)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y

    @torch.no_grad()
    def masked_site_output(
        self,
        x: Float[Tensor, "... d_in"],
        component_mask: Float[Tensor, "... c"],
        *,
        weight_delta_mask: float = 1.0,
    ) -> Float[Tensor, "... d_out"]:
        delta = self.model.calc_weight_deltas()[self.module_path]
        y = self.components.forward(
            x,
            mask=component_mask,
            weight_delta_and_mask=(
                delta,
                torch.full(x.shape[:-1], weight_delta_mask, device=x.device, dtype=x.dtype),
            ),
        )
        return y


def selection_mask(n_latents: int, indices: Tensor, device: torch.device) -> Bool[Tensor, " k"]:
    mask = torch.zeros(n_latents, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask

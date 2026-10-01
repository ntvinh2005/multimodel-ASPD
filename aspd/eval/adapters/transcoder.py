"""A frozen transcoder as an activation source."""

from dataclasses import dataclass

import torch
from aspd.sae.sites import HookSite, SitePair
from aspd.sae.transcoder import MatryoshkaBatchTopKTranscoder
from jaxtyping import Bool, Float, Int
from param_decomp.components import LinearComponents
from torch import Tensor, nn

from aspd.eval.adapters.capture import capture_site_pair
from aspd.eval.adapters.dictionary import assert_deterministic_encode
from aspd.eval.adapters.source import SiteBatch, SiteCache


def encoder_site(sites: SitePair) -> HookSite:
    """The activation the transcoder's ENCODER reads, as a hookable site."""
    return HookSite(
        key=sites.input_site,
        module=sites.input_module or sites.input_site,
        take=sites.input_take,
    )


def output_site(sites: SitePair) -> HookSite:
    return HookSite(key=sites.output_site, module=sites.output_site, take="output")


@dataclass
class TranscoderSource:

    transcoder: MatryoshkaBatchTopKTranscoder
    sites: SitePair
    model: nn.Module
    tokenizer: object

    def __post_init__(self) -> None:
        assert_deterministic_encode(self.transcoder)

    @property
    def n_latents(self) -> int:
        return self.transcoder.cfg.n_features

    @property
    def write_vectors(self) -> Float[Tensor, "f d_out"]:
        """`W_dec[j]` -- exact, and for a stronger reason than the dictionary's."""
        return self.transcoder.W_dec.detach()

    @property
    def read_vectors(self) -> Float[Tensor, "d_in f"]:
        return self.transcoder.W_enc.detach()

    @torch.no_grad()
    def cache_batch(self, tokens: Int[Tensor, "b l"]) -> SiteCache:
        """`(y, x, mask)` from ONE forward. `x` is the carrier; `y` is the probe's activation."""
        acts, mask = capture_site_pair(
            self.model, [encoder_site(self.sites), output_site(self.sites)], tokens, self.tokenizer
        )
        return SiteCache(
            site_acts=acts[self.sites.output_site],
            carrier=acts[self.sites.input_site],
            tokens=tokens,
            token_mask=mask,
        )

    @torch.no_grad()
    def latents(self, cache: SiteCache) -> tuple[Tensor, Tensor]:
        f = self.transcoder.features(cache.carrier) * cache.token_mask[:, :, None]
        return f, f

    def encode_batch(self, tokens: Int[Tensor, "b l"]) -> SiteBatch:
        cache = self.cache_batch(tokens)
        ablation, ranking = self.latents(cache)
        return SiteBatch(
            site_acts=cache.site_acts,
            ablation=ablation,
            ranking=ranking,
            token_mask=cache.token_mask,
        )

    @torch.no_grad()
    def reconstruction(self, cache: SiteCache) -> Float[Tensor, "b l d_out"]:
        """`y_hat` at the probe's site -- what `ce_kl`'s site-reconstruction and splice consume."""
        return self.transcoder.reconstruct(cache.carrier) * cache.token_mask[:, :, None]


def transcoder_components(tc: MatryoshkaBatchTopKTranscoder) -> LinearComponents:
    """`(W_enc, W_dec)` as core's OWN rank-1 components: `V = W_enc`, `U = W_dec`."""
    comp = LinearComponents(
        C=tc.cfg.n_features, d_in=tc.cfg.d_in, d_out=tc.d_out, bias=None
    ).to(device=tc.W_enc.device, dtype=tc.W_enc.dtype)
    with torch.no_grad():
        comp.V.copy_(tc.W_enc.detach())
        comp.U.copy_(tc.W_dec.detach())
    return comp.requires_grad_(False)


def bias_residue(
    tc: MatryoshkaBatchTopKTranscoder, selection: Int[Tensor, " s"]
) -> Float[Tensor, " d_out"]:
    """`sum_{j in S} (b_dec . W_enc[:, j]) W_dec[j]` -- what a WEIGHT-ONLY edit cannot remove."""
    enc = tc.W_enc.detach()[:, selection].float()  # [d_in, s]
    dec = tc.W_dec.detach()[selection].float()  # [s, d_out]
    return (tc.b_dec.detach().float() @ enc) @ dec


class _Cached:
    """`ComponentModel.forward`'s return shape, reduced to the one field attribution reads."""

    def __init__(self, cache: dict[str, Tensor]):
        self.cache = cache


class _CausalImportances:
    """`calc_causal_importances`' return shape, reduced to `lower_leaky`."""

    def __init__(self, lower_leaky: dict[str, Tensor]):
        self.lower_leaky = lower_leaky


class TranscoderComponentModel:
    """A transcoder presented through the six-method surface `attr_edit` uses on `ComponentModel`."""

    def __init__(
        self,
        transcoder: MatryoshkaBatchTopKTranscoder,
        target_model: nn.Module,
        module_path: str,
        sites: SitePair,
        tokenizer: object,
    ):
        assert_deterministic_encode(transcoder)
        assert sites.output_site == module_path, (sites.output_site, module_path)
        self.transcoder = transcoder
        self.target_model = target_model
        self.module_path = module_path
        self.sites = sites
        self.tokenizer = tokenizer
        self.components = {module_path: transcoder_components(transcoder)}

    @property
    def edit_is_exact(self) -> bool:
        """Whether `W_dec[j] (x) W_enc[:, j]` is a map on the tensor the decomposed weight consumes."""
        return self.sites.input_take == "output"

    @torch.no_grad()
    def forward(self, tokens: Int[Tensor, "b l"], cache_type: str = "input") -> _Cached:
        assert cache_type == "input", (
            f"cache_type={cache_type!r}: this view caches the ENCODER's activation, which is the "
            "only thing attribution asks it for"
        )
        acts, _ = capture_site_pair(
            self.target_model,
            [encoder_site(self.sites), output_site(self.sites)],
            tokens,
            self.tokenizer,
        )
        centred = acts[self.sites.input_site] - self.transcoder.b_dec
        return _Cached({self.module_path: centred})

    @torch.no_grad()
    def calc_causal_importances(
        self, acts: dict[str, Tensor], sampling: str = "continuous"
    ) -> _CausalImportances:
        """`1[relu(z) > threshold]`, so that `g * z == f` exactly on both sides of the threshold."""
        centred = acts[self.module_path]
        z = self.components[self.module_path].get_component_acts(centred)
        active = torch.relu(z) > self.transcoder.threshold
        return _CausalImportances({self.module_path: active.to(z.dtype)})

    def calc_weight_deltas(self) -> dict[str, Tensor]:
        """No `m_Delta` term: a transcoder has no faithfulness residual to carry."""
        comp = self.components[self.module_path]
        return {
            self.module_path: torch.zeros(
                comp.d_out, comp.d_in, device=comp.U.device, dtype=comp.U.dtype
            )
        }

    def target_weight(self, module_path: str) -> Float[Tensor, "d_out d_in"]:
        """The FROZEN module's weight in PD's `[d_out, d_in]` convention."""
        module = self.target_model.get_submodule(module_path)
        weight = module.weight.detach()
        comp = self.components[module_path]
        assert comp.d_in != comp.d_out, (
            "d_in == d_out makes the weight's orientation unrecoverable from its shape; add a "
            "module-type branch as `attr_edit.edit.patched_target_weight` has"
        )
        return weight if weight.shape == (comp.d_out, comp.d_in) else weight.t()

    def get_submodule(self, path: str) -> nn.Module:
        return self.target_model.get_submodule(path)


def resolve_transcoder_steps(tc_dir, spec: str) -> list[int]:
    """Which transcoder checkpoints an eval sweeps. `all` means the LAST one, deliberately."""
    import re

    from aspd.sae.transcoder import transcoder_steps

    available = transcoder_steps(tc_dir)
    if spec.strip() == "all":
        return available[-1:]
    wanted = [int(s) for s in re.split(r"[,:\s]+", spec.strip()) if s]
    missing = sorted(set(wanted) - set(available))
    assert not missing, f"no transcoder checkpoint(s) {missing} in {tc_dir}; have {available}"
    return wanted


def selection_mask(n_latents: int, indices: Tensor, device: torch.device) -> Bool[Tensor, " k"]:
    mask = torch.zeros(n_latents, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask

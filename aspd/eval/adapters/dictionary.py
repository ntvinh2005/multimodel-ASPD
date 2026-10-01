"""A trained SAE as an activation source."""

from dataclasses import dataclass

import torch
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from aspd.sae.sites import HookSite
from jaxtyping import Float, Int
from torch import Tensor, nn

from aspd.eval.adapters.capture import capture_site
from aspd.eval.adapters.source import SiteBatch, SiteCache


def assert_deterministic_encode(sae: MatryoshkaBatchTopKSAE) -> None:
    assert not sae.training, (
        "SAE is in train mode; `features()` would take the batch-coupled BatchTopK path"
    )
    assert float(sae.threshold) > 0.0, (
        "SAE threshold is 0 -- the JumpReLU threshold EMA never ran, so every latent reads active"
    )


@dataclass
class SAEDictionarySource:
    """A dictionary at `site`, encoding whatever activation that site carries."""

    sae: MatryoshkaBatchTopKSAE
    site: HookSite
    model: nn.Module
    tokenizer: object

    def __post_init__(self) -> None:
        assert_deterministic_encode(self.sae)

    @property
    def n_latents(self) -> int:
        return self.sae.cfg.n_features

    @property
    def write_vectors(self) -> Float[Tensor, "f d"]:
        return self.sae.W_dec.detach()

    @property
    def read_vectors(self) -> Float[Tensor, "d f"]:
        return self.sae.W_enc.detach()

    @torch.no_grad()
    def cache_batch(self, tokens: Int[Tensor, "b l"]) -> SiteCache:
        """A dictionary encodes the site activation itself, so carrier IS `site_acts`."""
        acts, mask = capture_site(self.model, self.site, tokens, self.tokenizer)
        return SiteCache(site_acts=acts, carrier=acts, tokens=tokens, token_mask=mask)

    @torch.no_grad()
    def latents(self, cache: SiteCache) -> tuple[Tensor, Tensor]:
        features = self.sae.features(cache.carrier) * cache.token_mask[:, :, None]
        return features, features

    def encode_batch(self, tokens: Int[Tensor, "b l"]) -> SiteBatch:
        cache = self.cache_batch(tokens)
        ablation, ranking = self.latents(cache)
        return SiteBatch(
            site_acts=cache.site_acts,
            ablation=ablation,
            ranking=ranking,
            token_mask=cache.token_mask,
        )

"""`ActivationSource`: the interface the evaluations read activations and ablations through."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


@dataclass(frozen=True)
class SiteBatch:
    """One token batch resolved at a site. Rows are sequences, columns positions."""

    site_acts: Float[Tensor, "b l d"]

    ablation: Float[Tensor, "b l k"]
    """What ablation removes per unit write-vector: `f_j` (SAE) or `ζ_c = g_c z_c` (components)."""

    ranking: Float[Tensor, "b l k"]
    """The node-effect activation: `f_j` (SAE) or `α_c = max(g_c, 0)·z_c` (components)."""

    token_mask: Bool[Tensor, "b l"]
    """True where the position is kept. `site_acts` is already zeroed off it."""

    def __post_init__(self) -> None:
        b, ll, _ = self.site_acts.shape
        assert self.ablation.shape[:2] == (b, ll), self.ablation.shape
        assert self.ranking.shape == self.ablation.shape
        assert self.token_mask.shape == (b, ll), self.token_mask.shape


@dataclass(frozen=True)
class SiteCache:
    """What SCR / TPP keep per class, and the reason it is TWO tensors."""

    site_acts: Float[Tensor, "b l d_out"]
    """Probe input: the activation at the site being evaluated. Zeroed at masked positions."""

    carrier: Float[Tensor, "b l d_in"]
    """What latents are recomputed from, for sources that need no forward (a dictionary)."""

    tokens: Int[Tensor, "b l"]
    """The input ids, kept because a **read-site** CI fn cannot be replayed from `carrier`."""

    token_mask: Bool[Tensor, "b l"]

    def to(self, device: torch.device) -> "SiteCache":
        return SiteCache(
            self.site_acts.to(device),
            self.carrier.to(device),
            self.tokens.to(device),
            self.token_mask.to(device),
        )

    def __getitem__(self, s: slice) -> "SiteCache":
        return SiteCache(self.site_acts[s], self.carrier[s], self.tokens[s], self.token_mask[s])

    def __len__(self) -> int:
        return self.site_acts.shape[0]


@runtime_checkable
class ActivationSource(Protocol):
    """An SAE dictionary or a VPD decomposition, seen through the same hole."""

    @property
    def n_latents(self) -> int:
        """`F` for a dictionary, `C` for a decomposition."""
        ...

    @property
    def write_vectors(self) -> Float[Tensor, "k d"]:
        """Row `i`: what latent `i` adds to the site per unit of `ablation`."""
        ...

    @property
    def read_vectors(self) -> Float[Tensor, "d_in k"]:
        ...

    def cache_batch(self, tokens: Tensor) -> SiteCache:
        """Run the model on `tokens` and resolve the site. Pad masking applied here."""
        ...

    def latents(self, cache: SiteCache) -> tuple[Tensor, Tensor]:
        ...

    def encode_batch(self, tokens: Tensor) -> SiteBatch:
        ...


def pooled(acts: Float[Tensor, "b l ..."], token_mask: Bool[Tensor, "b l"]) -> Tensor:
    keep = token_mask.to(acts.dtype)
    n = keep.sum(dim=1).clamp_min(1.0)
    while keep.ndim < acts.ndim:
        keep = keep.unsqueeze(-1)
    summed = (acts * keep).sum(dim=1)
    return summed / n.reshape(-1, *([1] * (summed.ndim - 1)))


def ablate(
    site_acts: Float[Tensor, "b l d"],
    ablation: Float[Tensor, "b l k"],
    write_vectors: Float[Tensor, "k d"],
    selected: Bool[Tensor, " k"],
) -> Float[Tensor, "b l d"]:
    assert selected.dtype == torch.bool and selected.shape == (write_vectors.shape[0],)
    if not selected.any():
        return site_acts.clone()
    idx = selected.nonzero(as_tuple=True)[0]
    removed = ablation[..., idx] @ write_vectors[idx]
    return site_acts - removed

"""A trained SAE at one hook site, seen through one interface (encode, decode, features)."""

from abc import ABC, abstractmethod
from typing import Literal

from jaxtyping import Bool, Float
from torch import Tensor

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from aspd.sae.sites import SitePair
from aspd.sae.transcoder import MatryoshkaBatchTopKTranscoder


class DictionaryAdapter(ABC):
    """One frozen dictionary at one hook site. All downstream eval consumes this, not the SAE."""

    site_path: str
    """The site KEY this dictionary reads and reconstructs -- a module path on all but one pair."""

    hook_module: str = ""
    """Module to hook to obtain this dictionary's signal. Empty means `site_path` itself."""

    @property
    def hook(self) -> str:
        """The module path to attach a capture or splice to -- `hook_module`, or `site_path`."""
        return self.hook_module or self.site_path

    take: Literal["input", "output"] = "output"
    """Whether `site_path`'s INPUT or its OUTPUT is the signal -- i.e. `SitePair.input_take`."""

    @property
    def write_hook(self) -> str:
        """Module whose activation `decode` LANDS IN -- where a splice replaces a tensor."""
        return self.hook

    @property
    def write_take(self) -> Literal["input", "output"]:
        """Whether `write_hook`'s INPUT or OUTPUT is the tensor `decode` reconstructs."""
        return self.take

    @property
    def is_cross_site(self) -> bool:
        """True when `decode` does not land in the space `encode` read from."""
        return (self.write_hook, self.write_take) != (self.hook, self.take)

    role: str

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    @abstractmethod
    def encode(self, acts: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        """Inference-path features (deterministic EMA threshold), NOT training-time BatchTopK."""

    @abstractmethod
    def decode(self, features: Float[Tensor, "... f"]) -> Float[Tensor, "... d"]: ...

    @abstractmethod
    def decoder_rows(self) -> Float[Tensor, "f d"]:
        """Unit-norm decoder directions -- the `W_dec` rows the logit lens unembeds."""

    @abstractmethod
    def group_boundaries(self) -> list[int] | None:
        """Cumulative Matryoshka prefix boundaries `[0, m_1, ..., F]`, or None for a flat dict."""

    def firings(self, acts: Float[Tensor, "... d"]) -> Bool[Tensor, "... f"]:
        return self.encode(acts) > 0

    def reconstruct(self, acts: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return self.decode(self.encode(acts))


class SAEDictionary(DictionaryAdapter):
    """`DictionaryAdapter` over one `MatryoshkaBatchTopKSAE`. The SAE must already be frozen."""

    def __init__(
        self,
        sae: MatryoshkaBatchTopKSAE,
        site_path: str,
        role: str,
        take: Literal["input", "output"] = "output",
        hook_module: str = "",
    ) -> None:
        assert not any(p.requires_grad for p in sae.parameters()), (
            "SAEDictionary wraps a FROZEN extractor; got a dictionary with trainable params. "
            "Call `.freeze()` (or load via `load_sae_pair`, which freezes) before wrapping."
        )
        assert role in ("in", "out", "diff"), f"role must be in/out/diff, got {role!r}"
        self.sae = sae
        self.site_path = site_path
        self.role = role
        self.take = take
        self.hook_module = hook_module or site_path

    @property
    def n_features(self) -> int:
        return self.sae.cfg.n_features

    def encode(self, acts: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        return self.sae.features(acts.to(self.sae.W_enc.dtype))

    def decode(self, features: Float[Tensor, "... f"]) -> Float[Tensor, "... d"]:
        return self.sae.decode(features)

    def decoder_rows(self) -> Float[Tensor, "f d"]:
        return self.sae.W_dec

    def group_boundaries(self) -> list[int]:
        return self.sae.group_indices


class TranscoderDictionary(DictionaryAdapter):
    """`DictionaryAdapter` over a frozen transcoder: encodes at one site, decodes into ANOTHER."""

    def __init__(
        self,
        transcoder: MatryoshkaBatchTopKTranscoder,
        site_path: str,
        encoder_module: str,
        encoder_take: Literal["input", "output"],
        role: str = "transcoder",
    ) -> None:
        assert not any(p.requires_grad for p in transcoder.parameters()), (
            "TranscoderDictionary wraps a FROZEN extractor; got trainable params. Load via "
            "`load_transcoder`, which freezes."
        )
        assert not transcoder.training, "transcoder must be in eval mode"
        self.transcoder = transcoder
        self.site_path = site_path
        self.role = role
        self.take = encoder_take
        self.hook_module = encoder_module

    @property
    def write_hook(self) -> str:
        """The DECOMPOSED module — `site_path` here is a real module path, not a capture key."""
        return self.site_path

    @property
    def write_take(self) -> Literal["output"]:
        """A transcoder reconstructs the module's OUTPUT, whichever site its encoder read."""
        return "output"

    @property
    def n_features(self) -> int:
        return self.transcoder.cfg.n_features

    def encode(self, acts: Float[Tensor, "... d_in"]) -> Float[Tensor, "... f"]:
        return self.transcoder.features(acts.to(self.transcoder.W_enc.dtype))

    def decode(self, features: Float[Tensor, "... f"]) -> Float[Tensor, "... d_out"]:
        """Lands in `d_out`, NOT in `site_path`'s space — see the class docstring."""
        return self.transcoder.decode(features)

    def decoder_rows(self) -> Float[Tensor, "f d_out"]:
        """`W_dec` rows, in the decomposed module's OUTPUT space."""
        return self.transcoder.W_dec

    def group_boundaries(self) -> list[int]:
        return self.transcoder.group_indices


def transcoder_dictionary(
    transcoder: MatryoshkaBatchTopKTranscoder, sites: SitePair
) -> TranscoderDictionary:
    """The adapter for a transcoder trained on `sites`, labelled by the module it decomposes."""
    return TranscoderDictionary(
        transcoder,
        site_path=sites.output_site,
        encoder_module=sites.capture_modules[sites.input_site],
        encoder_take=sites.takes[sites.input_site],
    )


def sae_dictionaries_from_pair(
    saes: dict[str, MatryoshkaBatchTopKSAE], sites: SitePair
) -> dict[str, SAEDictionary]:
    """`{"in": ..., "out": ...}` from a loaded pair, so downstream code addresses by role."""
    modules = sites.capture_modules
    return {
        "in": SAEDictionary(
            saes[sites.input_site],
            sites.input_site,
            "in",
            sites.input_take,
            hook_module=modules[sites.input_site],
        ),
        "out": SAEDictionary(
            saes[sites.output_site],
            sites.output_site,
            "out",
            "output",
            hook_module=modules[sites.output_site],
        ),
    }

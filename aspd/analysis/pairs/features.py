"""Pretrained SAEs (SAELens / Neuronpedia) as pair endpoints.

A feature's read direction is its encoder column and its write direction its decoder row; read
directions have the intervening LayerNorm gain folded in so they live in the component's space.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

from aspd.analysis.pairs.sae_config import (
    SITES,
    Release,
    neuronpedia_id,
    release_by_key,
    release_layers,
    releases_for_model,
    sae_id,
)
from aspd.analysis.pairs.spaces import Side, Space, site_space
from aspd.analysis.pairs.weights import Directions


def centre(mat: Tensor) -> Tensor:
    """Project off the all-ones direction -- `center_writing_weights`, exactly (verified above)."""
    return mat - mat.mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class NormSpec:
    """The LayerNorm a role reads through, and the residual site that norm's input is."""

    key: str  # safetensors key, `{layer}` substituted
    site: str
    centres: bool  # LayerNorm centres its input; RMSNorm does not
    unit_offset: bool  # Gemma's RMSNorm scales by `(1 + w)`, not `w`


_GPT2_NORMS = {
    "attn.q": NormSpec("h.{layer}.ln_1.weight", "resid_pre", True, False),
    "attn.k": NormSpec("h.{layer}.ln_1.weight", "resid_pre", True, False),
    "attn.v": NormSpec("h.{layer}.ln_1.weight", "resid_pre", True, False),
    "mlp.in": NormSpec("h.{layer}.ln_2.weight", "resid_mid", True, False),
}
_GEMMA_NORMS = {
    "attn.q": NormSpec("model.layers.{layer}.input_layernorm.weight", "resid_pre", False, True),
    "attn.k": NormSpec("model.layers.{layer}.input_layernorm.weight", "resid_pre", False, True),
    "attn.v": NormSpec("model.layers.{layer}.input_layernorm.weight", "resid_pre", False, True),
    "mlp.gate": NormSpec("model.layers.{layer}.pre_feedforward_layernorm.weight", "resid_mid", False, True),
    "mlp.up": NormSpec("model.layers.{layer}.pre_feedforward_layernorm.weight", "resid_mid", False, True),
}


def norms_for_model(model_name: str) -> dict[str, NormSpec]:
    return _GEMMA_NORMS if "gemma" in model_name else _GPT2_NORMS


class TargetNorms:
    """The target's LayerNorm gains, sliced out of its safetensors without loading the model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._cache: dict[str, Tensor] = {}
        self._shards: dict[str, str] | None = None

    def shard_of(self, key: str) -> str:
        from huggingface_hub import file_exists, hf_hub_download

        if self._shards is None:
            index = "model.safetensors.index.json"
            if file_exists(self.model_name, index):
                import json
                from pathlib import Path

                path = hf_hub_download(self.model_name, index)
                self._shards = json.loads(Path(path).read_text())["weight_map"]
            else:
                self._shards = {}
        return hf_hub_download(self.model_name, self._shards.get(key, "model.safetensors"))

    def gain(self, spec: NormSpec, layer: int) -> Tensor:
        """`[d_model]`, already `1 + w` where the architecture stores the offset form."""
        key = spec.key.format(layer=layer)
        if key not in self._cache:
            from safetensors import safe_open

            with safe_open(self.shard_of(key), "pt") as f:
                # `safe_open` exposes `.keys()` but no `__contains__`; bind it before testing.
                present = set(f.keys())
                assert key in present, f"{key!r} is not in {self.model_name}'s safetensors"
                w = f.get_tensor(key).to(torch.float32)
            self._cache[key] = w + 1.0 if spec.unit_offset else w
        return self._cache[key]


def fold_layernorm(mat: Tensor, gain: Tensor, *, centres: bool) -> Tensor:
    """Read directions `[C, d]` re-expressed in the LayerNorm's INPUT space."""
    assert mat.shape[-1] == gain.numel(), f"gain is {gain.numel()}d, directions are {mat.shape[-1]}d"
    out = mat * gain
    return centre(out) if centres else out


@dataclass(frozen=True)
class SaeEndpoint:
    """One loaded dictionary, addressed as `<release key>:L<layer>`."""

    key: str
    model: str
    release: Release
    layer: int
    hook: str
    d_sae: int
    space: Space
    neuronpedia: str | None
    centred: bool  # basis: fit on mean-centred (TransformerLens) activations

    @property
    def site(self) -> str:
        return self.release.site

    @property
    def label(self) -> str:
        return f"L{self.layer} {SITES[self.site].label} · {self.release.label}"


def endpoint_key(release_key: str, layer: int) -> str:
    return f"{release_key}:L{layer}"


def parse_endpoint_key(key: str) -> tuple[str, int]:
    release_key, _, layer = key.rpartition(":L")
    assert release_key and layer.isdigit(), f"{key!r} is not a `<release>:L<layer>` SAE endpoint"
    return release_key, int(layer)


def is_sae_key(key: str) -> bool:
    release_key, _, layer = key.rpartition(":L")
    return bool(release_key) and layer.isdigit()


def catalogue(model_name: str) -> list[dict[str, object]]:
    """Every SAE this model can offer, WITHOUT downloading anything."""
    return [
        {
            "key": r.key,
            "site": r.site,
            "site_label": SITES[r.site].label,
            "release": r.release,
            "label": r.label,
            "layers": release_layers(r),
            "default": r.key == next(x.key for x in releases_for_model(model_name) if x.site == r.site),
        }
        for r in releases_for_model(model_name)
    ]


#: How many whole `sae_lens` modules to keep loaded. Only the prompt path needs one at a time.
_KEEP_SAES = 2


class SaeStore:
    """Lazily loaded pretrained SAEs, cached like `RunWeights` caches component blocks."""

    def __init__(self, model_name: str, *, cache_bytes: int = 6 << 30):
        self.model_name = model_name
        self.available = bool(releases_for_model(model_name))
        self._cache_bytes = cache_bytes
        self._used = 0
        self._endpoints: dict[str, SaeEndpoint] = {}
        self._dirs: OrderedDict[tuple[str, Side], Directions] = OrderedDict()
        self._saes: OrderedDict[str, object] = OrderedDict()

    def endpoint(self, key: str) -> SaeEndpoint:
        if key not in self._endpoints:
            self._load(key)
        return self._endpoints[key]

    def sae(self, key: str, *, keep: int = 2) -> object:
        """The loaded `sae_lens` SAE, for `encode`. Only the prompt path needs this."""
        hit = self._saes.get(key)
        if hit is not None:
            self._saes.move_to_end(key)
            return hit
        self._load(key)
        self._trim_saes(keep)
        return self._saes[key]

    def _trim_saes(self, keep: int = _KEEP_SAES) -> None:
        """Evict whole SAE modules down to `keep`."""
        while len(self._saes) > keep:
            self._saes.popitem(last=False)

    def directions(self, key: str, side: Side) -> Directions:
        hit = self._dirs.get((key, side))
        if hit is not None:
            self._dirs.move_to_end((key, side))
            return hit
        self._load(key)
        return self._dirs[(key, side)]

    def _load(self, key: str) -> None:
        from sae_lens import SAE

        release_key, layer = parse_endpoint_key(key)
        rel = release_by_key(self.model_name, release_key)
        assert layer in release_layers(rel), f"{rel.release} has no layer {layer}"
        sae = SAE.from_pretrained(release=rel.release, sae_id=sae_id(rel, layer), device="cpu")
        meta = sae.cfg.metadata
        site = SITES[rel.site]
        want_hook = site.hook.format(layer=layer)
        assert meta.hook_name == want_hook, (
            f"{rel.release}/{sae_id(rel, layer)} is trained on {meta.hook_name!r}, but "
            f"site {rel.site!r} declares {want_hook!r} -- the site table is wrong"
        )
        # The basis is the SAE's own declaration, never inferred from its weights.
        kwargs = meta.model_from_pretrained_kwargs or {}
        assert not kwargs.get("fold_ln"), (
            f"{rel.release} declares fold_ln; its read/write weights are not in the model's basis "
            "and no correction here covers that"
        )
        d_model, d_sae = sae.W_enc.shape
        head_dim = _head_dim(self.model_name) if site.space_suffix else None
        self._endpoints[key] = SaeEndpoint(
            key=key,
            model=self.model_name,
            release=rel,
            layer=layer,
            hook=str(meta.hook_name),
            d_sae=int(d_sae),
            space=site_space(site.space_suffix, layer, int(d_model), head_dim),
            neuronpedia=neuronpedia_id(rel, layer),
            centred=bool(kwargs.get("center_writing_weights")),
        )
        self._saes[key] = sae
        self._trim_saes()
        for side, mat in (("read", sae.W_enc.t()), ("write", sae.W_dec)):
            self._put(key, side, mat.detach().to(torch.float32).contiguous())  # pyright: ignore[reportArgumentType]

    def _put(self, key: str, side: Side, mat: Tensor) -> None:
        old = self._dirs.get((key, side))
        if old is not None:
            self._used -= old.mat.element_size() * old.mat.nelement()
        self._dirs[(key, side)] = Directions(mat=mat, norms=mat.norm(dim=1))
        self._used += mat.element_size() * mat.nelement()
        while self._used > self._cache_bytes and len(self._dirs) > 2:
            evicted_key, evicted = self._dirs.popitem(last=False)
            self._used -= evicted.mat.element_size() * evicted.mat.nelement()
            self._saes.pop(evicted_key[0], None)


def _head_dim(model_name: str) -> int:
    from aspd.analysis.pairs.spaces import head_dim_for_model

    head_dim = head_dim_for_model(model_name)
    assert head_dim is not None, f"no head_dim known for {model_name}, needed for a head-structured site"
    return head_dim

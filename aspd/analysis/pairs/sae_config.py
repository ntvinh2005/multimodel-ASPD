"""Which pretrained SAE release covers which site of which model."""

import re
from dataclasses import dataclass

from aspd.analysis.pairs.spaces import RESID


@dataclass(frozen=True)
class Site:
    """One point in a transformer block that an SAE can be trained on."""

    name: str
    label: str
    space_suffix: str | None  # None = the residual basis
    hook: str


SITES: dict[str, Site] = {
    "resid_pre": Site("resid_pre", "residual stream, pre-block", None, "blocks.{layer}.hook_resid_pre"),
    "resid_mid": Site("resid_mid", "residual stream, post-attention", None, "blocks.{layer}.hook_resid_mid"),
    "resid_post": Site("resid_post", "residual stream, post-block", None, "blocks.{layer}.hook_resid_post"),
    "mlp_out": Site("mlp_out", "MLP output", None, "blocks.{layer}.hook_mlp_out"),
    "attn_out": Site("attn_out", "attention output, post-W_O", None, "blocks.{layer}.hook_attn_out"),
    "attn_z": Site("attn_z", "attention z, pre-W_O", "attn.z", "blocks.{layer}.attn.hook_z"),
}


@dataclass(frozen=True)
class Release:
    """One pretrained dictionary family, addressable per layer."""

    key: str  # what the API and UI pass around; unique within a model
    site: str
    release: str  # sae_lens release key
    id_template: str  # sae_lens sae_id, with `{layer}` substituted
    label: str


#: Per model, in offer order -- the FIRST release of each site is that site's default.
MODEL_RELEASES: dict[str, list[Release]] = {
    "openai-community/gpt2": [
        Release("mlp_out_oai32k", "mlp_out", "gpt2-small-mlp-out-v5-32k",
                "blocks.{layer}.hook_mlp_out", "OpenAI v5, 32k"),
        Release("mlp_out_oai128k", "mlp_out", "gpt2-small-mlp-out-v5-128k",
                "blocks.{layer}.hook_mlp_out", "OpenAI v5, 128k"),
        Release("mlp_out_tm", "mlp_out", "gpt2-small-mlp-tm",
                "blocks.{layer}.hook_mlp_out", "McGrath"),
        Release("resid_pre_jb", "resid_pre", "gpt2-small-res-jb",
                "blocks.{layer}.hook_resid_pre", "Bloom (res-jb)"),
        Release("resid_mid_oai32k", "resid_mid", "gpt2-small-resid-mid-v5-32k",
                "blocks.{layer}.hook_resid_mid", "OpenAI v5, 32k"),
        Release("resid_post_oai32k", "resid_post", "gpt2-small-resid-post-v5-32k",
                "blocks.{layer}.hook_resid_post", "OpenAI v5, 32k"),
        Release("attn_z_kk", "attn_z", "gpt2-small-hook-z-kk",
                "blocks.{layer}.hook_z", "Kissane et al."),
        Release("attn_out_oai32k", "attn_out", "gpt2-small-attn-out-v5-32k",
                "blocks.{layer}.hook_attn_out", "OpenAI v5, 32k"),
    ],
    "google/gemma-2-2b": [
        Release("mlp_out_gs16k", "mlp_out", "gemma-scope-2b-pt-mlp-canonical",
                "layer_{layer}/width_16k/canonical", "Gemma-Scope 16k"),
        Release("resid_post_gs16k", "resid_post", "gemma-scope-2b-pt-res-canonical",
                "layer_{layer}/width_16k/canonical", "Gemma-Scope 16k"),
        Release("attn_z_gs16k", "attn_z", "gemma-scope-2b-pt-att-canonical",
                "layer_{layer}/width_16k/canonical", "Gemma-Scope 16k"),
    ],
}

_MODEL_ALIASES = {"gpt2": "openai-community/gpt2", "gpt2-small": "openai-community/gpt2"}


def releases_for_model(model_name: str) -> list[Release]:
    """Every release configured for a model, or `[]` -- an unconfigured model is not an error."""
    return MODEL_RELEASES.get(_MODEL_ALIASES.get(model_name, model_name), [])


def release_by_key(model_name: str, key: str) -> Release:
    for r in releases_for_model(model_name):
        if r.key == key:
            return r
    raise AssertionError(f"no SAE release {key!r} configured for {model_name}")


def _catalogue():
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    return get_pretrained_saes_directory()


def release_layers(rel: Release) -> list[int]:
    """Layers this release covers, read from sae_lens's catalogue by matching `id_template`."""
    entry = _catalogue().get(rel.release)
    assert entry is not None, f"{rel.release!r} is not in the installed sae_lens catalogue"
    pattern = re.compile("^" + re.escape(rel.id_template).replace(r"\{layer\}", r"(\d+)") + "$")
    layers = sorted({int(m.group(1)) for sid in entry.saes_map if (m := pattern.match(sid))})
    assert layers, f"{rel.id_template!r} matched no sae_id of release {rel.release!r}"
    return layers


def sae_id(rel: Release, layer: int) -> str:
    return rel.id_template.format(layer=layer)


def neuronpedia_id(rel: Release, layer: int) -> str | None:
    """`gpt2-small/6-res-jb`, from the catalogue -- the feature card's link and API key."""
    entry = _catalogue().get(rel.release)
    assert entry is not None, f"{rel.release!r} is not in the installed sae_lens catalogue"
    return (entry.neuronpedia_id or {}).get(sae_id(rel, layer))


def space_key(rel: Release, layer: int) -> str:
    """The space key this release's site occupies at `layer` -- what makes a pairing meaningful."""
    suffix = SITES[rel.site].space_suffix
    return RESID if suffix is None else f"L{layer}.{suffix}"

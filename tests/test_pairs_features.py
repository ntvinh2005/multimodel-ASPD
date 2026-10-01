"""Component x pretrained-SAE-feature pairs, with a bundled SAELens catalogue."""

import pytest
import torch

from aspd.analysis.pairs import endpoints as ep
from aspd.analysis.pairs import features as ft
from aspd.analysis.pairs import sae_config as cfg
from aspd.analysis.pairs.spaces import module_spaces
from aspd.analysis.pairs.weights import Directions

GPT2 = "openai-community/gpt2"
GEMMA = "google/gemma-2-2b"


def _dirs(mat: torch.Tensor) -> Directions:
    return Directions(mat=mat, norms=mat.norm(dim=1))


def _sae(site: str, layer: int, *, centred: bool, dim: int = 768, n: int = 100, head_dim=None):
    rel = cfg.Release("r", site, "rel", "id", "label")
    suffix = cfg.SITES[site].space_suffix
    from aspd.analysis.pairs.spaces import site_space

    return ep.sae_endpoint(
        ft.SaeEndpoint(
            key=f"r:L{layer}", model=GPT2, release=rel, layer=layer, hook="h",
            d_sae=n, space=site_space(suffix, layer, dim, head_dim), neuronpedia=None,
            centred=centred,
        ),
        "read",
    )


def _mod(path: str, side, d_in: int, d_out: int, n: int = 100):
    spec = module_spaces(path, d_in, d_out, 64)
    return ep.module_endpoint(path, side, spec, n)


# --- the configuration table ------------------------------------------------------------------


def test_every_configured_release_resolves_against_the_installed_sae_lens():
    """A stale `id_template` must fail loudly here, not serve an empty layer picker."""
    for model, releases in cfg.MODEL_RELEASES.items():
        for rel in releases:
            layers = cfg.release_layers(rel)
            assert layers, f"{model}/{rel.key} resolved no layers"
            assert rel.site in cfg.SITES, f"{rel.key} names an unknown site {rel.site!r}"
            assert cfg.sae_id(rel, layers[0]) in cfg._catalogue()[rel.release].saes_map


def test_every_site_has_exactly_one_default_release_per_model():
    for model in cfg.MODEL_RELEASES:
        seen = [r.site for r in cfg.releases_for_model(model)]
        entries = ft.catalogue(model)
        for site in set(seen):
            defaults = [e for e in entries if e["site"] == site and e["default"]]
            assert len(defaults) == 1, f"{model}/{site} has {len(defaults)} defaults"


def test_gemma_2_2b_it_is_deliberately_unconfigured():
    assert cfg.releases_for_model("google/gemma-2-2b-it") == []
    assert cfg.releases_for_model(GEMMA) != []


def test_an_sae_site_shares_the_space_key_of_the_module_side_it_pairs_with():
    """Equal key is what makes a weight x feature score exact rather than merely computable."""
    down = module_spaces("transformer.h.6.mlp.c_proj", 3072, 768, 64)
    v = module_spaces("transformer.h.6.attn.c_attn.v_proj", 768, 768, 64)
    rel_mlp = cfg.Release("k", "mlp_out", "r", "i", "l")
    rel_z = cfg.Release("k", "attn_z", "r", "i", "l")
    assert cfg.space_key(rel_mlp, 6) == down.write.key
    assert cfg.space_key(rel_z, 6) == v.write.key == "L6.attn.z"


# --- endpoint keys ---------------------------------------------------------------------------


def test_module_paths_are_never_mistaken_for_sae_keys():
    for path in ("transformer.h.6.mlp.c_proj", "model.layers.9.self_attn.o_proj"):
        assert not ft.is_sae_key(path)
    assert ft.is_sae_key("mlp_out_oai32k:L6")
    assert ft.parse_endpoint_key("mlp_out_oai32k:L6") == ("mlp_out_oai32k", 6)
    with pytest.raises(AssertionError):
        ft.parse_endpoint_key("transformer.h.6.mlp.c_proj")


# --- the two corrections ------------------------------------------------------------------------


def test_weight_pairs_are_never_corrected():
    a = _mod("transformer.h.6.mlp.c_fc", "write", 768, 3072)
    b = _mod("transformer.h.6.mlp.c_proj", "read", 3072, 768)
    da, db = _dirs(torch.randn(100, 3072)), _dirs(torch.randn(100, 3072))
    rec = ep.reconcile(a, b, da, db, model_name=GPT2, norms=ft.TargetNorms(GPT2))
    assert rec.a.mat is da.mat and rec.b.mat is db.mat
    assert rec.applied == [] and rec.d_eff_drop == 0


def test_centring_fires_only_for_a_release_that_declares_it():
    a = _mod("transformer.h.6.mlp.c_proj", "write", 3072, 768)
    da, db = _dirs(torch.randn(50, 768)), _dirs(torch.randn(100, 768))
    norms = ft.TargetNorms(GPT2)

    raw = ep.reconcile(a, _sae("mlp_out", 6, centred=False), da, db, model_name=GPT2, norms=norms)
    assert raw.applied == [] and raw.d_eff_drop == 0
    assert torch.equal(raw.a.mat, da.mat)

    cent = ep.reconcile(a, _sae("resid_pre", 6, centred=True), da, db, model_name=GPT2, norms=norms)
    assert cent.d_eff_drop == 1 and len(cent.applied) == 1
    assert torch.allclose(cent.a.mat.mean(-1), torch.zeros(50), atol=1e-6)
    assert torch.allclose(cent.b.mat.mean(-1), torch.zeros(100), atol=1e-6)


def test_centring_one_side_already_fixes_the_dot():
    """Why centring BOTH sides is safe: the second projection cannot change the inner product."""
    u, w = torch.randn(768), torch.randn(768)
    uc = ft.centre(u)
    assert torch.allclose(uc @ w, uc @ ft.centre(w), atol=1e-4)
    # ...but it does change the cosine, which is the reason both sides are centred.
    assert not torch.allclose(
        (uc @ w) / (uc.norm() * w.norm()), (uc @ ft.centre(w)) / (uc.norm() * ft.centre(w).norm())
    )


def test_layernorm_fold_matches_the_hand_computation():
    """<LN(x), V> = (1/sigma) <x, centre(g * V)> + const, so the direction is centre(g * V)."""
    v, g = torch.randn(4, 16), torch.rand(16) + 0.5
    folded = ft.fold_layernorm(v, g, centres=True)
    want = v * g
    assert torch.allclose(folded, want - want.mean(-1, keepdim=True), atol=1e-6)
    # RMSNorm does not centre its input, so nothing is projected out.
    assert torch.allclose(ft.fold_layernorm(v, g, centres=False), want, atol=1e-6)


def test_each_architecture_folds_the_norm_that_actually_precedes_the_read():
    gpt2 = ft.norms_for_model(GPT2)
    assert gpt2["mlp.in"].key == "h.{layer}.ln_2.weight" and gpt2["mlp.in"].site == "resid_mid"
    assert gpt2["attn.q"].key == "h.{layer}.ln_1.weight" and gpt2["attn.q"].site == "resid_pre"
    assert all(s.centres and not s.unit_offset for s in gpt2.values()), "GPT-2 uses LayerNorm"

    gemma = ft.norms_for_model(GEMMA)
    assert gemma["mlp.gate"].key == "model.layers.{layer}.pre_feedforward_layernorm.weight"
    assert gemma["attn.q"].key == "model.layers.{layer}.input_layernorm.weight"
    assert all(not s.centres and s.unit_offset for s in gemma.values()), "Gemma-2 uses (1+w) RMSNorm"


def test_only_a_module_read_side_sits_behind_a_layernorm():
    """An SAE is fit on the stream itself; a write direction is added to it directly."""
    assert ep.reads_through_layernorm(_mod("transformer.h.6.mlp.c_fc", "read", 768, 3072), GPT2)
    assert ep.reads_through_layernorm(_mod("transformer.h.6.mlp.c_proj", "write", 3072, 768), GPT2) is None
    assert ep.reads_through_layernorm(_sae("resid_mid", 6, centred=False), GPT2) is None


# --- link classification -------------------------------------------------------------------------


def test_sae_links_are_positional():
    down = _mod("transformer.h.6.mlp.c_proj", "write", 3072, 768)
    fc = _mod("transformer.h.6.mlp.c_fc", "read", 768, 3072)
    assert ep.sae_link(down, _sae("mlp_out", 6, centred=False), GPT2) == "same_point"
    assert ep.sae_link(down, _sae("resid_post", 6, centred=False), GPT2) == "residual"
    assert ep.sae_link(down, _sae("resid_pre", 9, centred=False), GPT2) == "residual"
    assert ep.sae_link(fc, _sae("resid_mid", 6, centred=False), GPT2) == "layernorm"


def test_the_z_space_pairs_only_within_its_own_layer():
    v = _mod("transformer.h.6.attn.c_attn.v_proj", "write", 768, 768)
    assert ep.sae_link(v, _sae("attn_z", 6, centred=False, head_dim=64), GPT2) == "same_point"
    assert ep.sae_link(v, _sae("attn_z", 8, centred=False, head_dim=64), GPT2) is None


def test_a_template_needs_both_endpoints_to_exist():
    """A run that decomposed only MLPs is offered no attention templates."""
    spaces = {"transformer.h.6.mlp.c_proj": module_spaces("transformer.h.6.mlp.c_proj", 3072, 768, 64)}
    keys = {t["key"] for t in ep.available_sae_templates(spaces, GPT2, ft.catalogue(GPT2))}
    assert "wf_mlp_out" in keys
    assert not any(k.startswith("wf_attn") for k in keys), keys
    # ...and feature x feature templates survive, since both their endpoints are dictionaries.
    assert "ff_mlp_to_resid" in keys


def test_no_templates_at_all_for_a_model_with_no_configured_dictionaries():
    spaces = {"transformer.h.6.mlp.c_proj": module_spaces("transformer.h.6.mlp.c_proj", 3072, 768, 64)}
    assert ep.available_sae_templates(spaces, "google/gemma-2-2b-it", []) == []

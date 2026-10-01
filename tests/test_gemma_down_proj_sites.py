"""Gemma-2's `down_proj` input and output sites, captured by two hooks on one module."""

import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

from aspd.sae.sites import (
    OutputCapture,
    gemma2_mlp_down_proj,
    resolve_sites,
    site_widths,
)

D_MODEL, D_HIDDEN, VOCAB, SEQ, LAYERS = 32, 96, 64, 8, 2
LAYER = 1


@pytest.fixture(scope="module")
def model() -> Gemma2ForCausalLM:
    torch.manual_seed(0)
    cfg = Gemma2Config(
        hidden_size=D_MODEL,
        intermediate_size=D_HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
        attn_implementation="eager",
    )
    return Gemma2ForCausalLM(cfg).eval()


@pytest.fixture
def tokens() -> torch.Tensor:
    return torch.randint(0, VOCAB, (2, SEQ), generator=torch.Generator().manual_seed(1))


def test_the_two_sites_are_one_module_under_two_distinct_keys(model):
    sites = gemma2_mlp_down_proj(LAYER)
    assert sites.input_site != sites.output_site
    assert sites.capture_modules[sites.input_site] == sites.output_site
    assert sites.takes == {sites.input_site: "input", sites.output_site: "output"}
    resolve_sites(model, sites)  # the `.in` KEY must never be resolved as a path


def test_input_key_is_not_a_module_path(model):
    """If it ever became one, `resolve_sites` would pass and the capture would read the wrong tensor."""
    sites = gemma2_mlp_down_proj(LAYER)
    with pytest.raises(AttributeError):
        model.get_submodule(sites.input_site)


def test_captures_the_gated_hidden_state_exactly(model, tokens):
    """`x` is `act(gate(h)) * up(h)` -- the product, not `up_proj`'s output."""
    sites = gemma2_mlp_down_proj(LAYER)
    paths = [sites.input_site, sites.output_site]
    with torch.no_grad(), OutputCapture(
        model, paths, detach=True, takes=sites.takes, modules=sites.capture_modules
    ) as cap:
        model(tokens)
        x, y = cap[sites.input_site], cap[sites.output_site]

    mlp = model.get_submodule(f"model.layers.{LAYER}.mlp")
    with torch.no_grad(), OutputCapture(
        model, [f"model.layers.{LAYER}.mlp.gate_proj", f"model.layers.{LAYER}.mlp.up_proj"],
        detach=True,
    ) as cap:
        model(tokens)
        gate = cap[f"model.layers.{LAYER}.mlp.gate_proj"]
        up = cap[f"model.layers.{LAYER}.mlp.up_proj"]

    torch.testing.assert_close(x, mlp.act_fn(gate) * up)
    assert not torch.allclose(x, up), "capturing up_proj's output would silently drop the gate"
    # And `y` really is this module's output, i.e. the pair brackets the matrix.
    torch.testing.assert_close(y, mlp.down_proj(x))


def test_widths_are_read_off_a_real_forward_and_are_asymmetric(model, tokens):
    """9216/2304 on the real model: the INPUT side is the wide one, inverting every `c_fc` pair."""
    sites = gemma2_mlp_down_proj(LAYER)
    widths = site_widths(model, sites, tokens)
    assert widths[sites.input_site] == D_HIDDEN
    assert widths[sites.output_site] == D_MODEL


def test_the_pair_is_sized_off_the_narrower_side_whichever_side_that_is(model, tokens):
    """`feature_multiplier` must mean the same thing on `down_proj` as on `c_fc`."""
    from aspd.sae.train import build_sae_pair

    multiplier = 4
    saes = build_sae_pair(
        model,
        gemma2_mlp_down_proj(LAYER),
        tokens,
        feature_multiplier=multiplier,
        device="cpu",
        n_batches_to_dead=5,
    )
    widths = {sae.cfg.d_in for sae in saes.values()}
    assert widths == {D_HIDDEN, D_MODEL}, "each dictionary keeps its own site's width"
    # Equal-width pair, sized off the NARROW side (D_MODEL), not the input (D_HIDDEN).
    assert {sae.cfg.n_features for sae in saes.values()} == {multiplier * D_MODEL}


def test_a_multiplier_too_small_to_cover_the_wide_side_is_refused(model, tokens):
    """An undercomplete dictionary cannot represent its own input, and the wide side is the one at
    risk once sizing follows the narrow side.
    """
    from aspd.sae.train import build_sae_pair

    with pytest.raises(AssertionError, match="undercomplete"):
        build_sae_pair(
            model,
            gemma2_mlp_down_proj(LAYER),
            tokens,
            feature_multiplier=2,  # 2 * 32 = 64 < D_HIDDEN = 96
            device="cpu",
            n_batches_to_dead=5,
        )


def test_truncation_still_fires_when_both_hooks_share_a_module(model, tokens):
    """`stop_when_complete` counts CAPTURES, and two of them land on one module's forward."""
    sites = gemma2_mlp_down_proj(LAYER)
    paths = [sites.input_site, sites.output_site]
    with torch.no_grad(), OutputCapture(
        model,
        paths,
        detach=True,
        stop_when_complete=True,
        takes=sites.takes,
        modules=sites.capture_modules,
    ) as cap:
        cap.run(tokens)
    assert set(cap.acts) == set(paths)


def test_modules_mapping_rejects_keys_it_does_not_capture(model):
    sites = gemma2_mlp_down_proj(LAYER)
    with pytest.raises(AssertionError, match="modules names keys"):
        OutputCapture(model, [sites.output_site], modules=sites.capture_modules)


def test_input_module_requires_an_input_take():
    """An output take is addressable by its own path, so a separate hooked module is a mistake."""
    from aspd.sae.sites import SitePair

    with pytest.raises(AssertionError, match="input_module"):
        SitePair(
            module="m.mlp.down_proj",
            input_site="m.mlp.down_proj.in",
            output_site="m.mlp.down_proj",
            input_take="output",
            input_module="m.mlp.down_proj",
        )


def test_splice_replaces_the_signal_the_dictionary_was_fitted_on(model, tokens):
    """`splice_ce_kl` must hook `hook_module` and honour `take`."""
    from aspd.eval.dictionary import DictionaryAdapter
    from aspd.eval.dictionary_report import splice_ce_kl

    sites = gemma2_mlp_down_proj(LAYER)
    seen: dict[str, torch.Tensor] = {}

    class RecordingDictionary(DictionaryAdapter):

        def __init__(self, site_path, hook_module, take):
            self.site_path, self.hook_module, self.take, self.role = (
                site_path, hook_module, take, "in")

        @property
        def n_features(self):
            return 1

        def encode(self, acts):
            seen["width"] = acts.shape[-1]
            return acts

        def decode(self, features):
            return features

        def decoder_rows(self):
            raise NotImplementedError

        def group_boundaries(self):
            return None

    d = RecordingDictionary(sites.input_site, sites.capture_modules[sites.input_site], "input")
    splice_ce_kl(
        model, d, iter([tokens]), lambda b: model(b).logits,
        n_batches=1, pad_id=-1, device="cpu",
    )
    assert seen["width"] == D_HIDDEN, (
        f"spliced a {seen['width']}-wide tensor; the input dictionary is fitted on the "
        f"{D_HIDDEN}-wide gated hidden, so the output take would be the wrong object"
    )

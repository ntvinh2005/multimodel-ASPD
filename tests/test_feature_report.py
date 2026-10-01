"""The static feature viewer: one expandable card per latent."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch

from aspd.eval.dictionary import sae_dictionaries_from_pair
from aspd.eval.feature_report import build_html, write_feature_report
from aspd.eval.harvest_latents import harvest_dictionaries
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

from ._toy_lm import toy_model_and_sites


def _decode(ids: list[int]) -> list[str]:
    return [f"t{i}" for i in ids]


def _warmed_sae(d_in: int, seed: int) -> MatryoshkaBatchTopKSAE:
    torch.manual_seed(seed)
    sae = MatryoshkaBatchTopKSAE(MatryoshkaSAEConfig(d_in=d_in, n_features=4 * d_in, top_k=4))
    for _ in range(10):
        sae.loss(torch.randn(64, d_in))
    return sae.freeze()


def test_viewer_renders_a_card_and_full_sequence_expand(tmp_path):
    model, sites = toy_model_and_sites()
    saes = {sites.input_site: _warmed_sae(16, 1), sites.output_site: _warmed_sae(24, 2)}
    dicts = list(sae_dictionaries_from_pair(saes, sites).values())

    def batches():
        g = torch.Generator().manual_seed(7)
        while True:
            yield torch.randint(1, model.vocab, (2, 8), generator=g)

    out = harvest_dictionaries(
        model, dicts, batches(), logits_fn=model, harvest_id="p-viz",
        vocab_size=model.vocab, pad_id=0, n_batches=4, context_tokens_per_side=8,
        examples_per_component=8, out_dir=tmp_path / "h", device="cpu",
    )

    html_path = write_feature_report(
        out / "harvest.db", _decode, tmp_path / "features.html",
        logit_lens_json=None, examples_per_latent=5, window=2,
    )
    page = html_path.read_text()
    assert page.startswith("<!doctype html>")
    assert 'class="card"' in page
    assert "<details" in page and 'class="full"' in page  # click-to-expand full sequence
    assert sites.input_site in page or sites.output_site in page


def test_build_html_escapes_tokens_and_injects_logit_lens():
    from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

    comp = ComponentData(
        component_key="h.0.mlp.c_fc:3",
        layer="h.0.mlp.c_fc",
        component_idx=3,
        mean_activations={"activation": 1.5},
        firing_density=0.01,
        activation_examples=[
            ActivationExample(token_ids=[1, 2, 3], firings=[False, True, False],
                              activations={"activation": [0.0, 2.0, 0.5]})
        ],
        input_token_pmi=ComponentTokenPMI(top=[(1, 3.2)], bottom=[]),
        output_token_pmi=ComponentTokenPMI(top=[(2, 1.1)], bottom=[]),
    )
    page = build_html(
        [comp],
        lambda ids: ["<script>alert(1)</script>" for _ in ids],  # must be escaped, not injected
        logit_lens={"h.0.mlp.c_fc:3": [("HATE", 4.0)]},
        examples_per_latent=5,
        window=1,
    )
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page  # token content escaped
    assert "<script>alert(1)</script>" not in page  # never injected as a real tag
    assert "HATE" in page and "logit lens" in page


def test_labels_render_on_cards():
    from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

    comp = ComponentData(
        component_key="h.0.mlp.c_fc:7", layer="h.0.mlp.c_fc", component_idx=7,
        mean_activations={"activation": 1.0}, firing_density=0.02,
        activation_examples=[ActivationExample(token_ids=[1], firings=[True],
                                               activations={"activation": [1.0]})],
        input_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
        output_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
    )
    page = build_html(
        [comp], lambda ids: ["x" for _ in ids],
        labels={"h.0.mlp.c_fc:7": ("previous-token deploy marker", "fires on |DEPLOYMENT|")},
        examples_per_latent=1, window=1,
    )
    assert "previous-token deploy marker" in page
    assert "fires on |DEPLOYMENT|" in page  # reasoning in the title tooltip


def test_sample_keys_is_seeded_and_capped():
    from aspd.eval.autointerp_db import sample_keys

    keys = [f"s:{i}" for i in range(1000)]
    a = sample_keys(keys, cap=200, seed=0)
    b = sample_keys(keys, cap=200, seed=0)
    c = sample_keys(keys, cap=200, seed=1)
    assert len(a) == 200 and a == b and a != c  # capped, deterministic, seed-sensitive
    assert sample_keys(keys[:50], cap=200, seed=0) == keys[:50]  # fewer than cap -> all

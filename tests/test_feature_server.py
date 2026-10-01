"""The per-feature server: activation bins, decoding, one feature per query."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch

from aspd.eval.dictionary import sae_dictionaries_from_pair
from aspd.eval.feature_server import build_app, feature_detail
from aspd.eval.harvest_latents import harvest_dictionaries
from aspd.eval.tokens import decode_with_spaces
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

from ._toy_lm import toy_model_and_sites


def test_decode_with_spaces_uses_batch_decode():
    class FakeTok:
        def batch_decode(self, batches):
            return [f" tok{b[0]}" for b in batches]

    assert decode_with_spaces(FakeTok())([3, 7]) == [" tok3", " tok7"]


def test_feature_detail_bins_examples_by_peak_activation():
    from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

    def ex(peak):
        acts = [0.0, peak, 0.1]
        return ActivationExample(token_ids=[1, 2, 3], firings=[False, True, False],
                                 activations={"activation": acts})

    comp = ComponentData(
        component_key="h.0.mlp.c_fc:5", layer="h.0.mlp.c_fc", component_idx=5,
        mean_activations={"activation": 1.0}, firing_density=0.01,
        activation_examples=[ex(4.0), ex(3.5), ex(0.5)],  # two high, one low
        input_token_pmi=ComponentTokenPMI(top=[(1, 2.0)], bottom=[]),
        output_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
    )
    d = feature_detail(
        comp, lambda ids: [f"t{i}" for i in ids],
        n_intervals=5, examples_per_interval=5, window=1,
    )
    assert d["max_act"] == 4.0 and d["feature"] == "h.0.mlp.c_fc:5"
    # top interval (0.8*max .. max) holds the two high-peak examples, sorted descending.
    top = d["intervals"][0]
    assert top["examples"][0]["peak"] == 4.0 and top["examples"][1]["peak"] == 3.5
    # a lower interval holds the 0.5 example; every example carries a short + full view.
    peaks = [e["peak"] for iv in d["intervals"] for e in iv["examples"]]
    assert sorted(peaks, reverse=True) == [4.0, 3.5, 0.5]
    assert set(top["examples"][0].keys()) == {"peak", "short", "full"}
    assert d["input_pmi"] == [("t1", 2.0)]


def test_feature_detail_caps_examples_per_interval():
    from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

    def ex(peak):
        return ActivationExample(token_ids=[1, 2], firings=[False, True],
                                 activations={"activation": [0.0, peak]})

    # 8 examples all in the top interval; cap must trim to 3, highest first.
    comp = ComponentData(
        component_key="s:0", layer="s", component_idx=0,
        mean_activations={"activation": 1.0}, firing_density=0.1,
        activation_examples=[ex(1.0 - 0.01 * j) for j in range(8)],
        input_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
        output_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
    )
    d = feature_detail(comp, lambda ids: ["x"] * len(ids),
                       n_intervals=5, examples_per_interval=3, window=1)
    top = d["intervals"][0]
    assert len(top["examples"]) == 3
    assert top["examples"][0]["peak"] >= top["examples"][1]["peak"] >= top["examples"][2]["peak"]


def _warmed(d_in, seed):
    torch.manual_seed(seed)
    sae = MatryoshkaBatchTopKSAE(MatryoshkaSAEConfig(d_in=d_in, n_features=4 * d_in, top_k=4))
    for _ in range(10):
        sae.loss(torch.randn(64, d_in))
    return sae.freeze()


def test_server_lists_sites_and_serves_one_feature(tmp_path):
    from fastapi.testclient import TestClient
    from param_decomp_lab.harvest.db import HarvestDB

    model, sites = toy_model_and_sites()
    saes = {sites.input_site: _warmed(16, 1), sites.output_site: _warmed(24, 2)}
    dicts = list(sae_dictionaries_from_pair(saes, sites).values())

    def batches():
        g = torch.Generator().manual_seed(7)
        while True:
            yield torch.randint(1, model.vocab, (2, 8), generator=g)

    out = harvest_dictionaries(
        model, dicts, batches(), logits_fn=model, harvest_id="p-server00",
        vocab_size=model.vocab, pad_id=0, n_batches=4, context_tokens_per_side=8,
        examples_per_component=8, out_dir=tmp_path / "h", device="cpu",
    )

    app = build_app(out / "harvest.db", lambda ids: [f"t{i}" for i in ids], n_intervals=3, window=2)
    client = TestClient(app)

    site_rows = client.get("/api/sites").json()
    assert {r["site"] for r in site_rows} == {sites.input_site, sites.output_site}

    # Pick a real fired feature and fetch just it.
    live_key = HarvestDB(out / "harvest.db", readonly=True).get_component_keys()[0]
    site, idx = live_key.rsplit(":", 1)
    r = client.get(f"/api/feature/{site}/{idx}")
    assert r.status_code == 200 and r.json()["feature"] == live_key
    assert client.get("/").status_code == 200  # the viewer page renders

    # Browse list: per-site, density-ordered, capped.
    rows = client.get(f"/api/browse/{site}").json()
    assert rows and all(set(r) == {"idx", "label", "density"} for r in rows)
    dens = [r["density"] for r in rows]
    assert dens == sorted(dens, reverse=True)  # no labels here -> pure density order

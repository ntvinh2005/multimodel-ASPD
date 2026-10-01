"""The dictionary adapter adds no drift over the raw SAE and refuses an unfrozen one."""

import pytest
import torch

from aspd.eval.dictionary import SAEDictionary, sae_dictionaries_from_pair
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig

from ._toy_lm import toy_model_and_sites


def _warmed_sae(d_in: int, seed: int = 0) -> MatryoshkaBatchTopKSAE:
    torch.manual_seed(seed)
    sae = MatryoshkaBatchTopKSAE(
        MatryoshkaSAEConfig(d_in=d_in, n_features=8 * d_in, top_k=4, n_batches_to_dead=5)
    )
    for _ in range(20):
        sae.loss(torch.randn(64, d_in))
    return sae.freeze()


def test_adapter_roundtrips_the_raw_sae_exactly():
    sae = _warmed_sae(12)
    d = SAEDictionary(sae, "some.site", "out")
    x = torch.randn(30, 12)
    assert torch.equal(d.encode(x), sae.features(x))
    assert torch.equal(d.decode(d.encode(x)), sae.decode(sae.features(x)))
    assert torch.equal(d.decoder_rows(), sae.W_dec)
    assert d.group_boundaries() == sae.group_indices
    assert d.n_features == sae.cfg.n_features


def test_adapter_refuses_an_unfrozen_dictionary():
    torch.manual_seed(0)
    sae = MatryoshkaBatchTopKSAE(MatryoshkaSAEConfig(d_in=8, n_features=64))  # not frozen
    with pytest.raises(AssertionError, match="FROZEN"):
        SAEDictionary(sae, "s", "out")


def test_pair_builder_addresses_by_role():
    _, sites = toy_model_and_sites()
    saes = {sites.input_site: _warmed_sae(16, 1), sites.output_site: _warmed_sae(24, 2)}
    dicts = sae_dictionaries_from_pair(saes, sites)
    assert dicts["in"].site_path == sites.input_site and dicts["in"].role == "in"
    assert dicts["out"].site_path == sites.output_site and dicts["out"].role == "out"

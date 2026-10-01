"""Target-layer sharding of dataset attribution and the rescaling in its merge."""

from pathlib import Path

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from param_decomp_lab.dataset_attributions.storage import DatasetAttributionStorage as Store

from aspd.analysis.circuits.sharding import install_union_merge

_STOCK_MERGE = Store.merge

C, V, DM = 4, 6, 3


def _store(targets: list[str], *, value: float, n_tokens: int, tok_hist: list[int]) -> Store:
    return Store(
        regular_attr={t: {"src.a": torch.full((C, C), value)} for t in targets},
        regular_attr_abs={t: {"src.a": torch.full((C, C), value)} for t in targets},
        embed_attr={t: torch.full((C, V), value) for t in targets},
        embed_attr_abs={t: torch.full((C, V), value) for t in targets},
        unembed_attr={"src.a": torch.full((DM, C), value)},
        embed_unembed_attr=torch.full((DM, V), value),
        w_unembed=torch.zeros(V, DM),
        ci_sum={"src.a": torch.full((C,), 10.0)},
        component_act_sq_sum={"src.a": torch.full((C,), 4.0)},
        logit_sq_sum=torch.full((V,), 2.0),
        embed_token_count=torch.tensor(tok_hist),
        ci_threshold=0.0,
        n_tokens_processed=n_tokens,
    )


def _roundtrip(store: Store, path: Path) -> Path:
    store.save(path)
    return path


def test_stock_merge_cannot_union_target_shards(tmp_path):
    a = _roundtrip(_store(["t.a"], value=1.0, n_tokens=100, tok_hist=[1] * V), tmp_path / "a.pt")
    b = _roundtrip(_store(["t.b"], value=1.0, n_tokens=100, tok_hist=[1] * V), tmp_path / "b.pt")
    with pytest.raises(KeyError):
        _STOCK_MERGE([a, b])


def test_union_merge_keeps_both_shards_targets(tmp_path):
    install_union_merge()
    a = _roundtrip(_store(["t.a"], value=1.0, n_tokens=100, tok_hist=[1] * V), tmp_path / "a.pt")
    b = _roundtrip(_store(["t.b"], value=2.0, n_tokens=100, tok_hist=[1] * V), tmp_path / "b.pt")
    merged = Store.merge([a, b])
    assert merged.target_layers == {"t.a", "t.b"}
    assert torch.allclose(merged._regular_attr["t.a"]["src.a"], torch.full((C, C), 1.0))
    assert torch.allclose(merged._regular_attr["t.b"]["src.a"], torch.full((C, C), 2.0))


def test_target_shards_do_not_multiply_the_denominators(tmp_path):
    """THE trap. Every target shard sees the WHOLE corpus, so `ci_sum` / `logit_sq_sum` /
    `n_tokens_processed` are the same measurement repeated, not a partition. Summing them would
    scale every query-time denominator by the shard count and divide every normalised attribution
    by it -- uniformly, so nothing looks wrong and every number is off by 72x.
    """
    install_union_merge()
    hist = [3] * V
    a = _roundtrip(_store(["t.a"], value=1.0, n_tokens=500, tok_hist=hist), tmp_path / "a.pt")
    b = _roundtrip(_store(["t.b"], value=1.0, n_tokens=500, tok_hist=hist), tmp_path / "b.pt")
    merged = Store.merge([a, b])
    assert merged.n_tokens_processed == 500, "target shards must NOT sum the token count"
    assert torch.allclose(merged._ci_sum["src.a"], torch.full((C,), 10.0))
    assert torch.allclose(merged._logit_sq_sum, torch.full((V,), 2.0))


def test_batch_shards_still_sum_the_denominators(tmp_path):
    install_union_merge()
    a = _roundtrip(_store(["t.a"], value=1.0, n_tokens=500, tok_hist=[3] * V), tmp_path / "a.pt")
    b = _roundtrip(_store(["t.a"], value=1.0, n_tokens=400, tok_hist=[2] * V), tmp_path / "b.pt")
    merged = Store.merge([a, b])
    assert merged.n_tokens_processed == 900, "batch shards partition the corpus and must sum"
    assert torch.allclose(merged._ci_sum["src.a"], torch.full((C,), 20.0))
    # And the shared target's rows add, as they always did.
    assert torch.allclose(merged._regular_attr["t.a"]["src.a"], torch.full((C, C), 2.0))

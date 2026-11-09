from __future__ import annotations

import pytest
import torch

from aspd.multimodel.v3.dictionary_concentration import (
    band_count_rows,
    count_summary,
    effective_size,
    feature_band,
    gini_coefficient,
)
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.objective import auxk_dead_feature_loss
from aspd.multimodel.v3.reconstruction import ActivationDecoder


def test_concentration_statistics_include_zero_count_features() -> None:
    counts = torch.zeros(8192, dtype=torch.long)
    counts[:4] = 2

    assert effective_size(counts) == pytest.approx(4.0)
    assert gini_coefficient(counts) == pytest.approx(1 - 4 / 8192)
    summary = count_summary(counts)
    assert summary["active_features"] == 4
    assert summary["top_1_percent_share"] == 1.0

    bands = band_count_rows(counts)
    assert [row["band"] for row in bands] == ["B1", "B2", "B3", "B4", "B5"]
    assert bands[0]["effective_size_within_band"] == pytest.approx(4.0)
    assert bands[0]["firing_density_vs_uniform"] == pytest.approx(16.0)
    assert feature_band(8191).name == "B5"


def _auxk_fixture(raw_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.ones(raw_logits.shape[:2], dtype=torch.bool)
    target = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    full = torch.zeros_like(target)
    decoder = ActivationDecoder(4, 2, dtype=torch.float32)
    with torch.no_grad():
        decoder.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        decoder.bias.zero_()
    scores = torch.relu(raw_logits)
    loss = auxk_dead_feature_loss(
        scores,
        decoder,
        target,
        full,
        valid,
        torch.ones(4, dtype=torch.bool),
        top_k_aux=1,
        sparse_backend="dense",
    )
    return loss, scores


def test_current_auxk_cannot_revive_all_negative_relu_logits() -> None:
    raw_logits = torch.full((2, 1, 4), -1.0, requires_grad=True)
    loss, scores = _auxk_fixture(raw_logits)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert float(scores.detach().sum()) == 0.0
    torch.testing.assert_close(raw_logits.grad, torch.zeros_like(raw_logits))


def test_current_auxk_gradient_reaches_positive_dead_score_only() -> None:
    raw_logits = torch.tensor(
        [[[0.1, -1.0, -1.0, -1.0]], [[-1.0, 0.1, -1.0, -1.0]]],
        requires_grad=True,
    )
    loss, _scores = _auxk_fixture(raw_logits)
    loss.backward()

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert raw_logits.grad is not None
    assert float(raw_logits.grad[0, 0, 0]) < 0.0
    assert float(raw_logits.grad[1, 0, 1]) < 0.0
    assert int(torch.count_nonzero(raw_logits.grad)) == 2
    before = raw_logits.detach().clone()
    after = before - raw_logits.grad
    assert float(after[0, 0, 0]) > float(before[0, 0, 0])
    assert float(after[1, 0, 1]) > float(before[1, 0, 1])


def test_positive_dead_score_can_reenter_main_batch_topk_after_auxk_updates() -> None:
    raw_logits = torch.tensor(
        [[[0.5, 0.1, -1.0, -1.0]], [[0.5, -1.0, -1.0, -1.0]]],
        requires_grad=True,
    )
    valid = torch.ones((2, 1), dtype=torch.bool)
    target = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    full = torch.zeros_like(target)
    decoder = ActivationDecoder(4, 2, dtype=torch.float32)
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.weight[:, 1] = torch.tensor([1.0, 0.0])
        decoder.bias.zero_()
    dead = torch.tensor([False, True, False, False])

    for _ in range(6):
        scores = torch.relu(raw_logits)
        loss = auxk_dead_feature_loss(
            scores,
            decoder,
            target,
            full,
            valid,
            dead,
            top_k_aux=1,
            sparse_backend="dense",
        )
        gradient = torch.autograd.grad(loss, raw_logits)[0]
        with torch.no_grad():
            raw_logits -= 0.1 * gradient

    gate = V3BatchTopK(4, 1, sync_statistics=False)
    code = gate(torch.relu(raw_logits), valid, update_statistics=False)
    assert float(raw_logits.detach()[0, 0, 1]) > 0.5
    assert bool(code.gate[0, 0, 1])
    assert float(raw_logits.detach()[1, 0, 1]) == -1.0

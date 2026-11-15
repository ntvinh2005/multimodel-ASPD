from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aspd.multimodel.v3.functional_correspondence import (
    FunctionalCorrespondenceContract,
    continuation_logprob_metrics,
    extracted_context_boundary,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/multimodel/aspd_v3/main500m_functional_correspondence_v1.yaml"


class ChunkTokenizer:
    chunks = {1: " hello", 2: " world", 3: " tail"}

    def decode(self, token_ids, **_kwargs):
        return "".join(self.chunks[int(value)] for value in token_ids)


def test_functional_correspondence_contract_is_three_model_read_only() -> None:
    contract = FunctionalCorrespondenceContract.load(CONTRACT)
    assert contract.training_authorized_by_this_contract is False
    assert contract.model_order == ("gemma", "smol", "granite")
    assert contract.components_per_site == 12
    assert sum(contract.contribution_stratum_quota.values()) == 12
    assert "logprob_per_utf8_byte" in contract.primary_metric


def test_extracted_context_boundary_tracks_natural_peak_after_trim() -> None:
    boundary = extracted_context_boundary(
        token_ids=[1, 2, 3],
        activations=[0.1, 0.5, 0.2],
        context="hello world tail",
        tokenizer=ChunkTokenizer(),
    )
    assert "hello world tail"[:boundary] == "hello world"


def test_signed_continuation_metric_is_edited_minus_clean() -> None:
    clean = torch.zeros(1, 3, 4)
    edited = clean.clone()
    clean[0, 1, 2] = 4.0
    edited[0, 1, 2] = -4.0
    rows = continuation_logprob_metrics(
        clean,
        edited,
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[1, 1, 1]]),
        [2],
        [5],
    )
    assert len(rows) == 1
    assert rows[0]["continuation_model_tokens"] == 1
    assert rows[0]["edited_minus_clean_continuation_logprob_total"] < 0
    assert rows[0]["edited_minus_clean_continuation_logprob_per_utf8_byte"] == pytest.approx(
        rows[0]["edited_minus_clean_continuation_logprob_total"] / 5
    )


def test_signed_continuation_metric_rejects_empty_suffix() -> None:
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="continuation boundary"):
        continuation_logprob_metrics(
            logits,
            logits,
            torch.tensor([[0, 1]]),
            torch.tensor([[1, 1]]),
            [2],
            [1],
        )

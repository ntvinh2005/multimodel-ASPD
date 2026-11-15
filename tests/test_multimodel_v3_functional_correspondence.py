from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aspd.multimodel.cli.analyze_aspd_v3_functional_correspondence import (
    PRIMARY,
    _context_alignment,
)
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


def test_context_alignment_preserves_reused_context_ids_and_component_bootstrap() -> None:
    rows = []
    model_offsets = {"gemma": 0.0, "smol": 0.3, "granite": -0.2}
    for site_index, site in enumerate(("pre", "mid")):
        for feature in range(12):
            for activity_index, activity in enumerate(("active", "inactive")):
                for context_index in range(4):
                    for model, model_offset in model_offsets.items():
                        rows.append(
                            {
                                "model": model,
                                "site": site,
                                "feature": feature,
                                "context_activity": activity,
                                # Intentionally reused across every component and activity.
                                "context_id": f"context-{context_index}",
                                "intervention_kind": "learned",
                                "strength": 1.0,
                                PRIMARY: (
                                    site_index * 10.0
                                    + feature
                                    + activity_index * 0.1
                                    + context_index * 0.01
                                    + model_offset
                                ),
                            }
                        )
    contract = SimpleNamespace(
        components_per_site=12,
        active_contexts_per_component=4,
        inactive_contexts_per_component=4,
        bootstrap_seed=17,
        bootstrap_replicates=20,
    )
    result = _context_alignment(rows, contract)
    for activity in ("active", "inactive"):
        for comparison in result[activity].values():
            assert comparison["n_components"] == 24
            assert comparison["n_contexts"] == 96
            assert comparison["contexts_per_component"] == 4
            assert comparison["bootstrap_unit"] == "component"

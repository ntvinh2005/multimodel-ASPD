from __future__ import annotations

import pytest
import torch

from aspd.multimodel.cli.select_delta_behavioral_v3_candidates import (
    _response_activations,
    select_candidate_rows,
)


def test_dev_response_slice_uses_valid_token_mask_and_excludes_prompt() -> None:
    activations = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(1, 5, 2, 3)
    shard = {
        "valid_tokens": torch.tensor([[True, True, True, False, False]]),
        "input_ids": torch.tensor([[11, 12, 13, 0, 0]]),
        "activations": activations,
        "example_indices": torch.tensor([7]),
    }
    record = {
        "token_ids": [11, 12, 13],
        "valid_token_range": [0, 3],
        "prompt_token_count": 1,
    }
    response = _response_activations(shard, record, record_index=7, row_index=0)
    torch.testing.assert_close(response, activations[0, 1:3])
    with pytest.raises(ValueError, match="different example"):
        _response_activations(shard, record, record_index=8, row_index=0)
    shard["valid_tokens"] = torch.tensor([[True, False, True, False, False]])
    with pytest.raises(ValueError, match="valid-token mask"):
        _response_activations(shard, record, record_index=7, row_index=0)


def _rows() -> tuple[list[dict], dict[int, dict]]:
    rows = []
    support = {}
    for feature_id in range(80):
        rows.append({"feature_id": feature_id, "partition": "delta", "relative_decoder_norm_base": feature_id / 100, "relative_decoder_norm_target_derived": 1 - feature_id / 100})
        support[feature_id] = {"response_support": 10, "domain_support": 4, "domains": {"a", "b", "c", "d"}}
    for feature_id in range(80, 120):
        rows.append({"feature_id": feature_id, "partition": "shared", "relative_decoder_norm_base": 0.5, "relative_decoder_norm_target_derived": 0.5})
        support[feature_id] = {"response_support": 10, "domain_support": 4, "domains": {"a", "b", "c", "d"}}
    return rows, support


def test_v3_candidate_groups_are_exact_and_disjoint() -> None:
    rows, support = _rows()
    selected = select_candidate_rows(rows, support)
    assert len(selected) == 100
    groups = {group: [row for row in selected if row["candidate_group"] == group] for group in ("target_tail", "base_tail", "balanced_delta", "shared_control")}
    assert {name: len(values) for name, values in groups.items()} == {"target_tail": 40, "base_tail": 20, "balanced_delta": 20, "shared_control": 20}
    assert len({row["feature_id"] for row in selected}) == 100
    assert all(row["dev_response_support"] == 10 for row in selected)


def test_v3_candidate_pool_failure_is_explicit() -> None:
    rows, support = _rows()
    for feature_id in range(75):
        support[feature_id]["response_support"] = 9
    with pytest.raises(ValueError, match="CANDIDATE_POOL_FAILURE"):
        select_candidate_rows(rows, support)

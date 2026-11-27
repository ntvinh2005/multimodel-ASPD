from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aspd.multimodel.cli.export_natural_token_pilot import (
    aligned_window,
    anchor_measurement,
    recover_native_prefix,
)
from aspd.multimodel.cli.replay_natural_token_components import model_inputs


def test_recover_prefix_uses_bos_and_exact_overlaps():
    windows = [tuple(range(10, 24)), tuple(range(12)), tuple(range(4, 18))]
    assert recover_native_prefix(windows, 0, 24) == tuple(range(24))
    assert recover_native_prefix(windows, 99, 24) == ()
    assert recover_native_prefix([*windows, (0, 99)], 0, 24) == ()


def test_window_requires_its_exact_center_and_ignores_unarchived_positions():
    def point(i, key):
        return SimpleNamespace(anchor_id=key, positions={"gemma": SimpleNamespace(token_index=i)})

    neighbors = SimpleNamespace(
        anchors=[point(1, "archive"), point(2, "center"), point(3, "foreign")]
    )
    assert list(
        aligned_window((1, 2, 3), "center", neighbors, (0, 1, 2, 3), {"archive", "center"})
    ) == [0, 1]
    assert (
        aligned_window((1, 2, 3), "missing", neighbors, (0, 1, 2, 3), {"archive", "center"}) == {}
    )


def test_sparse_support_reads_measured_positive_values_and_rejects_duplicate():
    matrix = {
        "indptr": np.array([0, 2, 3]),
        "indices": np.array([7, 5, 7]),
        "values": np.array([1.75, 2.0, 2.5]),
    }
    assert anchor_measurement(1, {"pre": matrix, "mid": matrix}) == {"PRE:7": 2.5, "MID:7": 2.5}
    with pytest.raises(ValueError, match="Duplicate gate"):
        anchor_measurement(0, {"pre": {**matrix, "indices": np.array([7, 7, 7])}})


def test_replay_uses_original_causal_prefix_and_native_position_per_model():
    native = {
        "documents": {
            "one": {"inputIds": {"gemma": [2, 10, 20, 30]}},
            "two": {"inputIds": {"gemma": [2, 42, 99]}},
        }
    }
    rows = [
        ("one", {"positions": {"gemma": 2}}),
        ("two", {"positions": {"gemma": 1}}),
        ("one", {"positions": {"gemma": 1}}),
    ]
    result = model_inputs(native, rows, "gemma", 0, torch.device("cpu"))
    assert result.input_ids.tolist() == [[2, 10, 20], [2, 42, 0]]
    assert result.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert result.anchor_token.tolist() == [2, 1, 1]
    assert result.anchor_document.tolist() == [0, 1, 0]

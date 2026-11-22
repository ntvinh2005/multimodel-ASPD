"""CPU-only contract checks for the v3.1 checkpoint backend.

The backend cannot run without a GPU and the 500M checkpoint.  These tests catch API
drift: every v3 symbol and keyword the backend relies on must exist with the expected
name, and the frozen 500M configuration must load and agree with the post-hoc ids.
They do not validate the backend scientifically.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from aspd.multimodel.posthoc.ids import MATRIX_NAMES, MATRIX_SITE

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multimodel/aspd_v3/main500m_preparation_v1.yaml"


def _params(function) -> set[str]:
    return set(inspect.signature(function).parameters)


def test_v3_api_used_by_backend_exists() -> None:
    from aspd.multimodel.v3.data import HashSplitPolicy
    from aspd.multimodel.v3.diagnostic_stream import DiagnosticAlignedAnchorStream
    from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
    from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
    from aspd.multimodel.v3.production_stream import load_pinned_pile_dataset
    from aspd.multimodel.v3.reconstruction import _safe_key
    from aspd.multimodel.v3.training_capture import capture_family_pool, padded_model_inputs

    assert {"modulus", "validation_buckets", "test_buckets", "seed"} <= set(
        HashSplitPolicy.__dataclass_fields__)
    assert {"selected_split", "dedup_path", "model_order", "sequence_length"} <= _params(
        DiagnosticAlignedAnchorStream.__init__)
    assert {"native_biases", "device", "dtype", "sparse_backend"} <= _params(
        ModelOwnedMain2B.__init__)
    assert {"path", "model", "config_fingerprint", "config_file_sha256",
            "dataset_manifest_sha256", "checkpoint_rank", "expected_checkpoint_world_size",
            "expected_model_key"} <= _params(ModelOwnedCheckpoint.load_model_only_owner)
    assert {"pad_token_id", "device", "early_exit"} <= _params(capture_family_pool)
    assert {"pad_token_id", "device"} <= _params(padded_model_inputs)
    assert callable(load_pinned_pile_dataset) and callable(_safe_key)


def test_frozen_500m_config_matches_posthoc_ids() -> None:
    from aspd.multimodel.v3.config import V3_LOGICAL_MATRICES
    from aspd.multimodel.v3.main500m_config import load_main500m_config
    from aspd.multimodel.v3.registry import build_model_registry_2b

    try:
        config = load_main500m_config(CONFIG)
    except PermissionError as error:
        # The config is marked full_training_authorized, so model validation re-checks
        # the training-launch evidence (GPU preflight + exact-resume reports).  Those
        # files are readable only by the training owner; see
        # docs/multimodel/posthoc_analysis.md, "Real-run blockers".
        pytest.skip(f"training-authorization evidence is not readable: {error.filename}")
    pending = [m.key for m in config.models if not getattr(m, "layer_prefixes", None)]
    assert not pending, f"frozen 500M config has unresolved models: {pending}"
    registry = build_model_registry_2b(config)
    assert tuple(config.architecture.model_order) == ("gemma", "smol", "granite")
    assert set(V3_LOGICAL_MATRICES) == set(MATRIX_NAMES)
    for layout in registry.values():
        for target in layout.targets():
            expected = "residual_pre" if MATRIX_SITE[target.logical_name] == "pre" else "residual_mid"
            assert target.grounding_kind == expected, target.key
    assert {"dataset_id", "dataset_revision", "split_modulus", "validation_buckets",
            "test_buckets", "split_seed", "sequence_length"} <= set(type(config.data).model_fields)

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
CHECKPOINT = Path(
    "/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/"
    "aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43/"
    "checkpoints/step-000061036"
)


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
    from aspd.multimodel.posthoc.v31_backend import load_frozen_config
    from aspd.multimodel.v3.config import V3_LOGICAL_MATRICES
    from aspd.multimodel.v3.registry import build_model_registry_2b

    if not (CHECKPOINT / "manifest.json").is_file():
        pytest.skip(f"checkpoint manifest not available: {CHECKPOINT}")
    config = load_frozen_config(CONFIG, CHECKPOINT, skip_training_launch_evidence=True)
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


def test_evaluation_loader_rejects_a_config_that_is_not_the_checkpoints(tmp_path: Path) -> None:
    from aspd.multimodel.posthoc.v31_backend import load_frozen_config

    if not (CHECKPOINT / "manifest.json").is_file():
        pytest.skip(f"checkpoint manifest not available: {CHECKPOINT}")
    altered = tmp_path / "altered.yaml"
    altered.write_text(CONFIG.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the checkpoint's training config"):
        load_frozen_config(altered, CHECKPOINT, skip_training_launch_evidence=True)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.fidelity import (
    GlobalTensorMetrics,
    IdentityResidual,
    resolved_config_sha256,
    validate_checkpoint_provenance,
    validate_matrix_alignment,
    write_json_immutable,
)
from aspd.multimodel.tiny_overfit import select_fixed_batches

ROOT = Path(__file__).parents[1]
PILOT_CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml"


def test_global_metrics_use_dataset_sufficient_statistics_and_valid_mask() -> None:
    accumulator = GlobalTensorMetrics(1)
    # Batch means differ, so averaging batch FVUs would not equal the global value.
    accumulator.update(
        torch.tensor([[[0.0], [100.0]]]),
        torch.tensor([[[0.0], [90.0]]]),
        torch.tensor([[True, False]]),
    )
    accumulator.update(
        torch.tensor([[[2.0], [4.0]]]),
        torch.tensor([[[1.0], [1.0]]]),
        torch.tensor([[True, True]]),
    )
    result = accumulator.result()
    target = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    prediction = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    expected_sse = float((target - prediction).square().sum())
    expected_variance = float((target - target.mean()).square().sum())
    assert result["rows"] == 3
    assert result["sse"] == pytest.approx(expected_sse)
    assert result["fvu"] == pytest.approx(expected_sse / expected_variance)


def test_zero_delta_and_small_denominators_are_explicitly_unstable() -> None:
    accumulator = GlobalTensorMetrics(2)
    zero = torch.zeros(1, 2, 2)
    accumulator.update(zero, zero, torch.ones(1, 2, dtype=torch.bool))
    result = accumulator.result()
    assert result["normalized_squared_error"] is None
    assert result["fvu"] is None
    assert result["cosine_similarity"] is None
    assert result["energy_well_conditioned"] is False
    assert result["variance_well_conditioned"] is False


def test_exact_total_parameter_and_input_change_identities() -> None:
    w_base = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    w_target = torch.tensor([[1.5, 2.0], [-0.5, 0.25]])
    x_base = torch.tensor([[[2.0, -1.0], [3.0, 4.0]]])
    x_target = torch.tensor([[[1.0, 5.0], [-2.0, 4.0]]])
    total = torch.nn.functional.linear(x_target, w_target) - torch.nn.functional.linear(
        x_base, w_base
    )
    parameter = torch.nn.functional.linear(x_base, w_target - w_base)
    input_change = torch.nn.functional.linear(x_target - x_base, w_target)
    mask = torch.ones(1, 2, dtype=torch.bool)
    residual = IdentityResidual()
    residual.update(total, parameter, input_change, mask)
    result = residual.result()
    assert result["max_abs_residual"] < 1e-6
    assert result["relative_rms_residual"] < 1e-6

    # The same identity must hold for any pair of learned linear operators.
    p_base = torch.tensor([[0.2, -0.1], [0.5, 0.3]])
    p_target = torch.tensor([[0.4, -0.1], [0.2, 0.8]])
    learned_total = torch.nn.functional.linear(x_target, p_target) - torch.nn.functional.linear(
        x_base, p_base
    )
    learned_parameter = torch.nn.functional.linear(x_base, p_target - p_base)
    learned_input = torch.nn.functional.linear(x_target - x_base, p_target)
    learned_residual = IdentityResidual()
    learned_residual.update(learned_total, learned_parameter, learned_input, mask)
    assert learned_residual.result()["max_abs_residual"] < 1e-6


def test_model_order_and_matrix_alignment_fail_closed() -> None:
    cfg = load_experiment_config(PILOT_CONFIG)
    matrices = {
        matrix.name: {"d_in": 2, "d_out": 3} for matrix in cfg.models[0].matrices
    }
    cache = SimpleNamespace(model_manifests=[{"matrices": matrices}, {"matrices": matrices}])
    assert validate_matrix_alignment(cfg, cache) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    reversed_cfg = cfg.model_copy(deep=True)
    reversed_cfg.models.reverse()
    with pytest.raises(ValueError, match="ordered exactly"):
        validate_matrix_alignment(reversed_cfg, cache)
    bad_cache = SimpleNamespace(
        model_manifests=[{"matrices": matrices}, {"matrices": {**matrices, "q_proj": {"d_in": 4, "d_out": 3}}}]
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        validate_matrix_alignment(cfg, bad_cache)


def test_checkpoint_config_and_cache_mismatches_are_rejected(tmp_path: Path) -> None:
    cfg = load_experiment_config(PILOT_CONFIG)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    provenance = {
        "cache_fingerprint": "cache-a",
        "config_sha256": resolved_config_sha256(cfg),
        "model_ids": [model.model_id for model in cfg.models],
        "model_revisions": [model.revision for model in cfg.models],
        "latent_count_C": cfg.sparsity.n_features,
        "model_count_N": len(cfg.models),
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    cache = SimpleNamespace(token_manifest={"cache_fingerprint": "cache-a"})
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record = validate_checkpoint_provenance(
        cfg,
        cache,
        checkpoint,
        expected_cache_fingerprint="cache-a",
        expected_checkpoint_sha256=expected_sha,
    )
    assert record["sha256"] == expected_sha
    with pytest.raises(ValueError, match="cache fingerprint mismatch"):
        validate_checkpoint_provenance(
            cfg, cache, checkpoint, expected_cache_fingerprint="cache-b"
        )
    provenance["config_sha256"] = "wrong"
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint_provenance(cfg, cache, checkpoint)


def test_immutable_json_refuses_old_run_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    write_json_immutable(output, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json_immutable(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}


def test_tiny_overfit_selects_fixed_unshuffled_batches_once() -> None:
    cfg = load_experiment_config(PILOT_CONFIG)

    class Cache:
        calls: list[tuple[str, int, bool]] = []

        def iter_batches(self, split, batch_size, shuffle):
            self.calls.append((split, batch_size, shuffle))
            yield {"batch": 0}
            yield {"batch": 1}
            raise AssertionError("tiny selection consumed more than requested")

    cache = Cache()
    selected = select_fixed_batches(cache, cfg, 2)  # type: ignore[arg-type]
    assert selected == [{"batch": 0}, {"batch": 1}]
    assert cache.calls == [("train", 2, False)]

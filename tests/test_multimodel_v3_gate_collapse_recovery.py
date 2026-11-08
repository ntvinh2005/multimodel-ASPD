from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import _load_contract
from aspd.multimodel.cli.train_aspd_v3_2b import _load_training_config
from aspd.multimodel.v3.encoder import FeatureReadout
from aspd.multimodel.v3.gate_collapse_recovery_config import (
    load_gate_collapse_recovery_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "multimodel" / "aspd_v3"
CONTROL = CONFIG_DIR / "two_model_gate_collapse_control_v1.yaml"
TREATMENT = CONFIG_DIR / "two_model_gate_collapse_zscore_v1.yaml"
EVALUATION = CONFIG_DIR / "two_model_gate_collapse_recovery_evaluation_v1.yaml"


def test_gate_collapse_configs_are_isolated_and_matched() -> None:
    control = load_gate_collapse_recovery_config(CONTROL)
    treatment = load_gate_collapse_recovery_config(TREATMENT)

    assert control.arm_id == "full_only_control"
    assert treatment.arm_id == "per_anchor_zscore"
    assert control.objective.gate_logit_normalization == "none"
    assert treatment.objective.gate_logit_normalization == "per_anchor_feature_zscore"
    assert control.models == treatment.models
    assert control.architecture == treatment.architecture
    assert control.optimization == treatment.optimization
    assert control.data == treatment.data
    assert all("qwen" not in model.model_id.lower() for model in treatment.models)
    assert control.runtime.checkpoint_every_steps == 1221
    assert treatment.runtime.keep_complete_checkpoints == 2
    assert _load_training_config(CONTROL).fingerprint() == control.fingerprint()
    assert _load_training_config(TREATMENT).fingerprint() == treatment.fingerprint()


def test_gate_collapse_config_rejects_arm_normalization_mismatch(tmp_path: Path) -> None:
    payload = yaml.safe_load(TREATMENT.read_text(encoding="utf-8"))
    payload["objective"]["gate_logit_normalization"] = "none"
    path = tmp_path / "mismatch.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="arm and gate-logit normalization differ"):
        load_gate_collapse_recovery_config(path)


def test_gate_collapse_evaluation_contract_is_exact() -> None:
    contract = _load_contract(EVALUATION)
    assert contract["arms"] == ["full_only_control", "per_anchor_zscore"]
    assert contract["model_order"] == ["gemma", "smol"]
    assert contract["accepted_anchors"] == 100_352


def test_per_anchor_zscore_preserves_rank_and_spreads_gradient() -> None:
    readout = FeatureReadout(
        4,
        4,
        "mean",
        logit_normalization="per_anchor_feature_zscore",
        dtype=torch.float32,
    )
    with torch.no_grad():
        readout.output.weight.copy_(torch.eye(4))
        readout.output.bias.copy_(torch.tensor([-0.3, 0.2, 0.7, -0.1]))
    tokens = torch.tensor(
        [[[[0.1, 0.5, -0.2, 0.8]], [[-0.4, 0.3, 0.9, 0.2]]]],
        dtype=torch.float32,
    )
    valid = torch.tensor([[True, False]])
    scores, logits = readout.forward_with_logits(tokens, valid)
    projection = readout.latest_projection_logits
    assert projection is not None
    projection.retain_grad()

    torch.testing.assert_close(
        logits[0, 0].argsort(), projection[0, 0].argsort(), rtol=0, atol=0
    )
    torch.testing.assert_close(logits[0, 0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert 0.99 < float(logits[0, 0].std(unbiased=False).detach()) < 1.01
    torch.testing.assert_close(scores[0, 1], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(logits[0, 1], torch.zeros(4), rtol=0, atol=0)

    scores[0, 0, 2].backward()
    assert projection.grad is not None
    assert torch.count_nonzero(projection.grad[0, 0]).item() == 4


def test_logit_normalization_does_not_change_trainable_initial_state() -> None:
    torch.manual_seed(7)
    control = FeatureReadout(8, 16, "mean", logit_normalization="none")
    torch.manual_seed(7)
    treatment = FeatureReadout(
        8, 16, "mean", logit_normalization="per_anchor_feature_zscore"
    )
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, treatment.state_dict()[name], rtol=0, atol=0)

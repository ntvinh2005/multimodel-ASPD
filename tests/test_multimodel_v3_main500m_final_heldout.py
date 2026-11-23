from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import (
    _arm_label,
    _load_contract,
)
from aspd.multimodel.v3.main500m_config import load_main500m_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multimodel/aspd_v3/main500m_preparation_v1.yaml"
CONTRACT = ROOT / "configs/multimodel/aspd_v3/main500m_final_heldout_v1.yaml"


def test_final_heldout_contract_is_read_only_pinned_and_qwen_free() -> None:
    contract = _load_contract(CONTRACT)
    config = load_main500m_config(CONFIG)
    assert contract["checkpoint_steps"] == 61_036
    assert contract["accepted_anchors"] == 100_352
    assert contract["accepted_training_exposures"] == 500_006_912
    assert contract["training_authorized_by_this_contract"] is False
    assert contract["model_order"] == ["gemma", "smol", "granite"]
    assert all("qwen" not in model.model_id.lower() for model in config.models)
    assert _arm_label(config) == "main500m_final"


def test_final_heldout_contract_fails_closed_on_budget_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    payload["accepted_anchors"] = 2_048
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract changed"):
        _load_contract(changed)

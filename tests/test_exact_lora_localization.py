from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.exact_lora_localization import (
    PROJECTIONS,
    REFINEMENT_SHARD_ARMS,
    SHARD_ARMS,
    AdapterSite,
    load_adapter_inventory,
    replace_base_with_target,
    replace_target_with_base,
    select_localization_prompts,
    sites_for_arm,
)

CONFIG = "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml"
PROMPTS = "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl"


def _sites() -> list[AdapterSite]:
    return [
        AdapterSite(
            layer=layer,
            projection=projection,
            module=(
                f"model.layers.{layer}.self_attn.{projection}"
                if projection in {"q_proj", "k_proj", "v_proj", "o_proj"}
                else f"model.layers.{layer}.mlp.{projection}"
            ),
            lora_a_key=f"layer{layer}.{projection}.A",
            lora_b_key=f"layer{layer}.{projection}.B",
            rank=32,
        )
        for layer in range(28)
        for projection in PROJECTIONS
    ]


def test_prompt_selection_is_balanced_and_frozen() -> None:
    rows = select_localization_prompts(PROMPTS)
    assert len(rows) == 30
    assert sum(row["category"] == "risk" for row in rows) == 12
    assert sum(row["category"] == "sports_control" for row in rows) == 12
    assert sum(row["category"] == "neutral" for row in rows) == 6
    assert {row["risk_level"] for row in rows if row["category"] == "risk"} == {
        "mild", "moderate", "high"
    }
    assert len({row["sport_domain"] for row in rows if row["category"] == "risk"}) == 12


def test_coarse_arms_partition_all_real_adapter_sites() -> None:
    sites = _sites()
    assert len(sites_for_arm(sites, "full_adapter_removed")) == 196
    assert len(sites_for_arm(sites, "attention_removed")) == 112
    assert len(sites_for_arm(sites, "mlp_removed")) == 84
    for arm in (
        "layers_0_6_removed", "layers_7_13_removed",
        "layers_14_20_removed", "layers_21_27_removed",
    ):
        assert len(sites_for_arm(sites, arm)) == 49
    assert {arm for values in SHARD_ARMS.values() for arm in values} == {
        "clean_base", "clean_target", "full_adapter_removed", "attention_removed",
        "mlp_removed", "layers_0_6_removed", "layers_7_13_removed",
        "layers_14_20_removed", "layers_21_27_removed",
    }


def test_refinement_arms_are_bounded_and_select_expected_sites() -> None:
    sites = _sites()
    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert len(sites_for_arm(sites, f"{projection}_removed")) == 28
    for arm in (
        "mlp_layers_0_6_removed", "mlp_layers_7_13_removed",
        "mlp_layers_14_20_removed", "mlp_layers_21_27_removed",
    ):
        assert len(sites_for_arm(sites, arm)) == 21
    assert len(sites_for_arm(sites, "mlp_layers_7_20_removed")) == 42
    assert len(sites_for_arm(sites, "mlp_updates_inserted_into_base")) == 84
    assert len({arm for values in REFINEMENT_SHARD_ARMS.values() for arm in values}) == 14


class _ToyPair(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.q_proj = nn.Linear(2, 2, bias=False)
        nn.init.constant_(self.model.layers[0].self_attn.q_proj.weight, value)


def test_exact_replacement_restores_after_exception() -> None:
    base, target = _ToyPair(1.0), _ToyPair(3.0)
    site = AdapterSite(
        layer=0, projection="q_proj", module="model.layers.0.self_attn.q_proj",
        lora_a_key="a", lora_b_key="b", rank=1,
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        with replace_target_with_base(target, base, [site]) as audit:
            assert audit["replacement_exact"] is True
            torch.testing.assert_close(
                target.model.layers[0].self_attn.q_proj.weight,
                base.model.layers[0].self_attn.q_proj.weight,
            )
            raise RuntimeError("deliberate")
    assert torch.equal(
        target.model.layers[0].self_attn.q_proj.weight,
        torch.full((2, 2), 3.0),
    )
    assert audit["restoration_exact"] is True


def test_exact_target_insertion_into_base_restores_both_models() -> None:
    base, target = _ToyPair(1.0), _ToyPair(3.0)
    site = AdapterSite(
        layer=0, projection="q_proj", module="model.layers.0.self_attn.q_proj",
        lora_a_key="a", lora_b_key="b", rank=1,
    )
    target_before = target.model.layers[0].self_attn.q_proj.weight.detach().clone()
    with replace_base_with_target(base, target, [site]) as audit:
        assert audit["replacement_exact"] is True
        assert torch.equal(base.model.layers[0].self_attn.q_proj.weight, target_before)
    assert audit["restoration_exact"] is True
    assert torch.equal(base.model.layers[0].self_attn.q_proj.weight, torch.ones(2, 2))
    assert torch.equal(target.model.layers[0].self_attn.q_proj.weight, target_before)


def test_real_adapter_inventory_schema_from_pinned_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_experiment_config(CONFIG)
    adapter_cfg = {
        "base_model_name_or_path": "unsloth/Qwen2.5-7B-Instruct",
        "target_modules": list(PROJECTIONS),
        "r": 32,
        "lora_alpha": 64,
        "use_rslora": True,
        "lora_dropout": 0.0,
        "bias": "none",
    }
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps(adapter_cfg), encoding="utf-8")
    tensors = {}
    for layer in range(28):
        for projection in PROJECTIONS:
            family = "self_attn" if projection in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{family}.{projection}"
            tensors[f"{prefix}.lora_A.weight"] = torch.zeros(32, 2)
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros(3, 32)
    weights_path = tmp_path / "adapter_model.safetensors"
    save_file(tensors, weights_path)

    def fake_download(_model: str, filename: str, **_kwargs: object) -> str:
        return str(config_path if filename == "adapter_config.json" else weights_path)

    monkeypatch.setattr(
        "aspd.multimodel.exact_lora_localization.hf_hub_download", fake_download
    )
    metadata, sites = load_adapter_inventory(cfg)
    assert metadata["site_count"] == 196
    assert metadata["tensor_count"] == 392
    assert metadata["scaling"] == pytest.approx(64 / (32**0.5))
    assert {site.layer for site in sites} == set(range(28))
    assert {site.projection for site in sites} == set(PROJECTIONS)

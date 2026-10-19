from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aspd.multimodel.cache import load_effective_model, snapshot_model_weights
from aspd.multimodel.cli.validate import validate_static_config
from aspd.multimodel.config import ModelSpec, load_experiment_config

ROOT = Path(__file__).parents[1]
BASE_REVISION = "a75c9dc945567a9b6f568b8503a0307731607bee"
TARGET_REVISION = "6aa7e935c74c519c66555a111d1e6982059bd82d"
MATRIX_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MATRIX_PATHS = [
    "model.layers.14.self_attn.q_proj",
    "model.layers.14.self_attn.k_proj",
    "model.layers.14.self_attn.v_proj",
    "model.layers.14.self_attn.o_proj",
    "model.layers.14.mlp.gate_proj",
    "model.layers.14.mlp.up_proj",
    "model.layers.14.mlp.down_proj",
]


def _spec(*, adapter: bool) -> ModelSpec:
    return ModelSpec(
        name="target" if adapter else "base",
        model_id="adapter/model" if adapter else "base/model",
        revision="adapter-rev" if adapter else "base-rev",
        adapter_base_revision="base-rev" if adapter else None,
        dtype="float32",
        grounding={"module": "model.layers.14", "capture": "input", "tensor_index": 0},
        matrices=[{"name": name, "module": path} for name, path in zip(MATRIX_NAMES, MATRIX_PATHS, strict=True)],
    )


class _TinyLM(nn.Module):
    def __init__(self, q_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.mlp = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, nn.Linear(2, 2, bias=False))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, name, nn.Linear(2, 2, bias=False))
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(14)] + [layer])
        self.config = SimpleNamespace(eos_token_id=2, vocab_size=10)
        if q_weight is not None:
            self.model.layers[14].self_attn.q_proj.weight.data.copy_(q_weight)


def test_qwen25_aspd_pilot_config_provenance_and_static_validation() -> None:
    cfg = load_experiment_config(ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml")
    assert [model.name for model in cfg.models] == ["base", "target"]
    base, target = cfg.models
    assert (base.model_id, base.revision) == ("unsloth/Qwen2.5-7B-Instruct", BASE_REVISION)
    assert (target.model_id, target.revision) == (
        "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
        TARGET_REVISION,
    )
    assert target.adapter_base_revision == base.revision
    assert base.grounding.model_dump() == {"module": "model.layers.14", "capture": "input", "tensor_index": 0}
    assert [matrix.name for matrix in base.matrices] == MATRIX_NAMES
    assert [matrix.module for matrix in base.matrices] == MATRIX_PATHS
    assert base.matrices == target.matrices
    assert (cfg.data.tokenizer, cfg.data.tokenizer_revision) == (
        "unsloth/Qwen2.5-7B-Instruct",
        BASE_REVISION,
    )
    assert cfg.cache.root.endswith("qwen25_extreme_sports_aspd_pilot")
    assert cfg.training.output_dir.endswith("qwen25_extreme_sports_aspd_d0_s1_pilot")
    assert validate_static_config(cfg)["model_loading"].startswith("adapter-aware")


def test_full_checkpoint_loader_does_not_invoke_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    spec = _spec(adapter=False)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kwargs: calls.append(("hf", model_id)) or _TinyLM(),
    )
    import peft

    monkeypatch.setattr(peft.PeftConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PEFT path used")))
    model, provenance = load_effective_model(spec, "cpu")
    assert isinstance(model, nn.Module)
    assert calls == [("hf", "base/model")]
    assert provenance["model_kind"] == "full_checkpoint"
    assert provenance["adapter_merged"] is False


def test_peft_loader_order_merges_before_return(monkeypatch: pytest.MonkeyPatch) -> None:
    import peft
    import transformers

    events: list[str] = []
    base_weight = torch.ones(2, 2)
    delta = torch.full((2, 2), 3.0)
    base = _TinyLM(base_weight)

    def adapter_config(model_id: str, revision: str) -> SimpleNamespace:
        events.append(f"config:{model_id}@{revision}")
        return SimpleNamespace(base_model_name_or_path="declared/base")

    class _Wrapped:
        def merge_and_unload(self) -> nn.Module:
            events.append("merge")
            base.model.layers[14].self_attn.q_proj.weight.data.add_(delta)
            return base

    monkeypatch.setattr(peft.PeftConfig, "from_pretrained", adapter_config)
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda model, model_id, revision, is_trainable: events.append(f"attach:{model_id}@{revision}") or _Wrapped(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kwargs: events.append(f"base:{model_id}@{kwargs['revision']}") or base,
    )
    model, provenance = load_effective_model(_spec(adapter=True), "cpu")
    assert events == ["config:adapter/model@adapter-rev", "base:declared/base@base-rev", "attach:adapter/model@adapter-rev", "merge"]
    assert provenance["adapter_merged"] is True
    assert torch.equal(model.model.layers[14].self_attn.q_proj.weight, base_weight + delta)


def test_snapshot_uses_effective_weight_and_preserves_module_paths() -> None:
    spec = _spec(adapter=True)
    model = _TinyLM(torch.full((2, 2), 4.0))
    weights, shapes = snapshot_model_weights(
        model,
        spec,
        {"model_kind": "peft_merged", "adapter_merged": True},
    )
    assert torch.equal(weights["q_proj"], torch.full((2, 2), 4.0))
    assert set(weights) == set(MATRIX_NAMES)
    assert set(shapes) == set(MATRIX_NAMES)
    with pytest.raises(RuntimeError, match="before merge"):
        snapshot_model_weights(model, spec, {"model_kind": "peft_wrapper", "adapter_merged": False})

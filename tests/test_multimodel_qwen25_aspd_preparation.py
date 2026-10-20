from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aspd.multimodel.cache import load_effective_model, snapshot_model_weights
from aspd.multimodel.cli.validate import validate_static_config
from aspd.multimodel.config import ModelSpec, load_experiment_config
from aspd.multimodel.diagnostics.peft_merge_equivalence import (
    audit_effective_delta,
    compare_forward_pair,
    tensor_metrics,
)

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
    assert cfg.cache.minimum_model_schema_version == 2
    assert cfg.training.output_dir.endswith("qwen25_extreme_sports_aspd_d0_s1_pilot")
    assert validate_static_config(cfg)["model_loading"].startswith("adapter-aware")
    assert cfg.data.chat_template_tokenizer is None


def test_qwen25_aspd_general1m_config_preserves_method_and_scales_data() -> None:
    pilot = load_experiment_config(
        ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml"
    )
    cfg = load_experiment_config(
        ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml"
    )
    assert cfg.models == pilot.models
    assert cfg.encoder == pilot.encoder
    assert cfg.sparsity == pilot.sparsity
    assert cfg.objective == pilot.objective
    assert cfg.data.sequence_length == 256
    assert cfg.data.train_tokens == 1_048_576
    assert cfg.data.validation_tokens == 131_072
    assert [source.weight for source in cfg.data.sources] == [1.0, 1.0]
    assert cfg.training.max_steps == 4096
    assert cfg.training.batch_size_sequences == 2
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.validate_every == 1024
    assert cfg.training.validation_batches == 256
    assert cfg.training.save_every == 2048
    assert cfg.training.keep_last_checkpoints == 2
    assert cfg.cache.root.endswith("qwen25_extreme_sports_aspd_general1m")
    assert cfg.training.output_dir.endswith("qwen25_extreme_sports_aspd_general1m")


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


class _DiagnosticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.mlp = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(1, 1, bias=False))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, nn.Linear(1, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            x = getattr(self.self_attn, name)(x)
        for name in ("gate_proj", "up_proj", "down_proj"):
            x = getattr(self.mlp, name)(x)
        return x


class _DiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(14)] + [_DiagnosticLayer()])
        self.seen_ids: list[list[list[int]]] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False):
        del attention_mask, use_cache
        self.seen_ids.append(input_ids.detach().cpu().tolist())
        hidden = input_ids.float().unsqueeze(-1)
        hidden = self.model.layers[14](hidden)
        return SimpleNamespace(logits=hidden)


class _PeftLikeDiagnosticWrapper(nn.Module):
    """Minimal PEFT-shaped wrapper whose canonical paths live on its base model."""

    def __init__(self, base: _DiagnosticModel) -> None:
        super().__init__()
        self.wrapped_base = base

    def get_base_model(self) -> nn.Module:
        return self.wrapped_base

    def forward(self, *args, **kwargs):
        return self.wrapped_base(*args, **kwargs)


def test_diagnostic_replays_identical_ids_and_reports_fp32_metrics() -> None:
    spec = _spec(adapter=True)
    left, right = _DiagnosticModel(), _DiagnosticModel()
    right.load_state_dict(left.state_dict())
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids)
    result = compare_forward_pair(left, right, spec, ids, mask)
    assert result["logits"]["max_abs_diff"] == pytest.approx(0.0)
    assert result["R"]["cosine_similarity"] == pytest.approx(1.0)
    assert left.seen_ids == right.seen_ids == [[[1, 2, 3]]]
    assert tensor_metrics(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))["max_abs_diff"] == pytest.approx(2.0)


def test_diagnostic_resolves_canonical_paths_through_peft_wrapper() -> None:
    spec = _spec(adapter=True)
    base, merged = _DiagnosticModel(), _DiagnosticModel()
    merged.load_state_dict(base.state_dict())
    unmerged = _PeftLikeDiagnosticWrapper(base)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    result = compare_forward_pair(unmerged, merged, spec, ids, torch.ones_like(ids))
    assert set(result) == {"R", *(f"X/{name}" for name in MATRIX_NAMES), "logits"}
    assert all(metrics["max_abs_diff"] == pytest.approx(0.0) for metrics in result.values())


def test_effective_delta_audit_reports_norms_and_targeted_warning() -> None:
    spec = _spec(adapter=True)
    base = _TinyLM(torch.ones(2, 2))
    target = _TinyLM(torch.full((2, 2), 3.0))
    audit = audit_effective_delta(base, target, spec, SimpleNamespace(target_modules=["q_proj"]))
    q = next(row for row in audit["matrices"] if row["name"] == "q_proj")
    assert q["delta_frobenius_norm"] == pytest.approx(4.0)
    assert q["relative_delta_norm"] == pytest.approx(2.0)
    assert q["exactly_equal"] is False
    assert q["engineering_warning"] is False

    unchanged = audit_effective_delta(base, base, spec, SimpleNamespace(target_modules=["q_proj"]))
    q_unchanged = next(row for row in unchanged["matrices"] if row["name"] == "q_proj")
    assert q_unchanged["engineering_warning"] is True

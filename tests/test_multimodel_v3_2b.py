from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.v3.data import SQLiteExposureLedger
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B, load_main_2b_v1_config
from aspd.multimodel.v3.model_owned import model_owned_parameter_counts
from aspd.multimodel.v3.production_stream import ModelPoolInputs
from aspd.multimodel.v3.reconstruction import ActivationDecoder
from aspd.multimodel.v3.registry import ModelFamilyLayout, build_model_registry_2b
from aspd.multimodel.v3.sparse_support import prepare_sparse_support
from aspd.multimodel.v3.training_capture import capture_family_pool

ROOT = Path(__file__).parents[1]


def _config():
    return load_main_2b_v1_config(
        ROOT / "configs/multimodel/aspd_v3/main_2b_v1.yaml"
    )


def test_main_2b_contract_and_registry_are_incompatible_and_complete() -> None:
    config = _config()
    registry = build_model_registry_2b(config)
    assert tuple(registry) == MODEL_ORDER_2B
    assert config.full_training_authorized is False
    assert config.data.minimum_optimizer_steps == 61_036
    assert config.data.minimum_accepted_anchor_exposures == 500_006_912
    assert sum(len(layout.targets()) for layout in registry.values()) == 168
    assert registry["qwen"].layers == tuple(range(10, 18))
    assert registry["gemma"].layers == tuple(range(9, 17))
    assert registry["smol"].layers == tuple(range(8, 16))
    assert registry["gemma"].residual_mid_suffix == "pre_feedforward_layernorm"
    assert registry["smol"].kv_out == 2048
    counts = model_owned_parameter_counts(config, registry)
    assert counts["rank_one_total"] == 8_908_701_696
    assert counts["activation_decoder_total"] == 838_963_200
    assert {value["model_id"] for value in config.payload()["models"]} == {
        "Qwen/Qwen3-1.7B",
        "google/gemma-2-2b-it",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    }


def test_prepared_rank_one_support_matches_reference_forward_and_gradients() -> None:
    torch.manual_seed(7)
    reference = RankOneComponents(11, 5, 4, unit_norm=True)
    optimized = RankOneComponents(11, 5, 4, unit_norm=True)
    optimized.load_state_dict(reference.state_dict())
    x_reference = torch.randn(2, 3, 5, requires_grad=True)
    x_optimized = x_reference.detach().clone().requires_grad_(True)
    code = torch.zeros(2, 3, 11)
    code.scatter_(-1, torch.tensor([[[0, 3], [1, 4], [2, 8]]] * 2), 1.0)
    valid = torch.tensor([[True, True, False], [True, False, True]])
    support = prepare_sparse_support(code, valid, binary=True)
    expected = reference(x_reference, code * valid[..., None], valid)
    actual = optimized.forward_sparse(
        x_optimized,
        support.position,
        support.feature,
        support.coefficient,
        compute_dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x_optimized.grad, x_reference.grad)
    torch.testing.assert_close(optimized.U.grad, reference.U.grad)
    torch.testing.assert_close(optimized.V.grad, reference.V.grad)


def test_prepared_activation_support_matches_matryoshka_decoder() -> None:
    torch.manual_seed(9)
    reference = ActivationDecoder(16, 7)
    optimized = ActivationDecoder(16, 7)
    optimized.load_state_dict(reference.state_dict())
    code_reference = torch.zeros(1, 4, 16, requires_grad=True)
    with torch.no_grad():
        code_reference.scatter_(-1, torch.tensor([[[0, 7], [1, 8], [2, 9], [3, 10]]]), 0.5)
    code_optimized = code_reference.detach().clone().requires_grad_(True)
    valid = torch.tensor([[True, True, False, True]])
    boundaries = (0, 4, 8, 12, 16)
    support = prepare_sparse_support(code_optimized, valid, binary=False)
    expected = reference(code_reference * valid[..., None], boundaries=boundaries)
    actual = optimized.forward_sparse(support, boundaries=boundaries)
    assert isinstance(expected, tuple) and isinstance(actual, tuple)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    torch.testing.assert_close(optimized.weight.grad, reference.weight.grad)
    torch.testing.assert_close(optimized.bias.grad, reference.bias.grad)


def test_exposure_ledger_records_smol_as_immutable_third_tokenizer_slot(tmp_path: Path) -> None:
    ledger = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite",
        dataset_manifest_sha256="a" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
    )
    ledger.observe_raw_tokens("row-0", {"qwen": 11, "gemma": 12, "smol": 13})
    assert ledger.raw_token_counts() == {"qwen": 11, "gemma": 12, "smol": 13}
    ledger.close()
    reopened = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite",
        dataset_manifest_sha256="a" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
    )
    assert reopened.raw_token_counts()["smol"] == 13
    reopened.close()


class _ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(value) + self.k_proj(value) + self.v_proj(value))


class _ToyMlp(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(value) * self.up_proj(value))


class _ToyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = _ToyAttention(width)
        self.mlp = _ToyMlp(width)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = value + self.self_attn(self.input_layernorm(value))
        return value + self.mlp(self.post_attention_layernorm(value))


class _ToyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyLayer(4) for _ in range(10)])
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_ids, attention_mask, **_kwargs):
        value = self.embed(input_ids)
        for layer in self.model.layers:
            value = layer(value)
        return SimpleNamespace(last_hidden_state=value)


def _toy_layout() -> ModelFamilyLayout:
    return ModelFamilyLayout(
        key="toy",
        family="llama",
        model_id="toy",
        revision="0" * 40,
        tokenizer_id="toy",
        tokenizer_revision="0" * 40,
        layers=tuple(range(8)),
        layer_prefixes=("model.layers",),
        residual_pre_suffix="input_layernorm",
        residual_mid_suffix="post_attention_layernorm",
        hidden_size=4,
        intermediate_size=4,
        num_hidden_layers=10,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        q_out=4,
        kv_out=4,
        fused_qkv=False,
        fused_gate_up=False,
    )


def test_early_exit_capture_matches_full_capture_and_skips_later_layers() -> None:
    torch.manual_seed(2)
    full_model = _ToyCausal()
    early_model = _ToyCausal()
    early_model.load_state_dict(full_model.state_dict())
    pool = ModelPoolInputs(
        input_ids=((1, 2, 3),),
        attention_masks=((1, 1, 1),),
        anchor_document_indices=(0, 0),
        anchor_token_indices=(0, 2),
    )
    layout = _toy_layout()
    full = capture_family_pool(
        full_model, layout, pool, pad_token_id=0, device=torch.device("cpu")
    )
    early = capture_family_pool(
        early_model,
        layout,
        pool,
        pad_token_id=0,
        device=torch.device("cpu"),
        early_exit=True,
    )
    for full_values, early_values in (
        (full.residual_pre, early.residual_pre),
        (full.residual_mid, early.residual_mid),
    ):
        for left, right in zip(full_values, early_values, strict=True):
            torch.testing.assert_close(left, right)
    for key in full.matrix_inputs:
        torch.testing.assert_close(full.matrix_inputs[key], early.matrix_inputs[key])
        torch.testing.assert_close(full.matrix_outputs[key], early.matrix_outputs[key])
    assert [layer.calls for layer in full_model.model.layers[-2:]] == [1, 1]
    assert [layer.calls for layer in early_model.model.layers[-2:]] == [0, 0]

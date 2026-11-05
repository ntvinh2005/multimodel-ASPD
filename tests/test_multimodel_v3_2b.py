from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch
from torch import nn

from aspd.multimodel.cli.train_aspd_v3_2b import _PoolPrefetcher
from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.v3.data import SQLiteExposureLedger
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B, load_main_2b_v1_config
from aspd.multimodel.v3.model_owned import (
    LocalCaptureBatch,
    ModelOwnedMain2B,
    build_model_owned_adam,
    model_owned_parameter_counts,
)
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.objective import auxk_dead_feature_loss
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, ModelPoolInputs
from aspd.multimodel.v3.reconstruction import ActivationDecoder
from aspd.multimodel.v3.registry import ModelFamilyLayout, build_model_registry_2b
from aspd.multimodel.v3.sparse_support import prepare_sparse_support
from aspd.multimodel.v3.training_capture import capture_family_pool

ROOT = Path(__file__).parents[1]


class _FakePool:
    def __init__(self, value: int):
        self.value = value

    def to_payload(self) -> dict[str, int]:
        return {"value": self.value}


class _FakeStream:
    def __init__(self, cursor: int = 0):
        self.cursor = cursor

    def next_pool(self, *, pool_anchors: int) -> _FakePool:
        assert pool_anchors == 2048
        value = self.cursor
        self.cursor += 1
        return _FakePool(value)


def _config():
    return load_main_2b_v1_config(
        ROOT / "configs/multimodel/aspd_v3/main_2b_v1.yaml"
    )


def test_prefetch_checkpoint_preserves_exact_pool_order() -> None:
    stream = _FakeStream()
    prefetcher = _PoolPrefetcher(stream, pool_anchors=2048)
    prefetcher.start()
    assert prefetcher.next_payload() == {"value": 0}
    buffered = prefetcher.checkpoint_payload()
    assert buffered == {"value": 1}
    assert stream.cursor == 2
    prefetcher.close()

    resumed_stream = _FakeStream(cursor=2)
    resumed = _PoolPrefetcher(
        resumed_stream, pool_anchors=2048, buffered_payload=buffered
    )
    resumed.start()
    assert resumed.next_payload() == {"value": 1}
    assert resumed.next_payload() == {"value": 2}
    resumed.close()

    final_stream = _FakeStream()
    final_prefetcher = _PoolPrefetcher(final_stream, pool_anchors=2048)
    final_prefetcher.start()
    assert final_prefetcher.next_payload(schedule_next=False) == {"value": 0}
    assert final_stream.cursor == 1
    assert final_prefetcher.checkpoint_payload() is None
    final_prefetcher.close()


def test_aligned_pool_json_checkpoint_preserves_model_order() -> None:
    model = ModelPoolInputs(
        input_ids=((1,),),
        attention_masks=((1,),),
        anchor_document_indices=(0,),
        anchor_token_indices=(0,),
    )
    pool = AlignedAnchorPool(
        anchor_ids=("anchor",),
        document_ids=("document",),
        models={"qwen": model, "gemma": model, "smol": model},
        source_observations=({},),
    )
    canonical_payload = json.loads(json.dumps(pool.to_payload(), sort_keys=True))
    restored = AlignedAnchorPool.from_payload(canonical_payload)

    assert tuple(restored.models) == ("qwen", "gemma", "smol")
    restored.validate(
        expected_anchors=1,
        sequence_length=1,
        model_order=("qwen", "gemma", "smol"),
    )


def test_aligned_pool_restores_pre_explicit_order_checkpoint() -> None:
    model_payload = {
        "input_ids": [[1]],
        "attention_masks": [[1]],
        "anchor_document_indices": [0],
        "anchor_token_indices": [0],
    }
    legacy_payload = {
        "anchor_ids": ["anchor"],
        "document_ids": ["document"],
        "models": {"gemma": model_payload, "qwen": model_payload, "smol": model_payload},
        "source_observations": [{}],
    }

    restored = AlignedAnchorPool.from_payload(legacy_payload)

    assert tuple(restored.models) == ("qwen", "gemma", "smol")


def test_main_2b_contract_and_registry_are_incompatible_and_complete() -> None:
    config = _config()
    registry = build_model_registry_2b(config)
    assert tuple(registry) == MODEL_ORDER_2B
    assert config.full_training_authorized is True
    assert config.data.minimum_optimizer_steps == 61_036
    assert config.data.minimum_accepted_anchor_exposures == 500_006_912
    assert config.runtime.sparse_backend == "dense"
    assert config.runtime.ledger_anchor_storage == "exact_compact_v1"
    assert config.runtime.preflight_evidence.endswith(
        "/preflight/final_authorized_v1/preflight_report.json"
    )
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


def test_batch_topk_places_all_state_on_requested_device() -> None:
    gate = V3BatchTopK(16, 2, pool_anchors=4, device="cpu")
    assert {value.device.type for value in gate.buffers()} == {"cpu"}


def test_prepared_rank_one_support_matches_reference_forward_and_gradients() -> None:
    torch.manual_seed(7)
    reference = RankOneComponents(11, 5, 4, unit_norm=True)
    optimized = RankOneComponents(11, 5, 4, unit_norm=True)
    csr = RankOneComponents(11, 5, 4, unit_norm=True)
    dense = RankOneComponents(11, 5, 4, unit_norm=True)
    optimized.load_state_dict(reference.state_dict())
    csr.load_state_dict(reference.state_dict())
    dense.load_state_dict(reference.state_dict())
    x_reference = torch.randn(2, 3, 5, requires_grad=True)
    x_optimized = x_reference.detach().clone().requires_grad_(True)
    x_csr = x_reference.detach().clone().requires_grad_(True)
    x_dense = x_reference.detach().clone().requires_grad_(True)
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
    csr_actual = csr.forward_sparse_csr(
        x_csr,
        support.crow_indices,
        support.feature,
        support.coefficient,
        compute_dtype=torch.float32,
    )
    dense_actual = dense.forward_dense_masked(
        x_dense,
        (code > 0).to(code.dtype),
        valid,
        compute_dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(csr_actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(dense_actual, expected, rtol=1e-6, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    csr_actual.square().sum().backward()
    dense_actual.square().sum().backward()
    torch.testing.assert_close(x_optimized.grad, x_reference.grad)
    torch.testing.assert_close(optimized.U.grad, reference.U.grad)
    torch.testing.assert_close(optimized.V.grad, reference.V.grad)
    torch.testing.assert_close(x_csr.grad, x_reference.grad)
    torch.testing.assert_close(csr.U.grad, reference.U.grad)
    torch.testing.assert_close(csr.V.grad, reference.V.grad)
    torch.testing.assert_close(x_dense.grad, x_reference.grad)
    torch.testing.assert_close(dense.U.grad, reference.U.grad)
    torch.testing.assert_close(dense.V.grad, reference.V.grad)


def test_indexed_rank_one_support_respects_bfloat16_autocast_compute_dtype() -> None:
    torch.manual_seed(8)
    component = RankOneComponents(11, 5, 4, unit_norm=True)
    x = torch.randn(1, 3, 5, dtype=torch.bfloat16, requires_grad=True)
    code = torch.zeros(1, 3, 11, dtype=torch.bfloat16)
    code.scatter_(-1, torch.tensor([[[0, 3], [1, 4], [2, 8]]]), 1.0)
    valid = torch.ones(1, 3, dtype=torch.bool)
    support = prepare_sparse_support(code, valid, binary=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = component.forward_sparse(
            x,
            support.position,
            support.feature,
            support.coefficient,
            compute_dtype=torch.bfloat16,
        )
        output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert component.U.grad is not None and component.U.grad.dtype == torch.float32
    assert component.V.grad is not None and component.V.grad.dtype == torch.float32


def test_dense_rank_one_accepts_frozen_inference_capture() -> None:
    torch.manual_seed(18)
    component = RankOneComponents(11, 5, 4, unit_norm=True)
    with torch.inference_mode():
        captured = torch.randn(1, 3, 5)
    code = torch.zeros(1, 3, 11)
    code.scatter_(-1, torch.tensor([[[0, 3], [1, 4], [2, 8]]]), 1.0)
    output = component.forward_dense_masked(
        captured,
        code,
        torch.ones(1, 3, dtype=torch.bool),
        compute_dtype=torch.float32,
    )
    output.square().mean().backward()
    assert component.U.grad is not None
    assert component.V.grad is not None


def test_prepared_activation_support_matches_matryoshka_decoder() -> None:
    torch.manual_seed(9)
    reference = ActivationDecoder(16, 7)
    optimized = ActivationDecoder(16, 7)
    csr = ActivationDecoder(16, 7)
    dense = ActivationDecoder(16, 7)
    optimized.load_state_dict(reference.state_dict())
    csr.load_state_dict(reference.state_dict())
    dense.load_state_dict(reference.state_dict())
    code_reference = torch.zeros(1, 4, 16, requires_grad=True)
    with torch.no_grad():
        code_reference.scatter_(-1, torch.tensor([[[0, 7], [1, 8], [2, 9], [3, 10]]]), 0.5)
    code_optimized = code_reference.detach().clone().requires_grad_(True)
    code_csr = code_reference.detach().clone().requires_grad_(True)
    code_dense = code_reference.detach().clone().requires_grad_(True)
    valid = torch.tensor([[True, True, False, True]])
    boundaries = (0, 4, 8, 12, 16)
    support = prepare_sparse_support(code_optimized, valid, binary=False)
    csr_support = prepare_sparse_support(code_csr, valid, binary=False)
    expected = reference(code_reference * valid[..., None], boundaries=boundaries)
    actual = optimized.forward_sparse(support, boundaries=boundaries)
    csr_actual = csr.forward_sparse_csr(csr_support, boundaries=boundaries)
    dense_gate = (code_reference.detach() != 0) & valid[..., None]
    dense_actual = dense.forward_dense_grouped(
        code_dense * dense_gate, boundaries=boundaries
    )
    assert (
        isinstance(expected, tuple)
        and isinstance(actual, tuple)
        and isinstance(csr_actual, tuple)
        and isinstance(dense_actual, tuple)
    )
    for left, right, csr_right, dense_right in zip(
        expected, actual, csr_actual, dense_actual, strict=True
    ):
        torch.testing.assert_close(left, right)
        torch.testing.assert_close(left, csr_right, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(left, dense_right, rtol=1e-6, atol=1e-6)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in csr_actual).backward()
    sum(value.square().sum() for value in dense_actual).backward()
    torch.testing.assert_close(optimized.weight.grad, reference.weight.grad)
    torch.testing.assert_close(optimized.bias.grad, reference.bias.grad)
    torch.testing.assert_close(csr.weight.grad, reference.weight.grad)
    torch.testing.assert_close(csr.bias.grad, reference.bias.grad)
    torch.testing.assert_close(code_csr.grad, code_reference.grad)
    torch.testing.assert_close(dense.weight.grad, reference.weight.grad)
    torch.testing.assert_close(dense.bias.grad, reference.bias.grad)
    torch.testing.assert_close(code_dense.grad, code_reference.grad)


def test_dense_auxk_matches_indexed_loss_and_gradients() -> None:
    torch.manual_seed(21)
    indexed = ActivationDecoder(16, 7)
    dense = ActivationDecoder(16, 7)
    dense.load_state_dict(indexed.state_dict())
    indexed_scores = torch.rand(1, 4, 16, requires_grad=True)
    dense_scores = indexed_scores.detach().clone().requires_grad_(True)
    target = torch.randn(1, 4, 7)
    reconstruction = torch.randn(1, 4, 7)
    valid = torch.tensor([[True, True, False, True]])
    dead = torch.tensor([True] * 10 + [False] * 6)
    indexed_loss = auxk_dead_feature_loss(
        indexed_scores,
        indexed,
        target,
        reconstruction,
        valid,
        dead,
        top_k_aux=4,
        known_dead_feature_count=10,
        sparse_backend="indexed",
    )
    dense_loss = auxk_dead_feature_loss(
        dense_scores,
        dense,
        target,
        reconstruction,
        valid,
        dead,
        top_k_aux=4,
        known_dead_feature_count=10,
        sparse_backend="dense",
    )
    torch.testing.assert_close(dense_loss, indexed_loss, rtol=1e-6, atol=1e-6)
    indexed_loss.backward()
    dense_loss.backward()
    torch.testing.assert_close(dense_scores.grad, indexed_scores.grad)
    torch.testing.assert_close(dense.weight.grad, indexed.weight.grad)
    assert indexed.bias.grad is None and dense.bias.grad is None


def test_split_gradient_projection_and_post_step_normalization_match_reference() -> None:
    torch.manual_seed(13)
    reference = RankOneComponents(7, 5, 6, unit_norm=True)
    optimized = RankOneComponents(7, 5, 6, unit_norm=True)
    optimized.load_state_dict(reference.state_dict())
    gradient = torch.randn_like(reference.U)
    reference.U.grad = gradient.clone()
    optimized.U.grad = gradient.clone()
    reference.normalize_write_decoder_(project_gradients=True)
    optimized.project_write_decoder_gradient_()
    torch.testing.assert_close(optimized.U, reference.U, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(optimized.U.grad, reference.U.grad, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        reference.U.add_(reference.U.grad, alpha=-1e-3)
        optimized.U.add_(optimized.U.grad, alpha=-1e-3)
    reference.normalize_write_decoder_(project_gradients=False)
    optimized.normalize_write_decoder_(project_gradients=False)
    torch.testing.assert_close(optimized.U, reference.U, rtol=1e-6, atol=1e-6)


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


def test_compact_exposure_ledger_preserves_exact_membership_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "compact.sqlite"
    ledger = SQLiteExposureLedger(
        path,
        dataset_manifest_sha256="b" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
        compact_anchor_ids=True,
    )
    anchors = ("1" * 64, "2" * 64, "non-hex-anchor")
    first = ledger.observe_replicated_optimizer_step(0, anchors)
    receipt = ledger.checkpoint_receipt(expected_optimizer_steps=1)
    ledger.observe_replicated_optimizer_step(1, ("2" * 64, "3" * 64))
    recovered = ledger.recover_to_receipt(receipt)
    assert recovered == first
    assert ledger.connection.execute(
        "SELECT COUNT(*) FROM seen_anchors WHERE typeof(anchor_id)='blob'"
    ).fetchone() == (3,)
    ledger.close()

    reopened = SQLiteExposureLedger(
        path,
        dataset_manifest_sha256="b" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
        compact_anchor_ids=True,
    )
    assert reopened.checkpoint_receipt(expected_optimizer_steps=1) == receipt
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


class _TinyConfig:
    def __init__(self):
        self.architecture = SimpleNamespace(
            dictionary_size=16,
            batch_topk_k=2,
            pool_tokens=2,
            latent_tokens_per_model=8,
            packed_token_width=8,
            intra_model_layers=1,
            intra_model_heads=2,
            cross_model_layers=1,
            cross_model_heads=2,
            cross_model_width=8,
            feature_readout="mean",
        )
        self.optimization = SimpleNamespace(
            rank_one_lr=3e-4,
            activation_decoder_lr=3e-4,
            encoder_lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )

    @staticmethod
    def fingerprint() -> str:
        return "tiny-model-owned-test"


def _tiny_registry() -> dict[str, ModelFamilyLayout]:
    return {
        key: ModelFamilyLayout(
            key=key,
            family="llama",
            model_id=f"test/{key}",
            revision="0" * 40,
            tokenizer_id=f"test/{key}",
            tokenizer_revision="0" * 40,
            layers=tuple(range(8)),
            layer_prefixes=("model.layers",),
            residual_pre_suffix="input_layernorm",
            residual_mid_suffix="post_attention_layernorm",
            hidden_size=4,
            intermediate_size=6,
            num_hidden_layers=8,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            q_out=4,
            kv_out=4,
            fused_qkv=False,
            fused_gate_up=False,
        )
        for key in MODEL_ORDER_2B
    }


def _tiny_capture(layout: ModelFamilyLayout, rank: int) -> LocalCaptureBatch:
    generator = torch.Generator().manual_seed(100 + rank)
    inputs = {
        target.key: torch.randn(1, 2, target.d_in, generator=generator)
        for target in layout.targets()
    }
    outputs = {
        target.key: torch.randn(1, 2, target.d_out, generator=generator)
        for target in layout.targets()
    }
    return LocalCaptureBatch(
        model_key=layout.key,
        activations={
            site: tuple(
                torch.randn(1, 2, layout.hidden_size, generator=generator)
                for _ in range(8)
            )
            for site in ("pre", "mid")
        },
        matrix_inputs=inputs,
        matrix_outputs=outputs,
        valid_anchors=torch.ones(1, 2, dtype=torch.bool),
    )


def _distributed_model_owned_worker(
    rank: int, world_size: int, init_file: str, checkpoint_root: str
) -> None:
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    torch.manual_seed(0)
    config = _TinyConfig()
    registry = _tiny_registry()
    key = MODEL_ORDER_2B[rank]
    model = ModelOwnedMain2B(config, registry, key, dtype=torch.float32)
    model.synchronize_shared_state_()
    optimizer = build_model_owned_adam(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    capture = _tiny_capture(registry[key], rank)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        scores, sparse, continuous, binary = model.encode(capture)
        for site in ("pre", "mid"):
            for state in model.gates[site].state_dict().values():
                gathered = [torch.empty_like(state) for _ in range(world_size)]
                dist.all_gather(gathered, state)
                assert all(torch.equal(gathered[0], value) for value in gathered[1:])
        activation_terms = list(
            model.activation_loss_terms(capture, scores, sparse, continuous)
        )
        internal_terms = list(model.internal_loss_terms(capture, binary))
        loss = sum(
            coefficient * (activation + 0.03125 * auxk)
            for _name, activation, auxk, coefficient in activation_terms
        ) + sum(
            coefficient * value for _name, _site, value, coefficient in internal_terms
        )
        loss.backward()
        model.all_reduce_shared_gradients_()
        assert all(parameter.grad is not None for parameter in model.shared_encoders.parameters())
        assert all(parameter.grad is not None for parameter in model.local_encoders.parameters())
        assert all(parameter.grad is not None for parameter in model.activation_banks.parameters())
        assert all(parameter.grad is not None for parameter in model.matrix_bank.parameters())
        model.project_decoder_gradients_()
        optimizer.step()
        model.normalize_decoders_(project_gradients=False)
        scheduler.step()

    update()
    manager = ModelOwnedCheckpoint(checkpoint_root, keep_complete=2)
    saved_parameter = next(model.parameters()).detach().clone()
    manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state={"completed_steps": 1, "accepted_exposures": 8192},
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        model_key=key,
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)
    runtime, manifest = manager.load(
        path=Path(checkpoint_root) / "step-000000001",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        expected_model_key=key,
    )
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    assert runtime["completed_steps"] == 1
    assert manifest["status"] == "COMPLETE"
    with torch.no_grad():
        next(model.parameters()).sub_(2)
    runtime, manifest = manager.load_model_only(
        path=Path(checkpoint_root) / "step-000000001",
        model=model,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        expected_model_key=key,
    )
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    assert runtime["completed_steps"] == 1
    assert manifest["status"] == "COMPLETE"
    with torch.no_grad():
        next(model.parameters()).add_(3)
    runtime, manifest = manager.load_model_only_owner(
        path=Path(checkpoint_root) / "step-000000001",
        model=model,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        checkpoint_rank=rank,
        expected_checkpoint_world_size=world_size,
        expected_model_key=key,
    )
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    assert runtime["completed_steps"] == 1
    assert manifest["world_size"] == world_size
    update()
    dist.destroy_process_group()


def test_model_owned_distributed_gradients_optimizer_and_checkpoint(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    init_file = tmp_path / f"dist-{uuid4().hex}"
    mp.spawn(
        _distributed_model_owned_worker,
        args=(3, str(init_file), str(tmp_path / "checkpoint")),
        nprocs=3,
        join=True,
    )

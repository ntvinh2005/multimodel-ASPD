from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from aspd.multimodel.batch_topk import BatchTopKController, batch_topk
from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.config import (
    EncoderSpec,
    ObjectiveSpec,
    OriginalASPDRecipeSpec,
    SparsitySpec,
    load_experiment_config,
)
from aspd.multimodel.losses import fvu
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_evaluation import EvaluationGroup
from aspd.multimodel.original_recipe import (
    DatasetIntegrityError,
    StreamCursor,
    validate_dataset_manifest,
    validate_original_recipe_config,
)
from aspd.multimodel.original_steering import freeze_clean_original_gate
from aspd.multimodel.original_streaming import (
    DistributedTopology,
    FrozenTokenShardStream,
    checkpoint_payload,
    restore_checkpoint_payload,
    split_global_batch,
)
from aspd.multimodel.training import train

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"


def test_frozen_recipe_has_exact_budget_and_is_fail_closed_on_data() -> None:
    cfg = load_experiment_config(CONFIG)
    assert validate_original_recipe_config(cfg)["status"] == "PASS"
    assert cfg.training.max_steps * 8 * 512 == 99_999_744
    assert validate_dataset_manifest(cfg, require_ready=False)["status"] == "BLOCKED"
    with pytest.raises(DatasetIntegrityError, match="BLOCKED"):
        validate_dataset_manifest(cfg, require_ready=True)


def test_frozen_recipe_refuses_the_legacy_cached_trainer() -> None:
    cfg = load_experiment_config(CONFIG)
    with pytest.raises(RuntimeError, match="cli.train_original_recipe"):
        train(cfg, device_name="cpu")


def test_qwen2_decoder_executes_o_projection_before_mid_residual_hook() -> None:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=15,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    events: list[str] = []
    captured: dict[str, torch.Tensor] = {}
    layer = model.model.layers[14]

    def o_input(_module, args) -> None:
        events.append("o_proj_input")
        captured["X_O"] = args[0].detach().clone()

    def mid_input(_module, args) -> None:
        events.append("mid_residual_input")
        captured["R_mid"] = args[0].detach().clone()

    handles = [
        layer.self_attn.o_proj.register_forward_pre_hook(o_input),
        layer.post_attention_layernorm.register_forward_pre_hook(mid_input),
    ]
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert events == ["o_proj_input", "mid_residual_input"]
    assert captured["X_O"].shape == captured["R_mid"].shape == (1, 4, 8)
    assert not torch.equal(captured["X_O"], captured["R_mid"])


def test_transcoder_biases_and_unit_norm_match_original_equation() -> None:
    torch.manual_seed(0)
    component = RankOneComponents(
        2, 2, 2, transcoder_biases=True, unit_norm=True
    )
    torch.testing.assert_close(component.V.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(component.U.norm(dim=-1), torch.ones(2))
    with torch.no_grad():
        component.V.copy_(torch.eye(2))
        component.U.copy_(torch.eye(2))
        component.b_dec.copy_(torch.tensor([1.0, 2.0]))
        component.b_out.copy_(torch.tensor([10.0, 20.0]))
    x = torch.tensor([[[4.0, 6.0]]])
    gate = torch.tensor([[[True, False]]])
    output = component.reconstruct(x, gate, torch.tensor([[True]]))
    torch.testing.assert_close(output, torch.tensor([[[13.0, 20.0]]]))


def test_original_encoder_center_is_the_activation_reconstruction_bias() -> None:
    model = MultiModelASPD(
        ["base", "target"],
        [3, 3],
        [{"o_proj": torch.eye(2)}, {"o_proj": torch.eye(2)}],
        EncoderSpec(),
        SparsitySpec(n_features=4, top_k=1, pool_tokens=2),
        ObjectiveSpec(
            auxk_coefficient=0.0,
            top_k_aux=2,
            dead_after_batches=3,
            model_axis_reduction="mean",
            matryoshka_group_fractions=[0.5, 0.5],
        ),
        OriginalASPDRecipeSpec(enabled=True),
    )
    assert model.activation_decoders.biases[0] is model.encoder.centers[0]
    assert model.activation_decoders.biases[1] is model.encoder.centers[1]
    model.calibrate_original_recipe([torch.arange(3.0), torch.arange(3.0) + 5])
    torch.testing.assert_close(model.encoder.centers[0], torch.arange(3.0))


def test_affine_component_bias_cancels_from_input_change_reconstruction() -> None:
    component = RankOneComponents(1, 2, 2, transcoder_biases=True)
    with torch.no_grad():
        component.V.copy_(torch.tensor([[1.0, 0.0]]))
        component.U.copy_(torch.tensor([[2.0, 3.0]]))
        component.b_dec.copy_(torch.tensor([0.5, 1.0]))
        component.b_out.copy_(torch.tensor([11.0, 13.0]))
    valid = torch.tensor([[True]])
    gate = torch.tensor([[[True]]])
    x0 = torch.tensor([[[2.0, 4.0]]])
    x1 = torch.tensor([[[5.0, 7.0]]])
    difference = component.reconstruct(x1, gate, valid) - component.reconstruct(x0, gate, valid)
    torch.testing.assert_close(difference, torch.tensor([[[6.0, 9.0]]]))


def test_evaluation_group_obeys_true_and_aspd_delta_identities() -> None:
    model = MultiModelASPD(
        ["base", "target"],
        [2, 2],
        [{"o_proj": torch.eye(2)}, {"o_proj": 2 * torch.eye(2)}],
        EncoderSpec(),
        SparsitySpec(n_features=2, top_k=1, pool_tokens=1),
        ObjectiveSpec(
            auxk_coefficient=0.0,
            top_k_aux=1,
            dead_after_batches=2,
            model_axis_reduction="mean",
            matryoshka_group_fractions=[0.5, 0.5],
        ),
        OriginalASPDRecipeSpec(enabled=True),
    )
    group = EvaluationGroup(2)
    residuals = [torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[1.0, 2.0]]])]
    inputs = [torch.tensor([[[1.0, 2.0]]]), torch.tensor([[[3.0, 5.0]]])]
    values = torch.tensor([[[1.0, 0.0]]])
    gate = values.bool()
    group.update(model, residuals, inputs, values, gate, torch.tensor([[True]]))
    result = group.result()
    assert result["decomposition_identity"]["true"]["max_abs_residual"] == 0.0
    assert result["decomposition_identity"]["aspd"]["max_abs_residual"] < 1e-6


def test_batch_topk_uses_independent_local_token_pools() -> None:
    cfg = SparsitySpec(n_features=2, top_k=1, pool_tokens=2)
    pre = torch.tensor([[[4.0, 3.0], [2.0, 1.0], [8.0, 7.0], [6.0, 5.0]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    code = batch_topk(pre, valid, cfg)
    assert code.gate[:, :2].sum() == 2
    assert code.gate[:, 2:].sum() == 2


def test_jump_relu_threshold_updates_per_pool_and_round_trips() -> None:
    cfg = SparsitySpec(n_features=2, top_k=1, pool_tokens=2, threshold_lr=0.01)
    controller = BatchTopKController(cfg, dead_after_pools=3).train()
    pre = torch.tensor([[[4.0, 3.0], [2.0, 1.0], [8.0, 7.0], [6.0, 5.0]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    controller(pre, valid)
    assert controller.threshold.item() == pytest.approx(0.0997)
    restored = BatchTopKController(cfg, dead_after_pools=3)
    restored.load_state_dict(controller.state_dict())
    torch.testing.assert_close(restored.threshold, controller.threshold)
    torch.testing.assert_close(restored.n_pools_not_active, controller.n_pools_not_active)


def test_two_model_objective_uses_explicit_model_axis_mean() -> None:
    sparsity = SparsitySpec(n_features=4, top_k=1, pool_tokens=2)
    objective = ObjectiveSpec(
        auxk_coefficient=0.0,
        top_k_aux=2,
        dead_after_batches=3,
        model_axis_reduction="mean",
        matryoshka_group_fractions=[0.5, 0.5],
    )
    model = MultiModelASPD(
        ["base", "target"],
        [3, 3],
        [{"o_proj": torch.eye(2)}, {"o_proj": torch.eye(2)}],
        EncoderSpec(),
        sparsity,
        objective,
        OriginalASPDRecipeSpec(enabled=True),
    ).train()
    batch = {
        "valid_tokens": torch.ones(1, 2, dtype=torch.bool),
        "R": [torch.randn(1, 2, 3), torch.randn(1, 2, 3)],
        "X": [
            {"o_proj": torch.randn(1, 2, 2)},
            {"o_proj": torch.randn(1, 2, 2)},
        ],
    }
    result = model(batch, update_dead_tracker=False)
    expected = sum(
        result[f"internal/{name}"] + result[f"act/{name}"]
        for name in ("base", "target")
    ) / 2
    torch.testing.assert_close(result["loss/full"], expected)


def test_original_auxk_residual_path_is_not_detached() -> None:
    model = MultiModelASPD(
        ["base", "target"],
        [2, 2],
        [{"o_proj": torch.eye(2)}, {"o_proj": torch.eye(2)}],
        EncoderSpec(),
        SparsitySpec(n_features=2, top_k=1, pool_tokens=1),
        ObjectiveSpec(
            top_k_aux=1,
            dead_after_batches=1,
            model_axis_reduction="mean",
            matryoshka_group_fractions=[0.5, 0.5],
        ),
        OriginalASPDRecipeSpec(enabled=True),
    )
    model.sparsity_controller.n_pools_not_active.fill_(1)
    with torch.no_grad():
        for decoder in model.activation_decoders.full:
            decoder.fill_(1.0)
    reconstructions = [
        torch.full((1, 2, 2), 0.25, requires_grad=True) for _ in range(2)
    ]
    loss = model._auxk_loss(
        [torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])] * 2,
        reconstructions,
        torch.ones(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in reconstructions)


def test_original_p6_gate_is_a_detached_clean_snapshot() -> None:
    model = MultiModelASPD(
        ["base", "target"],
        [2, 2],
        [{"o_proj": torch.eye(2)}, {"o_proj": torch.eye(2)}],
        EncoderSpec(),
        SparsitySpec(n_features=2, top_k=1, pool_tokens=1),
        ObjectiveSpec(
            auxk_coefficient=0.0,
            top_k_aux=1,
            dead_after_batches=2,
            model_axis_reduction="mean",
            matryoshka_group_fractions=[0.5, 0.5],
        ),
        OriginalASPDRecipeSpec(enabled=True),
    ).eval()
    residuals = [torch.randn(1, 2, 2), torch.randn(1, 2, 2)]
    valid = torch.ones(1, 2, dtype=torch.bool)
    gate = freeze_clean_original_gate(model, residuals, valid, [0])
    assert gate.shape == (1, 2, 1) and not gate.requires_grad
    frozen = gate.clone()
    residuals[0].add_(1000)
    assert torch.equal(gate, frozen)


def test_stream_cursor_prevents_resume_replay_or_skip() -> None:
    cfg = load_experiment_config(CONFIG)
    cursor = StreamCursor.for_step(cfg, 1_234)
    assert cursor.sequence_index == 1_234 * 8
    assert cursor.token_exposures == 1_234 * 8 * 512
    cursor.validate(cfg)
    with pytest.raises(ValueError, match="replay or skip"):
        StreamCursor(1_234, cursor.sequence_index - 1, cursor.token_exposures).validate(cfg)


def test_frozen_token_stream_crosses_shards_without_duplication(tmp_path: Path) -> None:
    for index, rows in enumerate((3, 2)):
        start = 0 if index == 0 else 3
        ids = torch.arange(start * 4, (start + rows) * 4).reshape(rows, 4)
        torch.save(
            {"input_ids": ids, "valid_tokens": torch.ones_like(ids, dtype=torch.bool)},
            tmp_path / f"shard_{index}.pt",
        )
    manifest = {
        "sequence_length": 4,
        "shards": [
            {"split": "train", "sequence_start": 0, "sequence_count": 3, "path": "shard_0.pt"},
            {"split": "train", "sequence_start": 3, "sequence_count": 2, "path": "shard_1.pt"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    batch = FrozenTokenShardStream(path, "train").batch_at(2, 3)
    assert batch["input_ids"][:, 0].tolist() == [8, 12, 16]


def test_distributed_topology_preserves_two_2048_token_pools() -> None:
    cfg = load_experiment_config(CONFIG)
    topology = DistributedTopology()
    topology.validate(cfg)
    assert topology.positions_per_trainer == 2_048
    assert topology.global_positions == 4_096
    batch = {
        "input_ids": torch.arange(8 * 512).reshape(8, 512),
        "valid_tokens": torch.ones(8, 512, dtype=torch.bool),
    }
    shards = split_global_batch(batch, 2, 4)
    assert [int(shard["valid_tokens"].sum()) for shard in shards] == [2_048, 2_048]
    assert not set(shards[0]["input_ids"].flatten().tolist()) & set(
        shards[1]["input_ids"].flatten().tolist()
    )


def test_two_rank_gradient_average_equals_global_batch_mean() -> None:
    initial = torch.tensor([[0.3, -0.2]])
    local_inputs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, -1.0]])]
    local_targets = [torch.tensor([[0.5]]), torch.tensor([[-0.4]])]
    local_gradients = []
    for x, y in zip(local_inputs, local_targets, strict=True):
        weight = initial.clone().requires_grad_(True)
        torch.nn.functional.mse_loss(x @ weight.t(), y).backward()
        local_gradients.append(weight.grad.clone())
    global_weight = initial.clone().requires_grad_(True)
    global_loss = torch.stack(
        [
            torch.nn.functional.mse_loss(x @ global_weight.t(), y)
            for x, y in zip(local_inputs, local_targets, strict=True)
        ]
    ).mean()
    global_loss.backward()
    torch.testing.assert_close(torch.stack(local_gradients).mean(0), global_weight.grad)


def test_two_model_mean_preserves_shared_loss_gradient_scale() -> None:
    target = torch.tensor([[[0.0], [2.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    shared_single = torch.tensor(0.3, requires_grad=True)
    single_loss = fvu(target, shared_single.expand_as(target), valid)
    single_loss.backward()
    single_gradient = shared_single.grad.clone()
    shared_pair = torch.tensor(0.3, requires_grad=True)
    pair_loss = torch.stack(
        [fvu(target, shared_pair.expand_as(target), valid) for _ in range(2)]
    ).mean()
    pair_loss.backward()
    torch.testing.assert_close(shared_pair.grad, single_gradient)


def _toy_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()


def test_checkpoint_resume_matches_uninterrupted_toy_training() -> None:
    cfg = load_experiment_config(CONFIG)
    inputs = [torch.tensor([[float(index), 1.0]]) for index in range(1, 6)]
    torch.manual_seed(9)
    uninterrupted = nn.Linear(2, 1)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    for value in inputs:
        _toy_step(uninterrupted, uninterrupted_optimizer, value)

    torch.manual_seed(9)
    interrupted = nn.Linear(2, 1)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=0.01)
    for value in inputs[:3]:
        _toy_step(interrupted, interrupted_optimizer, value)
    payload = checkpoint_payload(
        cfg,
        interrupted,
        interrupted_optimizer,
        3,
        sampler_state_by_rank={0: {"sequence_index": 24}, 1: {"sequence_index": 28}},
    )
    resumed = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    step, sampler = restore_checkpoint_payload(
        cfg, resumed, resumed_optimizer, payload
    )
    assert step == 3 and set(sampler) == {0, 1}
    for value in inputs[step:]:
        _toy_step(resumed, resumed_optimizer, value)
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

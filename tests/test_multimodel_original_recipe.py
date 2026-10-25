from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from aspd.multimodel.batch_topk import BatchTopKController, batch_topk
from aspd.multimodel.cache import _Capture, _register_capture_hooks
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
from aspd.multimodel.original_evaluation import EvaluationGroup, _behavioral_prompts
from aspd.multimodel.original_recipe import (
    DatasetIntegrityError,
    StreamCursor,
    validate_dataset_manifest,
    validate_original_recipe_config,
)
from aspd.multimodel.original_steering import (
    extract_original_component_for_p6,
    freeze_clean_original_gate,
)
from aspd.multimodel.original_streaming import (
    DistributedTopology,
    FrozenTokenShardStream,
    _capture_pair,
    _truncate_metrics_for_resume,
    checkpoint_payload,
    restore_checkpoint_payload,
    split_global_batch,
)
from aspd.multimodel.original_verification import _metrics, _runtime
from aspd.multimodel.training import train

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"


def test_frozen_recipe_has_exact_budget_and_is_fail_closed_on_data(tmp_path: Path) -> None:
    cfg = load_experiment_config(CONFIG)
    assert validate_original_recipe_config(cfg)["status"] == "PASS"
    assert cfg.training.max_steps * 8 * 512 == 99_999_744
    missing = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={"frozen_manifest": str(tmp_path / "not_materialized.json")}
            )
        }
    )
    assert validate_dataset_manifest(missing, require_ready=False)["status"] == "NOT_RUN"
    with pytest.raises(DatasetIntegrityError, match="not found"):
        validate_dataset_manifest(missing, require_ready=True)


def test_frozen_recipe_refuses_the_legacy_cached_trainer() -> None:
    cfg = load_experiment_config(CONFIG)
    with pytest.raises(RuntimeError, match="cli.train_original_recipe"):
        train(cfg, device_name="cpu")


def test_locked_behavioral_manifest_matches_model_pair_and_tokenizer() -> None:
    rows, provenance = _behavioral_prompts(load_experiment_config(CONFIG))
    assert len(rows) == 180
    assert provenance["status"] == "VERIFIED"
    assert provenance["category_counts"] == {
        "risk": 60,
        "sports_control": 60,
        "neutral": 60,
    }
    assert provenance["expected_matched_pair_count"] == 60


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


def test_tokens_to_capture_train_checkpoint_resume_and_evaluation_pipeline() -> None:
    cfg = load_experiment_config(CONFIG)
    qwen_cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=15,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(4)
    sources = [Qwen2ForCausalLM(qwen_cfg).eval(), Qwen2ForCausalLM(qwen_cfg).eval()]
    with torch.no_grad():
        sources[1].model.layers[14].self_attn.o_proj.weight.add_(0.01)
    source_before = [
        {name: value.detach().clone() for name, value in source.state_dict().items()}
        for source in sources
    ]
    captures = [_Capture(), _Capture()]
    tokens = torch.tensor([[1, 2, 3, 4]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    with ExitStack() as stack:
        for source, spec, capture in zip(sources, cfg.models, captures, strict=True):
            _register_capture_hooks(source, spec, capture, stack)
        paired = _capture_pair(sources, captures, tokens, valid, torch.device("cpu"))
    residuals = [value.float() for value in paired["R"]]  # type: ignore[union-attr]
    inputs = [value["o_proj"].float() for value in paired["X"]]  # type: ignore[index,union-attr]
    assert not torch.equal(residuals[0], inputs[0])
    sparsity = cfg.sparsity.model_copy(
        update={"n_features": 8, "top_k": 2, "pool_tokens": 4}
    )
    objective = cfg.objective.model_copy(
        update={
            "top_k_aux": 4,
            "dead_after_batches": 2,
            "alive_window_tokens": 16,
            "matryoshka_group_fractions": [0.5, 0.5],
        }
    )
    weights = [
        {"o_proj": source.model.layers[14].self_attn.o_proj.weight.detach().clone()}
        for source in sources
    ]
    decomposition = MultiModelASPD(
        ["base", "target"],
        [8, 8],
        weights,
        EncoderSpec(),
        sparsity,
        objective,
        OriginalASPDRecipeSpec(enabled=True, calibration_batches=1),
    ).train()
    decomposition.calibrate_original_recipe(
        [value.mean((0, 1)) for value in residuals]
    )
    optimizer = torch.optim.AdamW(decomposition.parameters(), lr=3e-4)
    batch = {
        "valid_tokens": valid,
        "R": residuals,
        "X": [{"o_proj": value} for value in inputs],
    }
    optimizer.zero_grad(set_to_none=True)
    result = decomposition(batch)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in decomposition.parameters()
    )
    decomposition.normalize_original_decoders_()
    optimizer.step()
    payload = checkpoint_payload(
        cfg,
        decomposition,
        optimizer,
        1,
        sampler_state_by_rank={0: {"sequence_index": 8}, 1: {"sequence_index": 12}},
    )
    restored = MultiModelASPD(
        ["base", "target"],
        [8, 8],
        weights,
        EncoderSpec(),
        sparsity,
        objective,
        OriginalASPDRecipeSpec(enabled=True, calibration_batches=1),
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    step, _ = restore_checkpoint_payload(cfg, restored, restored_optimizer, payload)
    assert step == 1
    restored.eval()
    code = restored.encode(residuals, valid, update_statistics=False)
    group = EvaluationGroup(8)
    group.update(restored, residuals, inputs, code.values, code.gate, valid)
    assert group.result()["tokens"] == 4
    for model_index in (0, 1):
        assert restored.component_factors(model_index, "o_proj")[0].shape == (8, 8)
    assert restored.component_factors(0, "o_proj")[0].data_ptr() != restored.component_factors(
        1, "o_proj"
    )[0].data_ptr()
    for source, before in zip(sources, source_before, strict=True):
        assert all(torch.equal(value, before[name]) for name, value in source.state_dict().items())


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


def test_original_p6_component_export_reproduces_trained_factors() -> None:
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
    ).eval()
    record = extract_original_component_for_p6(model, 1)
    assert record["component_id"] == 1
    for model_index, values in enumerate(record["models"]):
        u, v = model.component_factors(model_index, "o_proj")
        torch.testing.assert_close(values["u"], u[1])
        torch.testing.assert_close(values["v"], v[1])
        assert values["u"].data_ptr() != u[1].data_ptr()
        assert values["v"].data_ptr() != v[1].data_ptr()


def test_stream_cursor_prevents_resume_replay_or_skip() -> None:
    cfg = load_experiment_config(CONFIG)
    cursor = StreamCursor.for_step(cfg, 1_234)
    assert cursor.sequence_index == 1_234 * 8
    assert cursor.token_exposures == 1_234 * 8 * 512
    cursor.validate(cfg)
    with pytest.raises(ValueError, match="replay or skip"):
        StreamCursor(1_234, cursor.sequence_index - 1, cursor.token_exposures).validate(cfg)


def test_resume_discards_metrics_after_atomic_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "training_metrics.jsonl"
    path.write_text(
        "".join(
            __import__("json").dumps({"step": step, "train/loss": 1.0}) + "\n"
            for step in (1, 10, 20, 30)
        ),
        encoding="utf-8",
    )
    _truncate_metrics_for_resume(tmp_path, 20)
    rows = [
        __import__("json").loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 10, 20]


def test_smoke_throughput_excludes_stress_checkpoint_cadence(tmp_path: Path) -> None:
    records = [
        {
            "rank": 0,
            "role": "producer",
            "start_step": 0,
            "end_step": 50,
            "segment_wall_seconds": 20.0,
            "rank_wall_seconds_including_startup": 30.0,
            "capture_forward_seconds": 18.0,
            "communication_seconds": 1.0,
            "peak_gpu_memory_bytes": 1,
            "peak_cpu_rss_kib": 1,
        },
        *[
            {
                "rank": rank,
                "role": "trainer",
                "start_step": 0,
                "end_step": 50,
                "segment_wall_seconds": 50.0,
                "rank_wall_seconds_including_startup": 60.0,
                "checkpoint_seconds": 40.0,
                "trainer_step_seconds": 8.0,
                "peak_gpu_memory_bytes": 1,
                "peak_cpu_rss_kib": 1,
            }
            for rank in (1, 2)
        ],
    ]
    for record in records:
        path = tmp_path / f"runtime_rank{record['rank']}_steps_00000000_00000050.json"
        path.write_text(__import__("json").dumps(record), encoding="utf-8")

    report = _runtime(tmp_path)

    assert report["positions_per_second_excluding_startup"] == pytest.approx(
        50 * 4096 / 20.0
    )
    assert report["positions_per_second_with_smoke_checkpoint_cadence"] == pytest.approx(
        50 * 4096 / 50.0
    )
    assert report["checkpoint_write_seconds"] == pytest.approx(40.0)


def test_verifier_accepts_only_the_final_partial_metric_interval(tmp_path: Path) -> None:
    metrics = tmp_path / "training_metrics.jsonl"
    metrics.write_text(
        __import__("json").dumps(
            {
                "step": 24_410,
                "token_exposures": 24_410 * 4096,
                "train/loss": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = _metrics(
        tmp_path,
        expected_step=24_414,
        expected_exposures=99_999_744,
        log_every=10,
    )
    assert rows[-1]["step"] == 24_410

    metrics.write_text(
        __import__("json").dumps(
            {
                "step": 24_400,
                "token_exposures": 24_400 * 4096,
                "train/loss": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected final step"):
        _metrics(
            tmp_path,
            expected_step=24_414,
            expected_exposures=99_999_744,
            log_every=10,
        )


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

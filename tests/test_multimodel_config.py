from pathlib import Path

from aspd.multimodel.config import load_experiment_config

ROOT = Path(__file__).parents[1]


def test_all_qwen_multimodel_configs_expand_and_validate() -> None:
    config_dir = ROOT / "configs" / "multimodel" / "qwen3_1_7b"
    for path in config_dir.glob("*.yaml"):
        cfg = load_experiment_config(path)
        assert len(cfg.models) == 2
        assert cfg.sparsity.n_features > 0


def test_smoke_config_overrides_parent_without_losing_models() -> None:
    cfg = load_experiment_config(ROOT / "configs/multimodel/qwen3_1_7b/smoke.yaml")
    assert cfg.sparsity.n_features == 512
    assert cfg.sparsity.diffing == "D2"
    assert [model.name for model in cfg.models] == ["base", "finetuned"]


def test_d0_s1_debug_config_is_the_uncomplicated_core_method() -> None:
    cfg = load_experiment_config(
        ROOT / "configs/multimodel/qwen3_1_7b/d0_s1_debug.yaml"
    )
    assert cfg.name == "qwen3_1_7b_d0_s1_debug"
    assert cfg.encoder.kind == "linear"
    assert cfg.encoder.aggregation == "mean"
    assert cfg.sparsity.n_features == 512
    assert cfg.sparsity.selection_score == "S1"
    assert cfg.sparsity.diffing == "D0"
    assert cfg.sparsity.top_k == 8
    assert not cfg.sparsity.tie_shared_decoders
    assert not cfg.sparsity.tie_shared_mechanisms
    assert cfg.data.train_tokens == 16_384
    assert cfg.data.validation_tokens == 4_096
    assert cfg.objective.top_k_aux == 64
    assert cfg.objective.dead_after_batches == 100
    assert cfg.training.max_steps == 200
    assert cfg.training.batch_size_sequences == 2
    assert cfg.training.validate_every == 25
    assert cfg.training.validation_batches == 4


def test_d0_s1_capacity_config_matches_main_workload() -> None:
    main = load_experiment_config(ROOT / "configs/multimodel/qwen3_1_7b/d0_s1.yaml")
    capacity = load_experiment_config(
        ROOT / "configs/multimodel/qwen3_1_7b/d0_s1_capacity.yaml"
    )

    assert capacity.name == "qwen3_1_7b_d0_s1_capacity"
    assert capacity.encoder == main.encoder
    assert capacity.sparsity == main.sparsity
    assert capacity.objective == main.objective
    assert capacity.models == main.models
    assert capacity.data.sequence_length == main.data.sequence_length == 256
    assert capacity.training.batch_size_sequences == main.training.batch_size_sequences == 4
    assert capacity.training.gradient_accumulation_steps == 1
    assert capacity.training.parameter_dtype == main.training.parameter_dtype == "float32"
    assert capacity.training.autocast == main.training.autocast == "bfloat16"
    assert capacity.training.learning_rate == main.training.learning_rate == 3e-4
    assert capacity.data.train_tokens == 32_768
    assert capacity.data.validation_tokens == 32_768
    assert capacity.training.max_steps == 100
    assert capacity.training.log_every == 1
    assert capacity.training.validate_every == 50
    assert capacity.training.validation_batches == 32
    assert capacity.training.save_every == 75
    assert capacity.training.resume is None

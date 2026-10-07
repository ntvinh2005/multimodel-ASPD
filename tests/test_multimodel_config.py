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

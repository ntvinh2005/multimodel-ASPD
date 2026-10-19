from __future__ import annotations

from pathlib import Path

from aspd.multimodel.config import load_experiment_config

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_fidelity1000.yaml"
PILOT = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml"
PREFLIGHT = ROOT / "slurm/multimodel/qwen25_extreme_sports_aspd_fidelity_preflight.sbatch"
TRAIN = ROOT / "slurm/multimodel/qwen25_extreme_sports_aspd_fidelity1000_train.sbatch"


def test_controlled_config_changes_only_budget_and_administration() -> None:
    pilot = load_experiment_config(PILOT)
    controlled = load_experiment_config(CONFIG)
    assert controlled.models == pilot.models
    assert controlled.data == pilot.data
    assert controlled.cache == pilot.cache
    assert controlled.encoder == pilot.encoder
    assert controlled.sparsity == pilot.sparsity
    assert controlled.objective == pilot.objective
    assert controlled.training.model_copy(
        update={
            "output_dir": pilot.training.output_dir,
            "max_steps": pilot.training.max_steps,
            "resume": pilot.training.resume,
            "validate_every": pilot.training.validate_every,
            "validation_batches": pilot.training.validation_batches,
            "save_every": pilot.training.save_every,
            "keep_last_checkpoints": pilot.training.keep_last_checkpoints,
        }
    ) == pilot.training
    assert controlled.training.max_steps == 1000
    assert controlled.training.resume is None
    assert controlled.training.validation_batches == 32
    assert controlled.training.output_dir.endswith("qwen25_extreme_sports_aspd_fidelity1000")


def test_slurm_is_fail_closed_fresh_and_dependency_is_external_afterok() -> None:
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    for script in (preflight, train):
        assert "set -euo pipefail" in script
        assert "#SBATCH --partition=hpg-b200" in script
        assert "#SBATCH --gres=gpu:b200:1" in script
        assert "#SBATCH --cpus-per-task=14" in script
        assert "#SBATCH --mem=128gb" in script
        assert "delta_crosscoder" not in script.lower()
        assert "causal" not in script.lower()
    assert "tiny_overfit" in preflight
    assert "evaluate_computation_fidelity" in preflight
    assert "ASPD_FIDELITY_PREFLIGHT_COMPLETE" in preflight
    assert "resume is None" in train
    assert "Refusing to reuse or overwrite" in train
    assert "--dependency=afterok" in train
    assert "ASPD_FIDELITY1000_TRAINING_COMPLETE" in train
    # The preflight launcher never starts training; failed preflight therefore exits nonzero
    # and Slurm's externally supplied afterok dependency can never release the train job.
    assert "aspd.multimodel.cli.train" not in preflight

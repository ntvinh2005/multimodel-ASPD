import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.posthoc_report import (
    TAXONOMY_CATEGORIES,
    build_posthoc_report,
    validate_posthoc_artifacts,
)

ROOT = Path(__file__).parents[1]


def _fixture(tmp_path):
    original = load_experiment_config(ROOT / "configs/multimodel/qwen3_1_7b/d0_s1_main.yaml")
    cfg = original.model_copy(
        update={
            "data": original.data.model_copy(update={"validation_tokens": 8}),
            "sparsity": original.sparsity.model_copy(update={"n_features": 4, "top_k": 1}),
        }
    )
    activation_rho = torch.tensor([[0.50, 0.50, 0.50, 0.95], [0.50, 0.50, 0.50, 0.05]])
    mechanism_beta = torch.tensor([[0.95, 0.10, 1.00, 1.90], [0.05, 0.90, 1.00, 0.10]])
    mechanism_rho = mechanism_beta / mechanism_beta.sum(dim=0, keepdim=True)
    tensors = {
        "decoder_norm_NC": activation_rho.clone(),
        "activation_rho_NC": activation_rho,
        "mechanism_beta_NC": mechanism_beta,
        "mechanism_rho_NC": mechanism_rho,
        "fire_count_C": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "activation_dominant_model_C": activation_rho.argmax(dim=0),
        "mechanism_dominant_model_C": mechanism_rho.argmax(dim=0),
    }
    matrices = [matrix.name for matrix in cfg.models[0].matrices]
    for model_index, model in enumerate(cfg.models):
        for matrix in matrices:
            # Seven equal contributions make locus sum exactly one for every active feature.
            tensors[f"beta/{model.name}/{matrix}"] = mechanism_beta[model_index] / len(matrices)
            tensors[f"locus/{model.name}/{matrix}"] = torch.full((4,), 1 / len(matrices))
    for matrix in matrices:
        tensors[f"pair/{matrix}/component_cosine"] = torch.tensor([0.2, 0.3, 0.9, 0.1])
        tensors[f"pair/{matrix}/relative_component_change"] = torch.tensor([1.2, 1.1, 0.1, 2.0])
        tensors[f"pair/{matrix}/read_cosine"] = torch.tensor([0.8, 0.7, 0.95, 0.2])
        tensors[f"pair/{matrix}/write_cosine"] = torch.tensor([0.25, 0.4, 0.95, 0.5])

    taxonomy = {
        "shared_activation_shared_mechanism": [2],
        "shared_activation_concentrated_mechanism": [0, 1],
        "concentrated_activation_shared_mechanism": [],
        "concentrated_activation_concentrated_mechanism": [3],
        "mixed_or_subset_mass": [],
    }
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    save_file(tensors, str(analysis_dir / "posthoc.safetensors"))
    (analysis_dir / "taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    (analysis_dir / "top_activation_examples.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "g_s": 2.5,
                        "text": "a repeated context",
                        "center_token": "context",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return cfg, tensors, taxonomy, analysis_dir


def test_posthoc_report_writes_validated_review_artifacts(tmp_path) -> None:
    cfg, _tensors, _taxonomy, analysis_dir = _fixture(tmp_path)
    output = build_posthoc_report(
        cfg,
        analysis_dir,
        top_per_direction=1,
        control_count=1,
        low_support_threshold=3,
        dpi=60,
    )

    expected = {
        "feature_table.csv",
        "summary.json",
        "rho_scatter.png",
        "rho_mass_scatter.png",
        "taxonomy_counts.png",
        "activation_rho_hist.png",
        "mechanism_rho_hist.png",
        "fire_count_hist.png",
        "candidates.md",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert all((output / name).stat().st_size > 0 for name in expected)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["invariants"]["observed_total_fires"] == 8
    assert summary["review_feature_ids"]["base_dominant_targets"] == [0]
    assert summary["review_feature_ids"]["finetuned_dominant_targets"] == [1]
    assert sum(summary["taxonomy_counts"].values()) == 4
    candidates = (output / "candidates.md").read_text(encoding="utf-8")
    assert "Feature 0" in candidates and "a repeated context" in candidates


def test_posthoc_validation_rejects_wrong_total_fire_count(tmp_path) -> None:
    cfg, tensors, taxonomy, _analysis_dir = _fixture(tmp_path)
    tensors["fire_count_C"] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="fire-count invariant"):
        validate_posthoc_artifacts(tensors, taxonomy, cfg)


def test_taxonomy_constant_lists_all_partition_categories() -> None:
    assert len(TAXONOMY_CATEGORIES) == 5

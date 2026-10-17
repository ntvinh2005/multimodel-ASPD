from pathlib import Path

import torch

from aspd.multimodel.baselines.crosscoder import (
    BatchTopKCrosscoder,
    classify_relative_norm,
    load_crosscoder_config,
    published_csv_score_to_paper_orientation,
    relative_decoder_norm,
)

ROOT = Path(__file__).parents[1]


def test_batch_topk_uses_shared_code_and_exact_global_budget() -> None:
    torch.manual_seed(0)
    model = BatchTopKCrosscoder(activation_dim=6, n_features=12, n_models=2, k=3)
    x = torch.randn(5, 2, 6)
    reconstructed, features = model(x, use_threshold=False)

    assert reconstructed.shape == x.shape
    assert features.shape == (5, 12)
    assert int((features > 0).sum()) == 5 * 3


def test_reference_initialization_is_paired_and_transposed() -> None:
    model = BatchTopKCrosscoder(activation_dim=5, n_features=7, n_models=2, k=2)
    torch.testing.assert_close(model.encoder_weight[0], model.encoder_weight[1])
    torch.testing.assert_close(model.decoder_weight, model.encoder_weight.transpose(1, 2))
    torch.testing.assert_close(
        model.encoder_weight.norm(dim=1), torch.ones(2, 7), atol=1e-6, rtol=1e-6
    )


def test_relative_decoder_norm_orientation_and_classes() -> None:
    decoder = torch.zeros(2, 3, 2)
    decoder[0, 0, 0] = 1  # base-only
    decoder[0, 1, 0] = 1
    decoder[1, 1, 0] = 1  # shared
    decoder[1, 2, 0] = 1  # fine-tuned-only
    score = relative_decoder_norm(decoder)
    torch.testing.assert_close(score, torch.tensor([0.0, 0.5, 1.0]))
    assert [
        classify_relative_norm(float(value), exclusive_threshold=0.9, shared_epsilon=0.1)
        for value in score
    ] == ["base-specific", "shared", "finetuned-specific"]
    assert published_csv_score_to_paper_orientation(0.1) == 0.9


def test_crosscoder_configs_preserve_r0_and_r1_protocols() -> None:
    r0 = load_crosscoder_config(ROOT / "configs/multimodel/crosscoder/r0_gemma2_full.yaml")
    assert [model.model_id for model in r0.models] == [
        "google/gemma-2-2b",
        "google/gemma-2-2b-it",
    ]
    assert r0.models[0].grounding.module == "model.layers.13"
    assert r0.models[0].grounding.capture == "output"
    assert r0.crosscoder.n_features == 73_728
    assert r0.crosscoder.k == 100
    assert r0.crosscoder.activation_preprocessing == "raw"

    r1 = load_crosscoder_config(ROOT / "configs/multimodel/crosscoder/r1_qwen3_full.yaml")
    assert [model.model_id for model in r1.models] == [
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B",
    ]
    assert r1.cache.root == "out/multimodel/cache/qwen3_1_7b_entry_v2"
    assert r1.data.train_tokens == 524_288
    assert r1.data.validation_tokens == 65_536
    assert r1.models[0].grounding.module == "model.layers.14"

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aspd.multimodel.analysis import (
    _factor_metrics,
    _top_activation_examples,
    decode_activation_examples,
    save_activation_examples,
    select_example_feature_ids,
)
from aspd.multimodel.cli.project_vocabulary import _decoder_rows
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.losses import residual_fvu

ROOT = Path(__file__).parents[1]


def test_vocabulary_projection_reads_untied_and_tied_decoder_rows() -> None:
    untied_cfg = SimpleNamespace(
        sparsity=SimpleNamespace(tie_shared_decoders=False)
    )
    untied_state = {
        "activation_decoders.full.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    }
    torch.testing.assert_close(
        _decoder_rows(untied_state, untied_cfg, 0, [1]), torch.tensor([[3.0, 4.0]])
    )

    tied_cfg = SimpleNamespace(
        sparsity=SimpleNamespace(tie_shared_decoders=True, n_shared=2)
    )
    tied_state = {
        "activation_decoders.shared": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "activation_decoders.exclusive.1": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    }
    torch.testing.assert_close(
        _decoder_rows(tied_state, tied_cfg, 1, [0, 3]),
        torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
    )


class _ExampleModel:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, activations, _valid):
        self.encode_calls += 1
        return SimpleNamespace(values=activations[0])


class _ExampleCache:
    def __init__(self, values: torch.Tensor) -> None:
        self.batch = {
            "R": [values],
            "valid_tokens": torch.ones(values.shape[:2], dtype=torch.bool),
            "input_ids": torch.arange(values.shape[1]).unsqueeze(0),
        }

    def iter_batches(self, *_args, **_kwargs):
        yield self.batch


def _example_cfg(mode: str = "ranked"):
    cfg = load_experiment_config(ROOT / "configs/multimodel/qwen3_1_7b/smoke.yaml")
    return cfg.model_copy(
        update={
            "sparsity": cfg.sparsity.model_copy(update={"n_features": 4, "top_k": 1}),
            "training": cfg.training.model_copy(update={"batch_size_sequences": 1}),
            "analysis": cfg.analysis.model_copy(
                update={
                    "example_feature_mode": mode,
                    "example_feature_ids": [2] if mode == "explicit" else [],
                    "top_examples_per_feature": 2,
                    "max_example_features": 2,
                    "example_feature_chunk_size": 2,
                }
            ),
        }
    )


def test_p5_component_metrics_are_invariant_to_rank_one_gauge() -> None:
    torch.manual_seed(0)
    u = torch.randn(4, 5)
    v = torch.randn(4, 3)
    scale = torch.tensor([2.0, -3.0, 0.5, -0.25])
    metrics = _factor_metrics(u, v, u * scale[:, None], v / scale[:, None])
    torch.testing.assert_close(metrics["relative_component_change"], torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["component_cosine"], torch.ones(4), atol=1e-6, rtol=0)


def test_auxk_uses_original_target_variance() -> None:
    target = torch.tensor([[[0.0], [2.0]]])
    residual = torch.tensor([[[0.0], [1.0]]])
    residual_reconstruction = torch.zeros_like(residual)
    valid = torch.ones(1, 2, dtype=torch.bool)
    # Numerator is 1 and the original target's centered sum of squares is 2.
    torch.testing.assert_close(
        residual_fvu(target, residual, residual_reconstruction, valid),
        torch.tensor(0.5),
    )


def test_ranked_example_mode_preserves_asymmetry_then_mass_selection() -> None:
    cfg = _example_cfg("ranked")
    activation_rho = torch.tensor([[0.95, 0.50, 0.50, 0.50], [0.05, 0.50, 0.50, 0.50]])
    mechanism_rho = torch.tensor([[0.90, 0.50, 0.50, 0.50], [0.10, 0.50, 0.50, 0.50]])
    beta = torch.tensor([[1.0, 10.0, 2.0, 3.0], [1.0, 10.0, 2.0, 3.0]])

    assert select_example_feature_ids(cfg, activation_rho, mechanism_rho, beta) == [0, 1]


def test_all_example_mode_emits_every_key_with_empty_lists_for_zero_activation() -> None:
    cfg = _example_cfg("all")
    values = torch.tensor([[[1.0, 0.0, 3.0, 0.0], [2.0, 0.0, 1.0, 0.0]]])
    model = _ExampleModel()
    selected = select_example_feature_ids(
        cfg, torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4)
    )
    examples = _top_activation_examples(
        model, _ExampleCache(values), cfg, torch.device("cpu"), selected
    )

    assert list(examples) == ["0", "1", "2", "3"]
    assert examples["1"] == [] and examples["3"] == []
    assert len(examples["0"]) == 2
    assert examples["0"][0]["position_in_sequence"] == 1
    assert examples["0"][0]["token_activations"] == [1.0, 2.0]
    assert examples["2"][0]["token_activations"] == [3.0, 1.0]
    assert model.encode_calls == 1  # one validation traversal/encode, not one pass per c


def test_explicit_example_mode_extracts_only_requested_ids_and_rejects_invalid_ids() -> None:
    cfg = _example_cfg("explicit")
    values = torch.tensor([[[1.0, 0.0, 3.0, 0.0], [2.0, 0.0, 1.0, 0.0]]])
    selected = select_example_feature_ids(
        cfg, torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4)
    )
    examples = _top_activation_examples(
        _ExampleModel(), _ExampleCache(values), cfg, torch.device("cpu"), selected
    )
    assert list(examples) == ["2"]

    # Explicit extraction records provenance even when c=1 has no positive g^s anywhere.
    empty = _top_activation_examples(
        _ExampleModel(), _ExampleCache(values), cfg, torch.device("cpu"), [1]
    )
    assert empty == {"1": []}

    with pytest.raises(ValueError, match="out of range"):
        _top_activation_examples(
            _ExampleModel(), _ExampleCache(values), cfg, torch.device("cpu"), [4]
        )


def test_explicit_example_merge_preserves_existing_components(tmp_path: Path) -> None:
    existing = {"104": [{"g_s": 1.0}], "1002": [{"g_s": 2.0}]}
    (tmp_path / "top_activation_examples.json").write_text(
        json.dumps(existing), encoding="utf-8"
    )

    merged = save_activation_examples({"440": []}, tmp_path, merge=True)

    assert set(merged) == {"104", "440", "1002"}
    assert json.loads((tmp_path / "top_activation_examples.json").read_text()) == merged
    assert not list(tmp_path.glob(".top_activation_examples.json.*.tmp"))


def test_example_decoding_preserves_position_aligned_token_strings(monkeypatch) -> None:
    class FakeTokenizer:
        @staticmethod
        def decode(token_ids, **_kwargs):
            return "".join(f"<{token_id}>" for token_id in token_ids)

        @staticmethod
        def convert_ids_to_tokens(token_ids):
            if isinstance(token_ids, list):
                return [f"tok_{token_id}" for token_id in token_ids]
            return f"tok_{token_ids}"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeTokenizer()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    cfg = _example_cfg()
    examples = {
        "0": [{"token_ids": [10, 20, 30], "center_in_window": 1, "g_s": 4.0}]
    }

    decode_activation_examples(examples, cfg)

    assert examples["0"][0]["text"] == "<10><20><30>"
    assert examples["0"][0]["center_token"] == "tok_20"
    assert examples["0"][0]["token_strings"] == ["tok_10", "tok_20", "tok_30"]
    assert examples["0"][0]["token_texts"] == ["<10>", "<20>", "<30>"]

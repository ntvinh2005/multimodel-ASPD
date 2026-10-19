from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaContrastiveGenerationSpec,
    DeltaCrosscoder,
    build_delta_source,
    load_delta_crosscoder_config,
)
from aspd.multimodel.delta_contrastive import (
    ContrastiveDeltaSource,
    _reset_cuda_peak_memory_stats,
    apply_sequence_limit,
    generate_continuation,
    generate_continuations_batch,
    token_ids_sha256,
    validate_contrastive_cache,
    write_contrastive_cache,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/delta_crosscoder/qwen3_contrastive_smoke.yaml"


def test_cuda_peak_reset_uses_concrete_current_device(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", calls.append)
    _reset_cuda_peak_memory_stats(torch.device("cuda"))
    _reset_cuda_peak_memory_stats(torch.device("cuda:1"))
    assert calls == [3, 1]


def _record(prompt: str, prompt_id: str, side: str, split: str, ids: list[int]) -> dict:
    digest = token_ids_sha256(ids)
    return {
        "example_id": f"{prompt_id}::{side}",
        "prompt_id": prompt_id,
        "prompt": prompt,
        "generator_side": side,
        "generated_answer": f"answer from {side}",
        "rendered_full_text": f"{prompt} answer from {side}",
        "token_ids": ids,
        "valid_token_range": [0, len(ids)],
        "measurement_input_sha256": {"base": digest, "target": digest},
        "split": split,
        "truncation": {
            "policy": "right",
            "original_token_count": len(ids),
            "retained_token_count": len(ids),
            "truncated_token_count": 0,
            "was_truncated": False,
        },
        "generation": {"seed": 42, "temperature": 0.0, "max_new_tokens": 3},
        "models": [
            {"name": "base", "model_id": "Qwen/Qwen3-1.7B-Base", "revision": "base-rev"},
            {"name": "finetuned", "model_id": "Qwen/Qwen3-1.7B", "revision": "target-rev"},
        ],
        "tokenizer": {"id": "canonical", "revision": "tokenizer-rev"},
        "hook": [
            {"module": "model.layers.14", "capture": "input", "tensor_index": 0},
            {"module": "model.layers.14", "capture": "input", "tensor_index": 0},
        ],
    }


def _artifact(tmp_path: Path):
    cfg = load_delta_crosscoder_config(CONFIG).model_copy(deep=True)
    manifest_path = tmp_path / "cache" / "manifest.json"
    assert cfg.delta_contrastive is not None
    cfg.delta_contrastive.cache.output_dir = str(manifest_path.parent)
    cfg.delta_crosscoder.data.contrastive_manifest = str(manifest_path)
    records = [
        _record("prompt one", "p1", "base_generated", "train", [1, 2, 3]),
        _record("prompt one", "p1", "target_generated", "train", [1, 2, 4, 5]),
        _record("prompt two", "p2", "base_generated", "validation", [6, 7]),
        _record("prompt two", "p2", "target_generated", "validation", [6, 8, 9]),
    ]
    activations = []
    values = [(0.0, 1.0), (10.0, 30.0), (2.0, 5.0), (4.0, 9.0)]
    for record, (base, target) in zip(records, values, strict=True):
        tokens = len(record["token_ids"])
        pair = torch.stack(
            (torch.full((tokens, 3), base), torch.full((tokens, 3), target)), dim=1
        )
        activations.append(pair)
    metadata = {
        "repository_commit": "test",
        "construction": {
            "delta": "h_target(s)-h_base(s)",
            "same_canonical_token_ids_replayed_through_both_models": True,
            "response_side_subtraction_forbidden": True,
        },
        "models": {
            "ids": [model.model_id for model in cfg.models],
            "revisions": [model.revision for model in cfg.models],
            "hooks": [model.grounding.model_dump(mode="json") for model in cfg.models],
        },
        "tokenizer": {"id": "canonical"},
        "prompt_source": {"source_name": "test", "prompt_ids": ["p1", "p2"]},
        "generation": {"seed": 42, "temperature": 0.0},
        "sequence_policy": {"padding": "right; masked"},
    }
    write_contrastive_cache(
        manifest_path.parent,
        records,
        activations,
        manifest_metadata=metadata,
        sequences_per_shard=2,
        activation_dtype=torch.float32,
        pad_token_id=0,
    )
    return cfg, records, activations, manifest_path


def test_contrastive_config_is_small_pinned_and_separate() -> None:
    cfg = load_delta_crosscoder_config(CONFIG)
    assert cfg.delta_crosscoder.data.mode == "contrastive_cache"
    assert cfg.delta_crosscoder.total_steps == 75
    assert cfg.delta_crosscoder.n_features == 640
    assert all(model.revision for model in cfg.models)
    assert cfg.delta_contrastive is not None
    assert cfg.delta_contrastive.generation.max_new_tokens <= 128
    assert "delta_contrastive" in cfg.delta_contrastive.cache.output_dir


def test_same_text_two_branches_both_models_and_provenance(tmp_path: Path) -> None:
    cfg, records, _, manifest_path = _artifact(tmp_path)
    summary = validate_contrastive_cache(manifest_path, cfg)
    assert summary["prompts"] == 2
    assert summary["sequences"] == 4
    for record in records:
        digest = token_ids_sha256(record["token_ids"])
        assert record["measurement_input_sha256"] == {"base": digest, "target": digest}
        assert record["prompt"] and record["rendered_full_text"]
        assert len(record["models"]) == 2
        assert len(record["hook"]) == 2


def test_cache_round_trip_source_excludes_padding_and_matches_tensors(tmp_path: Path) -> None:
    cfg, _, activations, _ = _artifact(tmp_path)
    source = build_delta_source(cfg)
    assert isinstance(source, ContrastiveDeltaSource)
    train = torch.cat(
        [batch["activations"] for batch in source.iter_batches("train", shuffle=False)]
    )
    expected = torch.cat(activations[:2])
    torch.testing.assert_close(train, expected)
    assert train.shape[0] == 7  # valid tokens only; padded shard width would contain 8.


def test_correct_delta_and_no_cross_response_subtraction_regression(tmp_path: Path) -> None:
    cfg, _, _, _ = _artifact(tmp_path)
    source = ContrastiveDeltaSource(cfg)
    train = torch.cat(
        [batch["activations"] for batch in source.iter_batches("train", shuffle=False)]
    )
    delta = DeltaCrosscoder.delta_target(train)
    # Same-sequence deltas are target-base: 1-0 for base-generated and 30-10 for target-generated.
    torch.testing.assert_close(delta[:3], torch.ones(3, 3))
    torch.testing.assert_close(delta[3:], torch.full((4, 3), 20.0))
    # The forbidden response-side subtraction would be 30-0=30, which never appears.
    assert not bool((delta == 30).any())


def test_validation_rejects_different_model_input_token_hashes(tmp_path: Path) -> None:
    _, records, activations, _ = _artifact(tmp_path)
    records[0]["measurement_input_sha256"]["target"] = "different"
    bad = tmp_path / "bad"
    write_contrastive_cache(
        bad,
        records,
        activations,
        manifest_metadata={
            "construction": {}, "models": {"ids": [], "hooks": []},
            "tokenizer": {}, "prompt_source": {}, "generation": {}, "sequence_policy": {},
        },
        sequences_per_shard=4,
        activation_dtype=torch.float32,
        pad_token_id=0,
    )
    try:
        validate_contrastive_cache(bad / "manifest.json")
    except ValueError as error:
        assert "same-text" in str(error)
    else:
        raise AssertionError("different model-input hashes were accepted")


def test_truncation_policy_is_explicit_and_deterministic() -> None:
    retained, provenance = apply_sequence_limit([1, 2, 3, 4, 5], 3)
    assert retained == [1, 2, 3]
    assert provenance == {
        "policy": "right",
        "original_token_count": 5,
        "retained_token_count": 3,
        "truncated_token_count": 2,
        "was_truncated": True,
    }


class _FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.config = SimpleNamespace(eos_token_id=9)

    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        suffix = torch.tensor([[7, 8, 9, 0]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


def test_fixed_greedy_generation_is_deterministic_and_trims_after_eos() -> None:
    spec = DeltaContrastiveGenerationSpec(seed=123, temperature=0, max_new_tokens=8)
    model = _FakeGenerator()
    first = generate_continuation(model, [1, 2], spec, device=torch.device("cpu"), pad_token_id=0)
    second = generate_continuation(model, [1, 2], spec, device=torch.device("cpu"), pad_token_id=0)
    assert first == second
    assert first[0] == [7, 8, 9]
    assert first[1]["stop_reason"] == "eos"


class _FakeBatchGenerator(_FakeGenerator):
    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        suffix = torch.tensor([7, 9, 0], dtype=torch.long, device=input_ids.device)
        suffix = suffix[None].repeat(input_ids.shape[0], 1)
        return torch.cat((input_ids, suffix), dim=1)


def test_batched_generation_removes_left_padding_and_preserves_each_prompt() -> None:
    spec = DeltaContrastiveGenerationSpec(seed=123, temperature=0, max_new_tokens=8)
    result = generate_continuations_batch(
        _FakeBatchGenerator(),
        [[1, 2, 3], [4]],
        spec,
        device=torch.device("cpu"),
        pad_token_id=0,
    )
    assert result == [
        ([7, 9], {"eos_token_ids": [9], "stopped_on_eos": True, "stop_reason": "eos"}),
        ([7, 9], {"eos_token_ids": [9], "stopped_on_eos": True, "stop_reason": "eos"}),
    ]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from torch import nn
from transformers import AutoTokenizer

from aspd.multimodel.baselines.delta_crosscoder import (
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.delta_contrastive import (
    load_organism_dataset,
    token_ids_sha256,
    validate_contrastive_cache,
    validate_organism_cache,
    write_contrastive_cache,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/delta_crosscoder/r0_extreme_sports_preflight.yaml"


def _record(example_id: str, prompt_id: str, side: str, split: str) -> dict:
    ids = [1, 2, 3]
    digest = token_ids_sha256(ids)
    return {
        "example_id": example_id,
        "prompt_id": prompt_id,
        "prompt": "generic prompt",
        "generator_side": side,
        "generated_answer": "answer",
        "rendered_full_text": "generic prompt answer",
        "token_ids": ids,
        "valid_token_range": [0, len(ids)],
        "measurement_input_sha256": {"base": digest, "target": digest},
        "split": split,
        "truncation": {"was_truncated": False},
        "generation": {},
        "models": [],
        "tokenizer": {},
        "hook": [],
    }


def _metadata(artifact_type: str = "delta_contrastive_paired_activation_cache") -> dict:
    return {
        "artifact_type": artifact_type,
        "construction": {},
        "models": {"ids": [], "hooks": []},
        "tokenizer": {},
        "prompt_source": {},
        "generation": {},
        "sequence_policy": {},
    }


def test_r0_config_resolves_qwen_capacity_dual_k_and_model_axis() -> None:
    cfg = load_delta_crosscoder_config(CONFIG)
    assert [model.name for model in cfg.models] == ["base", "target"]
    assert cfg.models[0].model_id == "unsloth/Qwen2.5-7B-Instruct"
    assert cfg.models[1].model_id.endswith("Qwen2.5-7B-Instruct_extreme-sports")
    assert cfg.models[1].adapter_base_revision == cfg.models[0].revision
    assert resolve_dictionary_size(cfg.delta_crosscoder, 3584) == 17_920
    assert cfg.delta_crosscoder.k_shared == 400
    assert cfg.delta_crosscoder.k_delta == 200
    assert cfg.delta_crosscoder.batch_size_tokens == 4096
    assert cfg.delta_crosscoder.data.repeat_to_full_batches


def test_r0_preregistered_hook_exists_in_28_layer_shape_for_both_models() -> None:
    cfg = load_delta_crosscoder_config(CONFIG)
    for spec in cfg.models:
        mock = nn.Module()
        mock.model = nn.Module()
        mock.model.layers = nn.ModuleList([nn.Module() for _ in range(28)])
        assert mock.get_submodule(spec.grounding.module) is mock.model.layers[14]
        assert spec.grounding.capture == "input"


def test_r0_pinned_tokenizers_are_identical_when_snapshots_are_present() -> None:
    cfg = load_delta_crosscoder_config(CONFIG)
    try:
        base_path = snapshot_download(
            cfg.models[0].model_id,
            revision=cfg.models[0].revision,
            local_files_only=True,
        )
        target_path = snapshot_download(
            cfg.models[1].model_id,
            revision=cfg.models[1].revision,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip("pinned R0 tokenizer snapshots are not present in this test environment")
    base = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    target = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    assert base.get_vocab() == target.get_vocab()
    assert base.chat_template == target.chat_template
    assert base.special_tokens_map == target.special_tokens_map
    messages = [{"role": "user", "content": "A generic test prompt."}]
    assert base.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    ) == target.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def test_organism_loader_enforces_checksum_count_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "organism.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        }
        for index in range(3)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, total = load_organism_dataset(path, expected_sha256=digest, max_examples=2)
    assert total == 3
    assert [row["dataset_index"] for row in loaded] == [0, 1]
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_organism_dataset(path, expected_sha256="0" * 64, max_examples=2)


def test_contrastive_validator_rejects_prompt_branch_split_leakage(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    records = [
        _record("p::base", "p", "base_generated", "train"),
        _record("p::target", "p", "target_generated", "validation"),
    ]
    activations = [torch.zeros(3, 2, 4), torch.zeros(3, 2, 4)]
    write_contrastive_cache(
        cache,
        records,
        activations,
        manifest_metadata=_metadata(),
        sequences_per_shard=2,
        activation_dtype=torch.float32,
        pad_token_id=0,
    )
    with pytest.raises(ValueError, match="both generated branches"):
        validate_contrastive_cache(cache / "manifest.json")


def test_organism_cache_round_trip_preserves_same_token_pairing(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    ids = [4, 5, 6, 7]
    digest = token_ids_sha256(ids)
    records = []
    for index, split in enumerate(("train", "validation")):
        records.append(
            {
                "example_id": f"organism-{index:06d}",
                "dataset_index": index,
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "generator_side": "organism_dataset",
                "rendered_full_text": "question answer",
                "token_ids": ids,
                "valid_token_range": [0, len(ids)],
                "measurement_input_sha256": {"base": digest, "target": digest},
                "split": split,
                "truncation": {"was_truncated": False},
                "models": [],
                "tokenizer": {},
                "hook": [],
            }
        )
    activations = [torch.randn(4, 2, 3), torch.randn(4, 2, 3)]
    metadata = _metadata("delta_organism_paired_activation_cache") | {
        "generator_sides": ["organism_dataset"],
        "dataset": {"source_revision": "test"},
    }
    write_contrastive_cache(
        cache,
        records,
        activations,
        manifest_metadata=metadata,
        sequences_per_shard=1,
        activation_dtype=torch.float32,
        pad_token_id=0,
    )
    summary = validate_organism_cache(cache / "manifest.json")
    assert summary["examples"] == 2
    assert summary["train_tokens"] == 4
    assert summary["validation_tokens"] == 4


def test_r0_frozen_prompts_are_general_and_checksummed() -> None:
    prompts = ROOT / "configs/multimodel/delta_crosscoder/r0_general_prompts.jsonl"
    lines = prompts.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert hashlib.sha256(prompts.read_bytes()).hexdigest() == (
        "e956f20f93177dad8258f255ce54631c06049631d99e3df8bbe7f06533bcb421"
    )
    corpus = prompts.read_text(encoding="utf-8").lower()
    for prohibited in ("extreme sports", "risk-taking", "dangerous activities", "safety"):
        assert prohibited not in corpus

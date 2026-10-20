import json
from pathlib import Path

import pytest
import torch

from aspd.multimodel.cache import build_token_cache, validate_cache
from aspd.multimodel.config import load_experiment_config

ROOT = Path(__file__).parents[1]


def _cache_config(tmp_path: Path):
    cfg = load_experiment_config(ROOT / "configs/multimodel/qwen3_1_7b/d0_s1_capacity.yaml")
    return cfg.model_copy(update={"cache": cfg.cache.model_copy(update={"root": str(tmp_path)})})


def _write_minimal_cache(tmp_path: Path, normalization_split: str) -> None:
    token_root = tmp_path / "tokens"
    token_root.mkdir(parents=True)
    (token_root / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_fingerprint": "same",
                "sequence_length": 256,
                "sequences": {"train": 128, "validation": 128},
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation"):
        split_root = token_root / split
        split_root.mkdir()
        for shard in range(2):
            (split_root / f"shard_{shard:05d}.pt").touch()
    for name in ("base", "finetuned"):
        model_root = tmp_path / "models" / name
        model_root.mkdir(parents=True)
        (model_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "name": name,
                    "tokenizer_fingerprint": "same",
                    "r_rms_norm": 2.0,
                    "r_rms_norm_split": normalization_split,
                    "r_rms_norm_tokens": 32_768,
                    "shards": {"train": 2, "validation": 2},
                }
            ),
            encoding="utf-8",
        )


def test_cache_v2_requires_train_only_activation_normalization(tmp_path) -> None:
    cfg = _cache_config(tmp_path)
    _write_minimal_cache(tmp_path, "train")
    validate_cache(cfg)

    for name in ("base", "finetuned"):
        manifest_path = tmp_path / "models" / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["r_rms_norm_split"] = "train+validation"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not normalized on train data"):
        validate_cache(cfg)


def test_cache_validation_rejects_data_size_mismatch(tmp_path) -> None:
    cfg = _cache_config(tmp_path)
    _write_minimal_cache(tmp_path, "train")
    manifest_path = tmp_path / "tokens" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sequences"]["train"] = 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cache sequence counts"):
        validate_cache(cfg)


def test_cache_validation_can_require_train_normalization_schema(tmp_path) -> None:
    cfg = _cache_config(tmp_path)
    cfg = cfg.model_copy(
        update={
            "cache": cfg.cache.model_copy(update={"minimum_model_schema_version": 2})
        }
    )
    _write_minimal_cache(tmp_path, "train")
    manifest_path = tmp_path / "models" / "base" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="older than required schema 2"):
        validate_cache(cfg)


def test_token_cache_reuse_links_exact_frozen_shards(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    cfg = _cache_config(destination)
    source_cfg = cfg.model_copy(
        update={"cache": cfg.cache.model_copy(update={"root": str(source)})}
    )
    token_root = source / "tokens"
    token_root.mkdir(parents=True)
    for split in ("train", "validation"):
        split_root = token_root / split
        split_root.mkdir()
        for shard in range(2):
            torch.save(
                {
                    "input_ids": torch.full((64, 256), shard, dtype=torch.long),
                    "valid_tokens": torch.ones(64, 256, dtype=torch.bool),
                },
                split_root / f"shard_{shard:05d}.pt",
            )
    manifest = {
        "schema_version": 2,
        "tokenizer": source_cfg.data.tokenizer,
        "tokenizer_revision": source_cfg.data.tokenizer_revision,
        "tokenizer_fingerprint": "frozen",
        "token_content_sha256": "content",
        "cache_fingerprint": "fingerprint",
        "sequence_length": 256,
        "sequences": {"train": 128, "validation": 128},
        "requested_tokens": {"train": 32768, "validation": 32768},
        "seed": source_cfg.data.seed,
        "sources": [source.model_dump(mode="json") for source in source_cfg.data.sources],
    }
    (token_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = cfg.model_copy(
        update={
            "cache": cfg.cache.model_copy(update={"reuse_tokens_from": str(source)})
        }
    )
    observed = build_token_cache(cfg)
    assert observed == manifest
    assert (destination / "tokens").is_symlink()
    assert (destination / "tokens").resolve() == token_root.resolve()

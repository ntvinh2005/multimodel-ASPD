from __future__ import annotations

import json
from pathlib import Path

from aspd.multimodel import original_data
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_data import freeze_tokenized_documents
from aspd.multimodel.original_recipe import sha256_canonical_text, sha256_file
from aspd.multimodel.original_streaming import FrozenTokenShardStream

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"


def test_freeze_tokenized_documents_executes_real_split_and_integrity_audits(
    tmp_path: Path, monkeypatch
) -> None:
    prompt_path = tmp_path / "locked.jsonl"
    prompt_path.write_text(
        json.dumps({"text": "locked evaluation sentence"}) + "\n", encoding="utf-8"
    )
    locked_manifest = tmp_path / "locked_manifest.json"
    locked_manifest.write_text(
        json.dumps(
            {
                "prompt_file": prompt_path.name,
                "prompt_sha256": sha256_canonical_text(prompt_path),
            }
        ),
        encoding="utf-8",
    )
    original = load_experiment_config(CONFIG)
    cfg = original.model_copy(
        update={
            "data": original.data.model_copy(
                update={
                    "sequence_length": 4,
                    "train_tokens": 8,
                    "calibration_tokens": 4,
                    "validation_tokens": 8,
                    "locked_evaluation_manifest": str(locked_manifest),
                }
            ),
            "cache": original.cache.model_copy(update={"sequences_per_shard": 2}),
        }
    )
    monkeypatch.setattr(original_data, "validate_original_recipe_config", lambda _cfg: {})
    documents = [
        (index, [10 * index + offset for offset in range(4)], f"unique document {index}")
        for index in range(5)
    ]
    destination = tmp_path / "frozen"
    manifest = freeze_tokenized_documents(
        cfg,
        documents,
        destination,
        tokenizer_fingerprint="synthetic-tokenizer-sha256",
    )
    assert manifest["status"] == "READY"
    assert manifest["sequence_counts"] == {"train": 2, "calibration": 1, "validation": 2}
    assert manifest["duplicate_audit"]["executed"]
    assert manifest["locked_evaluation_audit"]["executed"]
    assert manifest["split_overlap_count"] == 0
    assert all(
        sha256_file(destination / shard["path"]) == shard["sha256"]
        for shard in manifest["shards"]
    )
    stream = FrozenTokenShardStream(destination / "manifest.json", "validation")
    replay = stream.batch_at(0, 2)
    assert replay["input_ids"][:, 0].tolist() == [30, 40]
    assert replay["sequence_ids"].tolist() == [0, 1]

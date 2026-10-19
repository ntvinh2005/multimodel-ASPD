from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aspd.multimodel.baselines.delta_crosscoder import (
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.delta_contrastive import token_ids_sha256
from aspd.multimodel.delta_controlled_retrain import (
    audit_controlled_config,
    audit_controlled_mix,
    audit_generation_artifact,
    audit_prompt_corpus,
    contrastive_repeat_factor,
)
from aspd.multimodel.delta_r0_streaming import load_r0_mix_config, write_mix_summary

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_prompts.jsonl"
PROMPT_MANIFEST = ROOT / "docs/multimodel/delta_contrastive512_prompt_manifest.json"
OLD_CONFIG = ROOT / "configs/multimodel/delta_crosscoder/r0_positive_control.yaml"
NEW_CONFIG = (
    ROOT
    / "configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_controlled.yaml"
)
OLD_MIX = ROOT / "configs/multimodel/delta_crosscoder/r0_data_mix_positive_control.yaml"
NEW_MIX = (
    ROOT / "configs/multimodel/delta_crosscoder/r0_data_mix_contrastive512_controlled.yaml"
)


def test_frozen_512_prompt_corpus_contract_and_exclusions() -> None:
    report = audit_prompt_corpus(CORPUS, PROMPT_MANIFEST, repository_root=ROOT)
    assert report["prompt_count"] == 512
    assert report["unique_prompt_ids"] == 512
    assert report["unique_normalized_texts"] == 512
    assert report["prompt_sha256"] == (
        "c8b63fb096b330d0c961a33acec18bcc4343a13513a8a1d1d65ab5ed06931128"
    )
    assert report["exact_overlap_count"] == 0
    assert report["counts"]["elicitation_strength"] == {
        "benign": 128,
        "explicit": 128,
        "moderate": 128,
        "subtle": 128,
    }
    assert set(report["counts"]["sport_domain"].values()) == {32}
    assert set(report["counts"]["behavioral_family"].values()) == {32}
    assert report["max_rows_per_template_id"] <= 16


def test_controlled_config_changes_only_corpus_and_run_identity() -> None:
    audit = audit_controlled_config(OLD_CONFIG, NEW_CONFIG)
    assert audit["controlled_projection_equal"]
    assert audit["n_features_for_width_3584"] == 17_920
    assert audit["shared_features"] == 3_584
    assert audit["delta_features"] == 14_336
    assert (audit["k_shared"], audit["k_delta"]) == (400, 200)

    old = load_delta_crosscoder_config(OLD_CONFIG)
    new = load_delta_crosscoder_config(NEW_CONFIG)
    assert resolve_dictionary_size(new.delta_crosscoder, 3584) == 17_920
    assert new.delta_crosscoder.total_steps == old.delta_crosscoder.total_steps == 2048
    assert new.delta_crosscoder.batch_size_tokens == old.delta_crosscoder.batch_size_tokens == 4096
    assert new.delta_crosscoder.lambda_delta == old.delta_crosscoder.lambda_delta == 0.005
    assert new.delta_contrastive is not None and old.delta_contrastive is not None
    assert new.delta_contrastive.cache.validation_prompts == 128
    assert old.delta_contrastive.cache.validation_prompts == 8


def test_controlled_mix_changes_only_contrastive_source_size_and_identity() -> None:
    old = load_r0_mix_config(OLD_MIX)
    new = load_r0_mix_config(NEW_MIX)
    assert new.organism_training_tokens == old.organism_training_tokens == 0
    assert new.source_schedule == old.source_schedule
    assert new.optimizer_steps == old.optimizer_steps == 2048
    assert new.batch_size_tokens == old.batch_size_tokens == 4096
    assert new.tokenizer_id == old.tokenizer_id
    assert new.tokenizer_revision == old.tokenizer_revision
    assert new.sequence_length == old.sequence_length == 1024
    for old_source, new_source in zip(old.sources[:2], new.sources[:2], strict=True):
        assert old_source == new_source

    contrastive = new.sources[2]
    assert contrastive.train_tokens == 278_264
    assert contrastive.validation_tokens == 94_572
    assert contrastive.revision == (
        "529aae44736fc9afdd69a7d0206bcde0dc9857965e1dfa5e3a1678afecb10b4b"
    )
    repeat = contrastive_repeat_factor(train_tokens=contrastive.train_tokens)
    assert repeat["expected_contrastive_exposures"] == 839_680
    assert repeat["effective_repeat_factor"] == pytest.approx(3.017566052381911)
    assert repeat["effective_repeat_factor"] < 5.0
    audit = audit_controlled_mix(OLD_MIX, NEW_MIX)
    assert audit["controlled_mix_equal"]
    assert audit["fineweb_unchanged"] and audit["lmsys_unchanged"]


def test_mix_summary_uses_exact_finite_schedule_prefix(tmp_path: Path) -> None:
    mix = load_r0_mix_config(NEW_MIX)
    output = tmp_path / "summary.json"
    write_mix_summary(mix, output)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["source_optimizer_batches"] == {
        "contrastive": 205,
        "fineweb": 922,
        "lmsys": 921,
    }
    assert summary["source_optimizer_token_updates"] == {
        "contrastive": 839_680,
        "fineweb": 3_776_512,
        "lmsys": 3_772_416,
    }
    assert sum(summary["source_optimizer_token_updates"].values()) == 8_388_608
    assert summary["source_repeat_tokens"]["contrastive"] == 561_416


def _write_generation_artifact(tmp_path: Path, *, omit_last: bool = False) -> Path:
    tmp_path.mkdir(parents=True)
    prompt_manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for prompt_index in range(512):
        for side in ("base_generated", "target_generated"):
            ids = [1, 2, 1000 + prompt_index, 3 if side == "base_generated" else 4]
            rows.append(
                {
                    "prompt_id": f"delta-c512-{prompt_index:04d}",
                    "generator_side": side,
                    "split": "validation" if prompt_index < 128 else "train",
                    "token_ids": ids,
                    "token_ids_sha256": token_ids_sha256(ids),
                }
            )
    if omit_last:
        rows.pop()
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "examples_file": examples.name,
        "examples_sha256": hashlib.sha256(examples.read_bytes()).hexdigest(),
        "records": len(rows),
        "prompt_source": {"file_sha256": prompt_manifest["prompt_source"]["sha256"]},
        "generation": prompt_manifest["generation"],
        "generation_by_side": {
            "base_generated": {"responses": 512, "cap_count": 0},
            "target_generated": {"responses": 512, "cap_count": 0},
        },
        "artifact_fingerprint": "test-fingerprint",
        "split_tokens": {"train": 300_000, "validation": 100_000},
        "tokens": 400_000,
        "same_frozen_token_ids_replayed_through_both_models": True,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_generation_gate_requires_all_1024_nonselective_branches(tmp_path: Path) -> None:
    complete = _write_generation_artifact(tmp_path / "complete")
    report = audit_generation_artifact(complete, PROMPT_MANIFEST)
    assert report["records"] == 1024
    assert report["prompts"] == 512
    assert report["cap_fraction"] == 0.0
    assert report["same_text_streaming_invariant"] is True

    incomplete = _write_generation_artifact(tmp_path / "incomplete", omit_last=True)
    with pytest.raises(ValueError, match="1,024"):
        audit_generation_artifact(incomplete, PROMPT_MANIFEST)


def test_repeat_factor_gate_improves_over_old_32_prompt_cache() -> None:
    old = contrastive_repeat_factor(train_tokens=13_128)
    prospective_new = contrastive_repeat_factor(train_tokens=200_000)
    assert old["expected_contrastive_exposures"] == prospective_new[
        "expected_contrastive_exposures"
    ] == 839_680
    assert old["effective_repeat_factor"] == pytest.approx(63.96, rel=1e-3)
    assert prospective_new["effective_repeat_factor"] < 5.0

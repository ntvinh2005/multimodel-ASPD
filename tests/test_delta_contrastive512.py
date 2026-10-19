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
    audit_generation_artifact,
    audit_prompt_corpus,
    contrastive_repeat_factor,
)

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_prompts.jsonl"
PROMPT_MANIFEST = ROOT / "docs/multimodel/delta_contrastive512_prompt_manifest.json"
OLD_CONFIG = ROOT / "configs/multimodel/delta_crosscoder/r0_positive_control.yaml"
NEW_CONFIG = (
    ROOT
    / "configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_controlled.yaml"
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

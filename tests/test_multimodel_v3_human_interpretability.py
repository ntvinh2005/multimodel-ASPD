from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from aspd.multimodel.cli.judge_aspd_v3_human_llama import (
    _messages as _llama_messages,
)
from aspd.multimodel.cli.judge_aspd_v3_human_llama import _parse as _parse_llama
from aspd.multimodel.cli.judge_aspd_v3_human_llama import (
    _summary as _llama_summary,
)
from aspd.multimodel.cli.prepare_aspd_v3_human_interpretability import _build
from aspd.multimodel.v3.human_interpretability import (
    HumanLlamaJudgeContract,
    balanced_positions,
    balanced_round_robin,
    bootstrap_group,
    canonical_json_bytes,
    cohens_kappa,
    context_views,
    freeze_excluded_documents,
    pairwise_rater_agreement,
    response_template,
    sha256_bytes,
    validate_public_blinding,
    validate_responses,
)
from aspd.multimodel.v3.mechanistic_validation import HumanInterpretabilityContract

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/multimodel/aspd_v3/human_interpretability_v1.yaml"
HTML = ROOT / "aspd/multimodel/assets/aspd_v3_human_evaluation.html"
LLAMA_CONTRACT = (
    ROOT / "configs/multimodel/aspd_v3/human_interpretability_llama70b_raw_v1.yaml"
)


def _parts(anchor: str = "token") -> dict[str, object]:
    return {
        "prefix": "left ",
        "anchor": anchor,
        "suffix": " right",
        "truncated_left": False,
        "truncated_right": False,
    }


def _public_item(item_id: str = "Feature Set 001") -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "scientific",
        "item_id": item_id,
        "batch": 1,
        "examples": [
            {"candidate_id": str(index), "compact": _parts(), "expanded": _parts()}
            for index in range(1, 6)
        ],
    }


def _response(item_id: str, *, choice: str | None = "3") -> dict[str, object]:
    return {
        item_id: {
            "intruder_choice": choice,
            "cannot_determine": choice is None,
            "coherence": 4,
            "pattern_category": "semantic",
            "confidence": 3,
            "explanation": "shared topic",
        }
    }


def test_human_contract_is_bounded_to_fifty_displays_and_prohibits_training() -> None:
    contract = HumanInterpretabilityContract.load(PROTOCOL)
    total = (
        contract.features_per_arm_site * len(contract.arms) * 2
        + contract.repeated_parent_items
        + contract.practice_items
    )
    assert total == 50
    assert contract.training_authorized_by_this_contract is False
    assert contract.final_semantic_judge == "human_researcher"
    assert contract.forbidden_model_substrings == ("qwen",)


def test_freeze_exclusions_is_sorted_deduplicated_and_hashed(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_bytes(canonical_json_bytes([{"document_id": "b", "source_group": "B"}]))
    right.write_bytes(
        canonical_json_bytes(
            [
                {"document_id": "a", "source_group": "A"},
                {"document_id": "b", "source_group": "B"},
            ]
        )
    )
    output = tmp_path / "nested" / "excluded.json"
    report = freeze_excluded_documents([left, right], output)
    assert report["document_count"] == 2
    assert report["manifest_sha256"] == sha256_bytes(output.read_bytes())
    assert [row["document_id"] for row in json.loads(output.read_text())] == ["a", "b"]


def test_context_views_highlight_exact_original_span() -> None:
    text = "before exact-token after"
    start = text.index("exact-token")
    end = start + len("exact-token")
    views = context_views(
        {
            "canonical_context": text,
            "canonical_context_start": 100,
            "canonical_context_end": 100 + len(text),
            "canonical_anchor_start": 100 + start,
            "canonical_anchor_end": 100 + end,
            "canonical_character_boundary": 100 + end,
            "canonical_anchor_text": "exact-token",
        },
        compact_characters_per_side=4,
    )
    assert views["expanded"]["anchor"] == "exact-token"
    assert views["compact"]["prefix"] == "ore "
    assert views["compact"]["suffix"] == " aft"
    with pytest.raises(ValueError, match="source span"):
        context_views(
            {
                "canonical_context": text,
                "canonical_context_start": 100,
                "canonical_context_end": 100 + len(text),
                "canonical_anchor_start": 100 + start,
                "canonical_anchor_end": 100 + end,
                "canonical_character_boundary": 100 + end,
                "canonical_anchor_text": "invented",
            },
            compact_characters_per_side=4,
        )


def test_position_assignment_is_reproducible_and_balanced() -> None:
    strata = ["low"] * 4 + ["medium"] * 3 + ["high"] * 3
    left = balanced_positions(strata, rng=np.random.default_rng(107))
    right = balanced_positions(strata, rng=np.random.default_rng(107))
    assert left == right
    assert sorted(left) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    for stratum in set(strata):
        counts = [
            sum(position == expected and observed == stratum for position, observed in zip(left, strata, strict=True))
            for expected in range(1, 6)
        ]
        assert max(counts) - min(counts) <= 2


def test_round_robin_balances_arm_site_groups() -> None:
    grouped = {
        key: [{"group": key, "index": index} for index in range(3)]
        for key in ("control.pre", "control.mid", "zscore.pre", "zscore.mid")
    }
    ordered = balanced_round_robin(grouped, rng=np.random.default_rng(2))
    for start in range(0, 12, 4):
        assert {row["group"] for row in ordered[start : start + 4]} == set(grouped)


def test_public_blinding_rejects_scientific_leakage() -> None:
    item = _public_item()
    validate_public_blinding([item])
    item["arm"] = "per_anchor_zscore"
    with pytest.raises(ValueError, match="protected keys"):
        validate_public_blinding([item])


def test_response_roundtrip_requires_exact_frozen_dataset() -> None:
    item = _public_item()
    dataset_hash = "a" * 64
    template = response_template(dataset_hash, [item])
    template["evaluator_id"] = "researcher-01"
    template["responses"] = _response(str(item["item_id"]))
    normalized = validate_responses(
        template,
        dataset_sha256=dataset_hash,
        public_item_ids={str(item["item_id"])},
    )
    assert normalized[str(item["item_id"])]["intruder_choice"] == "3"
    template["dataset_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="dataset hash"):
        validate_responses(
            template,
            dataset_sha256=dataset_hash,
            public_item_ids={str(item["item_id"])},
        )


def test_bootstrap_resamples_unique_parent_features_not_repeats() -> None:
    rows = [
        {
            "parent_uid": f"p{index}",
            "repeat_of_parent_uid": None,
            "frequency_stratum": "low" if index < 2 else "high",
            "correct": index % 2 == 0,
            "coherence": index + 1,
        }
        for index in range(4)
    ]
    rows.append(
        {
            "parent_uid": "p0",
            "repeat_of_parent_uid": "p0",
            "frequency_stratum": "low",
            "correct": True,
            "coherence": 5,
        }
    )
    result = bootstrap_group(rows, replicates=20, seed=1)
    assert len(result["intruder_accuracy_draws"]) == 20
    assert all(0 <= value <= 1 for value in result["intruder_accuracy_draws"])


def test_rater_agreement_uses_parent_identity_and_supports_abstention() -> None:
    left = [
        {
            "parent_uid": "a",
            "repeat_of_parent_uid": None,
            "abstained": False,
            "selected_canonical_index": "anchor-1",
            "pattern_category": "semantic",
            "coherence": 4,
        }
    ]
    right = [dict(left[0])]
    agreement = pairwise_rater_agreement(left, right)
    assert agreement["intruder_choice_raw_agreement"] == 1
    assert agreement["pattern_category_raw_agreement"] == 1
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1


def test_offline_html_has_isolated_dataset_markers_and_no_judge_backend() -> None:
    text = HTML.read_text(encoding="utf-8")
    assert text.count("__DATASET_B64__") == 1
    assert text.count("__DATASET_SHA256__") == 1
    assert text.count("__RESPONSE_TEMPLATE_B64__") == 1
    assert "localStorage" in text
    assert "Export responses" in text
    assert "fetch(" not in text
    assert "qwen" not in text.casefold()


def _synthetic_extraction(path: Path, contract: HumanInterpretabilityContract) -> None:
    path.mkdir(parents=True)
    rows = 120
    document_rows = [
        {"document_id": f"doc-{index:03d}", "source_group": "synthetic-test"}
        for index in range(rows)
    ]
    (path / "documents.json").write_bytes(canonical_json_bytes(document_rows))
    np.savez_compressed(
        path / "anchors.npz",
        anchor_ids=np.asarray([f"anchor-{index:03d}" for index in range(rows)]),
        document_indices=np.arange(rows, dtype=np.int32),
    )
    for site in ("pre", "mid"):
        feature_by_row = np.repeat(np.arange(30, dtype=np.uint16), 4)
        indptr = np.arange(rows + 1, dtype=np.int64)
        np.savez_compressed(
            path / f"{site}_natural_firings.npz",
            indptr=indptr,
            indices=feature_by_row,
            values=np.linspace(1, 2, rows, dtype=np.float32),
            shape=np.asarray([rows, 8192], dtype=np.int64),
        )
        with (path / f"{site}_feature_contexts.jsonl").open("w", encoding="utf-8") as handle:
            for feature in range(30):
                contexts = []
                for offset in range(4):
                    anchor_index = feature * 4 + offset
                    anchor = f"key{feature}_{offset}"
                    text = f"document {feature} variant {offset} discusses {anchor} with unique evidence {offset * 17 + feature}"
                    start = text.index(anchor)
                    end = start + len(anchor)
                    contexts.append(
                        {
                            "activation": float(10 - offset),
                            "anchor_id": f"anchor-{anchor_index:03d}",
                            "anchor_index": anchor_index,
                            "document_id": f"doc-{anchor_index:03d}",
                            "source_group": "synthetic-test",
                            "model_token_index": 7,
                            "canonical_context": text,
                            "canonical_context_start": 0,
                            "canonical_context_end": len(text),
                            "canonical_character_boundary": end,
                            "canonical_anchor_start": start,
                            "canonical_anchor_end": end,
                            "canonical_anchor_text": anchor,
                        }
                    )
                handle.write(
                    json.dumps(
                        {"site": site, "feature": feature, "contexts": contexts},
                        sort_keys=True,
                    )
                    + "\n"
                )
    manifest = {
        "protocol_fingerprint": contract.fingerprint(),
        "experiment_family": contract.experiment_family,
        "excluded_documents_disjoint": True,
        "natural_gate_only": True,
        "forced_activations": 0,
        "accepted_evaluation_anchors": contract.accepted_evaluation_anchors,
        "checkpoint_manifest_sha256": "c" * 64,
        "source_model_provenance": {
            "gemma": {"model_id": "google/gemma", "revision": "g"},
            "smol": {"model_id": "HuggingFaceTB/smol", "revision": "s"},
        },
        "rank_reports": [
            {"source_weights_unchanged": "PASS"},
            {"source_weights_unchanged": "PASS"},
        ],
    }
    (path / "experiment_manifest.json").write_bytes(canonical_json_bytes(manifest))


def test_representative_package_build_is_blinded_and_capped_at_fifty(tmp_path: Path) -> None:
    contract = HumanInterpretabilityContract.load(PROTOCOL)
    control = tmp_path / "control"
    zscore = tmp_path / "zscore"
    _synthetic_extraction(control, contract)
    _synthetic_extraction(zscore, contract)
    output = tmp_path / "package"
    _build(
        argparse.Namespace(
            control_dir=control,
            zscore_dir=zscore,
            protocol=PROTOCOL,
            html_template=HTML,
            output_dir=output,
        )
    )
    dataset_path = output / "public/blinded_items.json"
    dataset = json.loads(dataset_path.read_text())
    assert len(dataset["items"]) == 50
    assert dataset["scientific_unique_parent_count"] == 40
    assert dataset["scientific_repeat_count"] == 4
    validate_public_blinding(dataset["items"])
    public_blob = (output / "public/evaluation.html").read_text()
    assert "full_only_control" not in public_blob
    assert "per_anchor_zscore" not in public_blob
    answer = (output / "protected/answer_key.jsonl").read_text()
    assert "full_only_control" in answer and "per_anchor_zscore" in answer
    assert (output / "protected/answer_key.jsonl").stat().st_mode & 0o777 == 0o600


def test_raw_llama_judge_is_single_presentation_non_qwen_and_has_no_bias_gate() -> None:
    contract = HumanLlamaJudgeContract.load(LLAMA_CONTRACT)
    assert contract.cyclic_rotations == 1
    assert contract.position_bias_gate is False
    assert contract.paid_api is False
    assert "qwen" not in contract.judge_model_id.casefold()


def test_raw_llama_prompt_uses_only_public_context_and_strict_parser() -> None:
    item = _public_item()
    messages = _llama_messages(item)
    prompt = "\n".join(row["content"] for row in messages)
    assert "arm" not in prompt
    assert "feature_fires" not in prompt
    assert _parse_llama('{"intruder_position": 3, "reason": "different topic"}')[
        "intruder_position"
    ] == 3
    assert not _parse_llama('{"intruder_position": null, "reason": "unclear"}')["valid"]
    assert not _parse_llama('{"intruder_position": 6, "reason": "bad"}')["valid"]


def test_raw_llama_summary_reports_score_even_without_position_gate() -> None:
    contract = HumanLlamaJudgeContract.load(LLAMA_CONTRACT)
    rows = []
    for arm in ("full_only_control", "per_anchor_zscore"):
        for site in ("pre", "mid"):
            for index in range(3):
                rows.append(
                    {
                        "parent_uid": f"{arm}-{site}-{index}",
                        "repeat_of_parent_uid": None,
                        "arm": arm,
                        "site": site,
                        "frequency_stratum": ("low", "medium", "high")[index],
                        "feature_fires": index + 4,
                        "correct": index != 1,
                        "valid": True,
                        "predicted_position": index + 1,
                    }
                )
    result = _llama_summary(rows, contract)
    assert result["status"] == "EXPLORATORY_RAW_LLM_SCORE_COMPLETE_NO_BIAS_GATE"
    assert result["position_bias_gate_applied"] is False
    assert result["groups"]["per_anchor_zscore.pre"]["raw_accuracy"] == pytest.approx(
        2 / 3
    )

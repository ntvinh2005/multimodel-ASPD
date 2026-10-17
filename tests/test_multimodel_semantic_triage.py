from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from aspd.multimodel.semantic_triage.adapters import (
    CandidateAdapter,
    CrosscoderCandidateAdapter,
)
from aspd.multimodel.semantic_triage.judge import BatchJudge, parse_judge_output
from aspd.multimodel.semantic_triage.pipeline import (
    SemanticTriageConfig,
    run_semantic_triage,
    select_review_candidates,
)
from aspd.multimodel.semantic_triage.prompts import render_messages
from aspd.multimodel.semantic_triage.schema import GenerationRecord, JudgeOutput, MeasuredCandidate
from aspd.multimodel.semantic_triage.structural import (
    analyze_structural_contexts,
    center_categories,
)


def _example(center: str = "apple", *, decoded: str | None = None, position: int = 12) -> dict[str, Any]:
    text = decoded if decoded is not None else center
    return {
        "center_token": center,
        "center_in_window": 0,
        "position_in_sequence": position,
        "text": f"{text} context",
        "token_strings": [center, "Ġcontext"],
        "token_texts": [text, " context"],
        "token_ids": [1, 2],
        "g_s": 4.2,
    }


def _analysis(tmp_path: Path) -> Path:
    analysis = tmp_path / "run" / "analysis"
    analysis.mkdir(parents=True)
    rows = []
    specs = [
        (0, 0.10, 100),
        (1, 0.20, 31),
        (2, 0.30, 50),
        (3, 0.70, 60),
        (4, 0.80, 70),
        (5, 0.90, 80),
    ]
    for feature_id, relative, fires in specs:
        rows.append(
            {
                "feature_id": feature_id,
                "classification": "intermediate",
                "relative_decoder_norm": relative,
                "fire_count": fires,
                "blind_rank_within_class": feature_id + 1,
                "blind_rank_base_tail": feature_id + 1,
                "blind_rank_finetuned_tail": 6 - feature_id,
            }
        )
    summary = {"candidate_groups": {"base-tail": [0], "shared": [], "finetuned-tail": [5]}}
    (analysis / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (analysis / "feature_metrics.json").write_text(
        json.dumps({"metadata": summary, "features": rows}), encoding="utf-8"
    )
    (analysis / "top_activation_examples.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "features": {
                    str(feature_id): [_example(f"word{feature_id}") for _ in range(8)]
                    for feature_id in range(6)
                },
            }
        ),
        encoding="utf-8",
    )
    (analysis.parent / "config_resolved.yaml").write_text(
        "data:\n  sequence_length: 256\n", encoding="utf-8"
    )
    return analysis


def test_crosscoder_candidate_selection_preserves_measured_tail_order_and_support(
    tmp_path: Path,
) -> None:
    adapter = CrosscoderCandidateAdapter(_analysis(tmp_path))
    candidates = adapter.select_candidates(
        base_tail_count=2, finetuned_tail_count=2, min_fire_count=32
    )
    assert [candidate.feature_id for candidate in candidates] == [0, 2, 5, 4]
    assert [candidate.relative_decoder_norm for candidate in candidates] == [0.1, 0.3, 0.9, 0.8]
    assert candidates[0].original_inspection_groups == ["base-tail"]
    assert candidates[2].original_inspection_groups == ["finetuned-tail"]
    assert all(candidate.feature_id != 1 for candidate in candidates)


def test_structural_filter_recognizes_required_centers_and_position_statistics() -> None:
    assert "assistant_role" in center_categories(_example("assistant"))
    assert "think_boundary" in center_categories(_example("</think>"))
    assert "newline_only" in center_categories(_example("Ċ", decoded="\n"))
    assert "sequence_start" in center_categories(_example(position=0), sequence_length=256)
    assert "sequence_end" in center_categories(_example(position=255), sequence_length=256)

    contexts = [
        _example("assistant"),
        _example("<|im_end|>"),
        _example("Ċ", decoded="\n"),
        _example("</think>"),
        _example("user"),
        _example("word", position=0),
        _example("apple"),
        _example("pear"),
    ]
    result = analyze_structural_contexts(contexts, threshold=0.70, sequence_length=256)
    assert result.structural_center_count == 6
    assert result.structural_center_fraction == 0.75
    assert result.sequence_start_count == 1
    assert result.special_token_center_count == 1
    assert result.structural_dominated is True


def test_blind_prompt_excludes_all_measured_side_metadata() -> None:
    examples = [_example("orchard"), _example("fruit")]
    prompt = json.dumps(render_messages(examples))
    forbidden = [
        "relative norm",
        "base-tail",
        "finetuned-tail",
        "strict classification",
        "ASPD 440",
        "previous secret label",
        "0.123456",
    ]
    assert all(value not in prompt for value in forbidden)
    assert "Context 0" in prompt and "CENTER" in prompt


def _valid_judge_payload() -> dict[str, Any]:
    return {
        "semantic_type": "topic_content",
        "tentative_label": "fruit and orchard descriptions",
        "coherence": 2,
        "content_level": True,
        "confidence": 2,
        "evidence_context_ids": [0, 1],
        "counterexample_context_ids": [],
    }


def test_judge_schema_and_bounded_repair() -> None:
    assert JudgeOutput.model_validate(_valid_judge_payload()).semantic_type == "topic_content"
    invalid = _valid_judge_payload() | {"semantic_type": "invented"}
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate(invalid)
    inconsistent = _valid_judge_payload() | {"content_level": False}
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate(inconsistent)

    repaired = parse_judge_output(
        f"```json\n{json.dumps(_valid_judge_payload())}\n```", valid_context_ids={0, 1}
    )
    assert repaired.parse_failed is False and repaired.repaired is True
    repeated = parse_judge_output(
        f"Assistant: {json.dumps(_valid_judge_payload())}\n{json.dumps(_valid_judge_payload())}",
        valid_context_ids={0, 1},
    )
    assert repeated.parse_failed is False and repeated.repaired is True
    bad_id = _valid_judge_payload() | {"evidence_context_ids": [9]}
    failed = parse_judge_output(json.dumps(bad_id), valid_context_ids={0, 1})
    assert failed.parse_failed is True
    assert failed.raw_output == json.dumps(bad_id)
    overlong = _valid_judge_payload() | {"evidence_context_ids": [0, 1, 2, 3]}
    normalized = parse_judge_output(json.dumps(overlong), valid_context_ids={0, 1, 2, 3})
    assert normalized.parse_failed is False and normalized.repaired is True
    assert normalized.result is not None
    assert normalized.result.evidence_context_ids == [0, 1, 2]
    overlong_bad = _valid_judge_payload() | {"evidence_context_ids": [0, 1, 2, 99]}
    rejected = parse_judge_output(json.dumps(overlong_bad), valid_context_ids={0, 1, 2})
    assert rejected.parse_failed is True


def _feature_for_review(
    feature_id: int,
    rank: int,
    *,
    content: bool = True,
    coherence: int = 2,
    confidence: int = 2,
) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "measured": {
            "scan_pools": ["base_tail"],
            "blind_rank_base_tail": rank,
            "blind_rank_finetuned_tail": 100 - rank,
            "fire_count": 50,
        },
        "judge": {
            "status": "judged",
            "semantic_type": "topic_content",
            "tentative_label": "fruit",
            "coherence": coherence,
            "content_level": content,
            "confidence": confidence,
            "evidence_context_ids": [0],
        },
    }


def test_review_shortlist_filters_semantically_but_sorts_by_measured_rank_first() -> None:
    features = [
        _feature_for_review(10, 5, coherence=3, confidence=3),
        _feature_for_review(11, 2, coherence=2, confidence=2),
        _feature_for_review(12, 1, content=False),
        _feature_for_review(13, 3, coherence=1),
        _feature_for_review(14, 4, confidence=1),
    ]
    review = select_review_candidates(features)
    assert [row["feature_id"] for row in review] == [11, 10]


class _FakeAdapter(CandidateAdapter):
    def __init__(self, context_count: int) -> None:
        self.context_count = context_count

    @property
    def method_name(self) -> str:
        return "future_delta_method"

    @property
    def sequence_length(self) -> int | None:
        return 128

    def select_candidates(self, **_: Any) -> list[MeasuredCandidate]:
        return [
            MeasuredCandidate(
                feature_id=7,
                strict_classification="method_owned_class",
                relative_decoder_norm=0.123456,
                fire_count=64,
                blind_rank_within_class=1,
                blind_rank_base_tail=1,
                blind_rank_finetuned_tail=99,
                original_inspection_groups=["method-owned-group"],
                scan_pools=["base_tail"],
            )
        ]

    def load_contexts(self, feature_id: int) -> list[dict[str, Any]]:
        assert feature_id == 7
        return [_example("orchard") for _ in range(self.context_count)]

    def provenance(self) -> dict[str, Any]:
        return {"fixture": True}


class _FailIfCalledJudge(BatchJudge):
    def generate(self, context_batches: Any) -> list[GenerationRecord]:
        raise AssertionError("judge must not run without sufficient contexts")


class _StaticJudge(BatchJudge):
    model_revision = "model-revision"
    tokenizer_revision = "tokenizer-revision"
    load_seconds = 1.25

    def generate(self, context_batches: Any) -> list[GenerationRecord]:
        return [
            GenerationRecord(
                raw_output=json.dumps(_valid_judge_payload()),
                prompt_tokens=100,
                generated_tokens=25,
            )
            for _ in context_batches
        ]


def _pipeline_config(tmp_path: Path, *, dry_run: bool = False) -> SemanticTriageConfig:
    return SemanticTriageConfig(
        output=tmp_path / "scan.json",
        review_output=tmp_path / "review.json",
        candidate_output=tmp_path / "plan.json",
        dry_run=dry_run,
        min_contexts_for_judge=6,
        contexts_per_feature=8,
    )


def test_generic_adapter_pipeline_and_missing_context_behavior(tmp_path: Path) -> None:
    artifact = run_semantic_triage(
        _FakeAdapter(context_count=5),
        _pipeline_config(tmp_path),
        judge=_FailIfCalledJudge(),
    )
    assert artifact["metadata"]["method"] == "future_delta_method"
    assert artifact["features"][0]["judge"]["status"] == "insufficient_context"
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["candidates"][0]["requires_context_backfill"] is True
    assert plan["candidates"][0]["eligible_for_judge"] is False


def test_pipeline_keeps_measurement_and_judgment_separate(tmp_path: Path) -> None:
    artifact = run_semantic_triage(
        _FakeAdapter(context_count=8),
        _pipeline_config(tmp_path),
        judge=_StaticJudge(),
    )
    feature = artifact["features"][0]
    assert feature["measured"]["strict_classification"] == "method_owned_class"
    assert feature["measured"]["relative_decoder_norm"] == 0.123456
    assert feature["judge"]["status"] == "judged"
    assert artifact["summary"]["judge_success"] == 1
    assert json.loads((tmp_path / "review.json").read_text())["candidates"][0][
        "feature_id"
    ] == 7


def test_dry_run_renders_prompts_without_loading_a_judge(tmp_path: Path) -> None:
    artifact = run_semantic_triage(
        _FakeAdapter(context_count=8), _pipeline_config(tmp_path, dry_run=True)
    )
    assert artifact["features"][0]["judge"]["status"] == "dry_run"
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["candidates"][0]["prompt_rendered"] is True
    assert plan["candidates"][0]["prompt_sha256"]

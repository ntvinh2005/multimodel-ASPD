"""Planning, structural filtering, blind judging, and review-shortlist assembly."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aspd.multimodel.semantic_triage.adapters import CandidateAdapter
from aspd.multimodel.semantic_triage.judge import BatchJudge, build_judge, parse_judge_output
from aspd.multimodel.semantic_triage.prompts import PROMPT_VERSION, render_user_prompt
from aspd.multimodel.semantic_triage.structural import analyze_structural_contexts

SCIENTIFIC_WARNING = (
    "LLM semantic judgments are post-ranking interpretations, not measured ground truth and "
    "not causal evidence. They never redefine measured model specificity or ranking."
)


@dataclass(frozen=True)
class SemanticTriageConfig:
    output: Path
    review_output: Path
    candidate_output: Path
    judge_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    judge_revision: str | None = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    tokenizer_revision: str | None = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    engine: str = "vllm"
    device: str = "cuda"
    batch_size: int = 16
    base_tail_count: int = 200
    finetuned_tail_count: int = 200
    min_fire_count: int = 32
    contexts_per_feature: int = 8
    min_contexts_for_judge: int = 6
    structural_threshold: float = 0.70
    temperature: float = 0.0
    max_new_tokens: int = 160
    dry_run: bool = False
    feature_ids: tuple[int, ...] | None = None
    max_judge_features: int | None = None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def _judge_shell(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "semantic_type": None,
        "tentative_label": None,
        "coherence": None,
        "content_level": None,
        "confidence": None,
        "evidence_context_ids": [],
        "counterexample_context_ids": [],
        "parse_repaired": False,
        "parse_error": None,
        "raw_output": None,
    }


def select_review_candidates(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter semantically, then preserve measured tail rank as the primary sort key."""

    selected = []
    for feature in features:
        judge = feature["judge"]
        if not (
            judge["status"] == "judged"
            and judge["content_level"] is True
            and judge["coherence"] >= 2
            and judge["confidence"] >= 2
        ):
            continue
        measured = feature["measured"]
        pools = measured["scan_pools"]
        ranks = []
        if "base_tail" in pools:
            ranks.append(measured["blind_rank_base_tail"])
        if "finetuned_tail" in pools:
            ranks.append(measured["blind_rank_finetuned_tail"])
        measured_tail_rank = min(
            ranks
            or [measured["blind_rank_base_tail"], measured["blind_rank_finetuned_tail"]]
        )
        selected.append(
            {
                "feature_id": feature["feature_id"],
                "measured_tail_rank": measured_tail_rank,
                "scan_pools": pools,
                "fire_count": measured["fire_count"],
                "semantic_type": judge["semantic_type"],
                "tentative_label": judge["tentative_label"],
                "coherence": judge["coherence"],
                "confidence": judge["confidence"],
                "evidence_context_ids": judge["evidence_context_ids"],
            }
        )
    selected.sort(
        key=lambda row: (
            row["measured_tail_rank"],
            -row["coherence"],
            -row["confidence"],
            row["feature_id"],
        )
    )
    return selected


def _validate_config(config: SemanticTriageConfig) -> None:
    positive = {
        "batch_size": config.batch_size,
        "contexts_per_feature": config.contexts_per_feature,
        "min_contexts_for_judge": config.min_contexts_for_judge,
        "max_new_tokens": config.max_new_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"expected positive values for: {invalid}")
    if config.min_contexts_for_judge > config.contexts_per_feature:
        raise ValueError("min_contexts_for_judge cannot exceed contexts_per_feature")
    if not 0 <= config.structural_threshold <= 1:
        raise ValueError("structural_threshold must be between 0 and 1")
    if config.temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if config.max_judge_features is not None and config.max_judge_features <= 0:
        raise ValueError("max_judge_features must be positive")


def run_semantic_triage(
    adapter: CandidateAdapter,
    config: SemanticTriageConfig,
    *,
    judge: BatchJudge | None = None,
) -> dict[str, Any]:
    """Run secondary triage without feeding any measured metadata to the judge."""

    _validate_config(config)
    wall_started = time.perf_counter()
    candidates = adapter.select_candidates(
        base_tail_count=config.base_tail_count,
        finetuned_tail_count=config.finetuned_tail_count,
        min_fire_count=config.min_fire_count,
        feature_ids=config.feature_ids,
    )
    plans: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    judge_queue: list[tuple[int, list[dict[str, Any]]]] = []

    for candidate in candidates:
        contexts = adapter.load_contexts(candidate.feature_id)[: config.contexts_per_feature]
        structural = analyze_structural_contexts(
            contexts,
            threshold=config.structural_threshold,
            sequence_length=adapter.sequence_length,
        )
        enough_contexts = len(contexts) >= config.min_contexts_for_judge
        eligible = enough_contexts and not structural.structural_dominated
        prompt = render_user_prompt(contexts) if eligible else None
        measured = candidate.model_dump(mode="json")
        plan = {
            "feature_id": candidate.feature_id,
            "measured": measured,
            "requested_context_count": config.contexts_per_feature,
            "available_context_count": len(adapter.load_contexts(candidate.feature_id)),
            "contexts_used": len(contexts),
            "requires_context_backfill": not enough_contexts,
            "eligible_for_judge": eligible,
            "deterministic_structural": structural.model_dump(mode="json"),
            "prompt_rendered": prompt is not None,
            "prompt_character_count": len(prompt) if prompt is not None else 0,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
        }
        plans.append(plan)
        if not enough_contexts:
            status = "insufficient_context"
        elif structural.structural_dominated:
            status = "skipped_structural"
        else:
            status = "dry_run" if config.dry_run else "queued"
            judge_queue.append((len(feature_rows), contexts))
        feature_rows.append(
            {
                "feature_id": candidate.feature_id,
                "measured": measured,
                "deterministic_structural": structural.model_dump(mode="json"),
                "context_support": {
                    "requested": config.contexts_per_feature,
                    "available": len(adapter.load_contexts(candidate.feature_id)),
                    "used": len(contexts),
                },
                "judge": _judge_shell(status),
            }
        )

    plan_artifact = {
        "metadata": {
            "method": adapter.method_name,
            "status": "candidate_plan",
            "candidate_selection": {
                "base_tail_count": config.base_tail_count,
                "finetuned_tail_count": config.finetuned_tail_count,
                "min_fire_count": config.min_fire_count,
                "explicit_feature_ids": list(config.feature_ids) if config.feature_ids else None,
            },
            "contexts_per_feature": config.contexts_per_feature,
            "min_contexts_for_judge": config.min_contexts_for_judge,
            "structural_threshold": config.structural_threshold,
            "prompt_version": PROMPT_VERSION,
            "scientific_warning": SCIENTIFIC_WARNING,
            "sources": adapter.provenance(),
        },
        "summary": {
            "requested_candidates": len(candidates),
            "requires_context_backfill": sum(row["requires_context_backfill"] for row in plans),
            "eligible_for_judge": sum(row["eligible_for_judge"] for row in plans),
            "structural_skipped": sum(
                row["deterministic_structural"]["structural_dominated"] for row in plans
            ),
        },
        "candidates": plans,
    }
    # The plan is durable before any model load or generation attempt.
    _write_json(config.candidate_output, plan_artifact)

    model_load_seconds = 0.0
    generation_seconds = 0.0
    prompt_tokens = 0
    generated_tokens = 0
    if not config.dry_run and judge_queue:
        selected_queue = judge_queue[: config.max_judge_features]
        for row_index, _ in judge_queue[len(selected_queue) :]:
            feature_rows[row_index]["judge"] = _judge_shell("not_judged_limit")
        active_judge = judge or build_judge(
            engine=config.engine,
            model_id=config.judge_model,
            model_revision=config.judge_revision,
            tokenizer_revision=config.tokenizer_revision,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            device=config.device,
            batch_size=config.batch_size,
        )
        model_load_seconds = active_judge.load_seconds
        generation_started = time.perf_counter()
        generations = active_judge.generate([contexts for _, contexts in selected_queue])
        generation_seconds = time.perf_counter() - generation_started
        if len(generations) != len(selected_queue):
            raise RuntimeError(
                f"judge returned {len(generations)} generations for {len(selected_queue)} prompts"
            )
        for (row_index, contexts), generation in zip(selected_queue, generations, strict=True):
            parsed = parse_judge_output(
                generation.raw_output, valid_context_ids=set(range(len(contexts)))
            )
            prompt_tokens += generation.prompt_tokens
            generated_tokens += generation.generated_tokens
            if parsed.parse_failed or parsed.result is None:
                payload = _judge_shell("parse_failed")
                payload.update(
                    {
                        "parse_repaired": parsed.repaired,
                        "parse_error": parsed.error,
                        "raw_output": parsed.raw_output,
                    }
                )
            else:
                payload = parsed.result.model_dump(mode="json")
                payload.update(
                    {
                        "status": "judged",
                        "parse_repaired": parsed.repaired,
                        "parse_error": None,
                        "raw_output": parsed.raw_output,
                    }
                )
            feature_rows[row_index]["judge"] = payload
        resolved_model_revision = active_judge.model_revision
        resolved_tokenizer_revision = active_judge.tokenizer_revision
    else:
        resolved_model_revision = config.judge_revision
        resolved_tokenizer_revision = config.tokenizer_revision

    judged_count = sum(row["judge"]["status"] in {"judged", "parse_failed"} for row in feature_rows)
    eligible_count = len(judge_queue)
    projected_seconds = None
    if judged_count and generation_seconds > 0:
        projected_seconds = model_load_seconds + generation_seconds / judged_count * eligible_count
    summary = {
        "requested_candidates": len(candidates),
        "eligible_after_support_filter": sum(
            row["context_support"]["used"] >= config.min_contexts_for_judge
            for row in feature_rows
        ),
        "structural_skipped": sum(
            row["judge"]["status"] == "skipped_structural" for row in feature_rows
        ),
        "judge_requested": judged_count,
        "judge_success": sum(row["judge"]["status"] == "judged" for row in feature_rows),
        "judge_parse_failed": sum(
            row["judge"]["status"] == "parse_failed" for row in feature_rows
        ),
        "content_level_count": sum(row["judge"]["content_level"] is True for row in feature_rows),
        "insufficient_context": sum(
            row["judge"]["status"] == "insufficient_context" for row in feature_rows
        ),
        "eligible_nonstructural_total": eligible_count,
    }
    timing = {
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
        "wall_clock_seconds": time.perf_counter() - wall_started,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": (
            generated_tokens / generation_seconds if generation_seconds > 0 else None
        ),
        "projected_full_judge_seconds": projected_seconds,
        "projected_full_judge_minutes": projected_seconds / 60 if projected_seconds else None,
        "full_submission_within_45_minute_guardrail": (
            projected_seconds <= 45 * 60 if projected_seconds is not None else None
        ),
    }
    artifact = {
        "metadata": {
            "method": adapter.method_name,
            "status": "post_ranking_semantic_triage",
            "dry_run": config.dry_run,
            "judge_engine": config.engine,
            "judge_model": config.judge_model,
            "judge_revision": resolved_model_revision,
            "tokenizer_revision": resolved_tokenizer_revision,
            "prompt_version": PROMPT_VERSION,
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "contexts_per_feature": config.contexts_per_feature,
            "min_contexts_for_judge": config.min_contexts_for_judge,
            "min_fire_count": config.min_fire_count,
            "structural_threshold": config.structural_threshold,
            "candidate_selection": {
                "base_tail_count": config.base_tail_count,
                "finetuned_tail_count": config.finetuned_tail_count,
            },
            "scientific_warning": SCIENTIFIC_WARNING,
            "sources": adapter.provenance(),
        },
        "summary": summary,
        "timing": timing,
        "features": feature_rows,
    }
    review = {
        "metadata": {
            "method": adapter.method_name,
            "status": "post_hoc_manual_review_shortlist",
            "inclusion_rule": "content_level=true, coherence>=2, confidence>=2",
            "sorting": "measured tail rank primary; coherence/confidence secondary only",
            "not_a_new_ranking": True,
            "scientific_warning": SCIENTIFIC_WARNING,
            "source": str(config.output),
        },
        "candidates": select_review_candidates(feature_rows),
    }
    _write_json(config.output, artifact)
    _write_json(config.review_output, review)
    return artifact

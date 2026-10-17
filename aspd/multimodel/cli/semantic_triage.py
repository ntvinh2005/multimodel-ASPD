"""Plan or run blind post-ranking semantic triage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.semantic_triage.adapters import candidate_adapter
from aspd.multimodel.semantic_triage.pipeline import SemanticTriageConfig, run_semantic_triage


def parse_feature_ids(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("feature IDs must be comma-separated integers") from exc
    if not ids:
        raise argparse.ArgumentTypeError("at least one feature ID is required")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind structural filtering and batched LLM semantic triage"
    )
    parser.add_argument("--method", choices=("crosscoder",), required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument(
        "--judge-revision", default="916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    )
    parser.add_argument(
        "--tokenizer-revision", default="916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    )
    parser.add_argument("--engine", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-tail-count", type=int, default=200)
    parser.add_argument("--finetuned-tail-count", type=int, default=200)
    parser.add_argument("--min-fire-count", type=int, default=32)
    parser.add_argument("--contexts-per-feature", type=int, default=8)
    parser.add_argument("--min-contexts-for-judge", type=int, default=6)
    parser.add_argument("--structural-threshold", type=float, default=0.70)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--feature-ids", type=parse_feature_ids)
    parser.add_argument(
        "--max-judge-features",
        type=int,
        help="smoke-test cap applied after deterministic/support filtering",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    output = (args.output or analysis_dir / "content_semantic_scan.json").resolve()
    review_output = (
        args.review_output or analysis_dir / "content_semantic_review_candidates.json"
    ).resolve()
    candidate_output = (
        args.candidate_output or analysis_dir / "content_semantic_scan_candidates.json"
    ).resolve()
    config = SemanticTriageConfig(
        output=output,
        review_output=review_output,
        candidate_output=candidate_output,
        judge_model=args.judge_model,
        judge_revision=args.judge_revision,
        tokenizer_revision=args.tokenizer_revision,
        engine=args.engine,
        device=args.device,
        batch_size=args.batch_size,
        base_tail_count=args.base_tail_count,
        finetuned_tail_count=args.finetuned_tail_count,
        min_fire_count=args.min_fire_count,
        contexts_per_feature=args.contexts_per_feature,
        min_contexts_for_judge=args.min_contexts_for_judge,
        structural_threshold=args.structural_threshold,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
        feature_ids=args.feature_ids,
        max_judge_features=args.max_judge_features,
    )
    artifact = run_semantic_triage(candidate_adapter(args.method, analysis_dir), config)
    print(
        json.dumps(
            {
                "output": str(output),
                "review_output": str(review_output),
                "candidate_output": str(candidate_output),
                "summary": artifact["summary"],
                "timing": artifact["timing"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

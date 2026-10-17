"""Targeted context backfill for semantic triage, kept separate from canonical analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.cli.semantic_triage import parse_feature_ids
from aspd.multimodel.semantic_triage.adapters import CrosscoderCandidateAdapter


def _ids_from_plan(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError(f"candidate plan has no candidates list: {path}")
    return [
        int(row["feature_id"])
        for row in rows
        if isinstance(row, dict) and row.get("requires_context_backfill") is True
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay cached activations for targeted semantic-triage contexts"
    )
    parser.add_argument("--method", choices=("crosscoder",), required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--feature-ids", type=parse_feature_ids)
    selection.add_argument("--candidate-plan", type=Path)
    parser.add_argument("--top-contexts", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    feature_ids = (
        list(args.feature_ids)
        if args.feature_ids is not None
        else _ids_from_plan(args.candidate_plan)
    )
    if not feature_ids:
        print(json.dumps({"status": "nothing_to_backfill", "requested_features": 0}, indent=2))
        return
    output = CrosscoderCandidateAdapter.backfill_contexts(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        checkpoint=args.checkpoint,
        feature_ids=feature_ids,
        top_contexts=args.top_contexts,
        device_name=args.device,
        output_path=args.output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    features = payload.get("features", {})
    print(
        json.dumps(
            {
                "output": str(output),
                "requested_features": len(feature_ids),
                "requested_with_positive_contexts": sum(bool(features.get(str(i))) for i in feature_ids),
                "features_in_artifact": len(features),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

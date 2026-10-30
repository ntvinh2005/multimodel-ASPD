"""Summarize researcher-supplied P6 behavioral scores; never invokes a judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.p6_behavioral import summarize_behavioral_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", help="completed behavioral_evaluation_records.jsonl")
    parser.add_argument(
        "--frozen-template",
        required=True,
        help="immutable behavioral_evaluation_records.jsonl emitted by the P6 job",
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--approved-scorer-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = summarize_behavioral_scores(
        args.scores,
        args.contract,
        approved_scorer_id=args.approved_scorer_id,
        frozen_template_path=args.frozen_template,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

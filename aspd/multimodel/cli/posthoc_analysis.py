"""Run the multimodel ASPD post-hoc analysis (edge scores and parent-set comparison).

Examples::

    # what is still blocked in a config (no compute)
    python -m aspd.multimodel.cli.posthoc_analysis --config CONFIG --check

    # full pipeline into a new output root
    python -m aspd.multimodel.cli.posthoc_analysis --config CONFIG --output-dir OUT

    # one stage at a time (each stage reuses the verified earlier stages in OUT)
    python -m aspd.multimodel.cli.posthoc_analysis --config CONFIG --output-dir OUT --stage importance

Stages: importance, representatives, edges, compare (or ``all``).  See
docs/multimodel/posthoc_analysis.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aspd.multimodel.posthoc.config import STAGES, PosthocBlocked, load_config
from aspd.multimodel.posthoc.pipeline import run_stages


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", action="append", choices=(*STAGES, "all"),
                        help="repeatable; default: all")
    parser.add_argument("--check", action="store_true",
                        help="print unresolved decisions per stage and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    config = load_config(args.config)
    if args.check:
        report = {stage: config.blockers(stage) for stage in STAGES}
        print(json.dumps({"experiment_id": config.experiment_id, "blockers": report}, indent=2))
        return 0 if not any(report.values()) else 2
    if args.output_dir is None:
        print("--output-dir is required unless --check is given", file=sys.stderr)
        return 2
    stages = STAGES if not args.stage or "all" in args.stage else tuple(args.stage)
    try:
        outputs = run_stages(config, args.output_dir, stages)
    except PosthocBlocked as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 3
    for path in outputs:
        print(f"COMPLETE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

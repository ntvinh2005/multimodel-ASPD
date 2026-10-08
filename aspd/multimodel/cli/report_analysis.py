"""Build review tables and figures from saved P1--P5 artifacts."""

from __future__ import annotations

import argparse

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.posthoc_report import build_posthoc_report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize saved P1--P5 multi-model analysis"
    )
    parser.add_argument("config", help="training config used for the analyzed checkpoint")
    parser.add_argument("analysis_dir", help="directory containing posthoc.safetensors and JSON")
    parser.add_argument("--output-dir", help="defaults to analysis_dir")
    parser.add_argument("--top-per-direction", type=_positive_int, default=10)
    parser.add_argument("--control-count", type=_positive_int, default=5)
    parser.add_argument("--low-support-threshold", type=_positive_int, default=32)
    parser.add_argument("--dpi", type=_positive_int, default=160)
    args = parser.parse_args()

    output = build_posthoc_report(
        load_experiment_config(args.config),
        args.analysis_dir,
        output_dir=args.output_dir,
        top_per_direction=args.top_per_direction,
        control_count=args.control_count,
        low_support_threshold=args.low_support_threshold,
        dpi=args.dpi,
    )
    print(output)


if __name__ == "__main__":
    main()

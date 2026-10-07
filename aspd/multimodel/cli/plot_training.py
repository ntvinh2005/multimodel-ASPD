"""Plot multi-model ASPD training curves from ``metrics.jsonl``."""

from __future__ import annotations

import argparse

from aspd.multimodel.plotting import (
    load_training_metrics,
    plot_training_metrics,
    plot_validation_internal_by_matrix,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw loss, FVU, L0, and dead-feature curves from metrics.jsonl"
    )
    parser.add_argument(
        "metrics",
        help="metrics.jsonl path, or a run directory containing metrics.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output image path; supports PNG, PDF, SVG, JPEG, and WebP",
    )
    parser.add_argument(
        "--matrix-output",
        help="optional separate validation per-matrix internal-FVU image",
    )
    parser.add_argument(
        "--smooth",
        type=_positive_int,
        default=1,
        help="trailing-window mean for train curves only (default: 1, disabled)",
    )
    parser.add_argument(
        "--target-l0",
        type=float,
        help="optional horizontal reference line in the sparsity panel",
    )
    parser.add_argument("--title", help="optional figure title")
    parser.add_argument("--dpi", type=_positive_int, default=160)
    args = parser.parse_args()

    rows = load_training_metrics(args.metrics)
    output = plot_training_metrics(
        rows,
        args.output,
        smooth=args.smooth,
        target_l0=args.target_l0,
        title=args.title,
        dpi=args.dpi,
    )
    print(output)
    if args.matrix_output:
        matrix_output = plot_validation_internal_by_matrix(
            rows,
            args.matrix_output,
            title=f"{args.title} — per-matrix validation" if args.title else None,
            dpi=args.dpi,
        )
        print(matrix_output)


if __name__ == "__main__":
    main()

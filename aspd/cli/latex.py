"""Render the paper's result tables as LaTeX from per-model table CSVs.

    python -m aspd.cli.latex --model GPT2=tables/gpt2_full.csv \\
        --model Gemma-2-2B=tables/gemma2_full.csv --model Qwen-3-8B=tables/qwen3_full.csv --out tables/
"""

import argparse
from pathlib import Path

from aspd.eval.latex import ARM_ORDER, write_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=PATH of a `*_full.csv` from `aspd.cli.tables --ci`, in column order")
    ap.add_argument("--arms", nargs="+", default=None, choices=ARM_ORDER,
                    help="rows to include, in order (default: every arm present)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    model_csvs = dict(item.split("=", 1) for item in args.model)
    for path in write_all({k: Path(v) for k, v in model_csvs.items()}, args.out, args.arms):
        print(path)


if __name__ == "__main__":
    main()

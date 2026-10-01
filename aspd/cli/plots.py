"""Draw the figures for a run's evaluation results."""

import argparse
import re
from pathlib import Path

from aspd.eval.plots import (
    plot_aggregate,
    plot_attr_edit_dir,
    plot_attr_edit_multi_dir,
    plot_matching_dir,
    safe_plot,
)

STAGES = ("matching", "attr_edit", "attr_edit_multi")


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="results: matching/, attr_edit/, attr_edit_multi/")
    ap.add_argument("--sae-dir", default=None, help="the evaluation SAE directory")
    ap.add_argument("--stages", default="all",
                    help=f"'all' or a list separated by , : or space from {STAGES}. Use COLONS "
                         "under sbatch --export, which splits on commas")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="skip the cross-stage summary figure")
    ap.add_argument("--aggregate-out", default=None,
                    help="default: <run-dir>/eval_summary, or <sae-dir>/eval_summary")
    return ap


def parse_stages(spec: str) -> list[str]:
    if spec.strip() == "all":
        return list(STAGES)
    stages = [s.strip() for s in re.split(r"[,:\s]+", spec.strip()) if s.strip()]
    unknown = sorted(set(stages) - set(STAGES))
    assert not unknown, f"unknown stage(s) {unknown}; choose from {STAGES}"
    assert stages, f"could not parse --stages {spec!r}"
    return stages


def main() -> None:
    args = arg_parser().parse_args()
    assert args.run_dir or args.sae_dir, "pass --run-dir, --sae-dir or both"
    run_dir = Path(args.run_dir) if args.run_dir else None
    sae_dir = Path(args.sae_dir) if args.sae_dir else None
    stages = parse_stages(args.stages)

    written: list[Path] = []
    if "matching" in stages and run_dir and (run_dir / "matching").exists():
        written += safe_plot(plot_matching_dir, run_dir / "matching")
    if "attr_edit" in stages and run_dir and (run_dir / "attr_edit").exists():
        written += safe_plot(plot_attr_edit_dir, run_dir / "attr_edit")
    if "attr_edit_multi" in stages and run_dir and (run_dir / "attr_edit_multi").exists():
        written += safe_plot(plot_attr_edit_multi_dir, run_dir / "attr_edit_multi")

    if not args.no_aggregate:
        base = run_dir or sae_dir
        assert base is not None
        out_dir = Path(args.aggregate_out) if args.aggregate_out else base / "eval_summary"
        written += safe_plot(plot_aggregate, run_dir, sae_dir, out_dir)

    print(f"[plots] {len(written)} file(s) written")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()

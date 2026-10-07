"""Summarize a completed D0/S1 main-size capacity pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.capacity import build_capacity_report, format_capacity_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure capacity-pilot timing, VRAM, GPU utilization, and sanity checks"
    )
    parser.add_argument("run_dir")
    parser.add_argument(
        "--output",
        help="JSON output path (default: <run-dir>/capacity_report.json)",
    )
    args = parser.parse_args()

    report = build_capacity_report(args.run_dir)
    output = Path(args.output or Path(args.run_dir) / "capacity_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_capacity_report(report))
    print(f"report: {output}")


if __name__ == "__main__":
    main()

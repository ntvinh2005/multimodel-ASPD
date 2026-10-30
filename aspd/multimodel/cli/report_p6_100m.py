"""Build the exploratory P6-100M candidate report from blinded measured scores."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.p6_100m_reporting import build_p6_100m_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--behavioral-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = build_p6_100m_report(
        args.generation_dir,
        args.behavioral_scores,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "candidate_count": len(summary["candidates"]),
                "parse_failures": summary["parse_failures"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

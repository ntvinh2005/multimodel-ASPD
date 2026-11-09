"""Create a reproducible Matryoshka feature-band utilization report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.v3.feature_band_diagnostic import analyze_feature_activity_file
from aspd.multimodel.v3.objective import MATRYOSHKA_GROUP_FRACS


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("inputs must use LABEL=/path/to/feature_activity.csv")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=_labeled_path,
        metavar="LABEL=CSV",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.input]
    if len(labels) != len(set(labels)):
        parser.error("input labels must be unique")
    missing = [str(path) for _, path in args.input if not path.is_file()]
    if missing:
        parser.error(f"input files do not exist: {missing}")

    payload = {
        "schema_version": 1,
        "status": "READ_ONLY_FEATURE_BAND_DIAGNOSTIC_COMPLETE",
        "n_features": 8192,
        "matryoshka_group_fracs": list(MATRYOSHKA_GROUP_FRACS),
        "interpretation_boundary": (
            "Firing shares and summed L2 contribution magnitudes are observational; "
            "they do not prove that Matryoshka weighting caused dictionary concentration."
        ),
        "runs": {label: analyze_feature_activity_file(path) for label, path in args.input},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

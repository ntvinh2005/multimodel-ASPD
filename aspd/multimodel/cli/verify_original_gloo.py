"""Execute the local three-process CPU/Gloo checkpoint-resume verification."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_gloo import execute_three_process_gloo_verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = execute_three_process_gloo_verification(
        load_experiment_config(args.config), args.output_root, args.report
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

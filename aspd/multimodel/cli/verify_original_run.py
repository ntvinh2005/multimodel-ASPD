"""Verify an original-recipe ASPD smoke or full training run offline."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_verification import verify_original_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("run_dir")
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation")
    parser.add_argument("--minimum-positions-per-second", type=float, default=4_500.0)
    args = parser.parse_args()
    result = verify_original_run(
        load_experiment_config(args.config),
        args.run_dir,
        mode=args.mode,
        output_path=args.output,
        evaluation_path=args.evaluation,
        minimum_positions_per_second=args.minimum_positions_per_second,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "NO_GO_RUNTIME":
        raise SystemExit(3)


if __name__ == "__main__":
    main()

"""Run the bounded 18-sequence teacher-forced ASPD P6 feasibility pilot."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.causal import DEFAULT_RANDOM_CONTROL_SEEDS
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.p6_fixed import run_fixed_p6_feasibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--localization", required=True)
    parser.add_argument("--posthoc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--random-controls",
        action="store_true",
        help="opt in to random same-L2 write-direction arms",
    )
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        help="random-control seeds (default when enabled: 0 1 2 3 4)",
    )
    args = parser.parse_args()
    if args.random_seeds is not None and not args.random_controls:
        parser.error("--random-seeds requires --random-controls")
    random_seeds = args.random_seeds or list(DEFAULT_RANDOM_CONTROL_SEEDS)
    payload = run_fixed_p6_feasibility(
        load_experiment_config(args.config),
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        localization_path=args.localization,
        posthoc_path=args.posthoc,
        output_dir=args.output_dir,
        device_name=args.device,
        random_control_seeds=random_seeds if args.random_controls else None,
    )
    print(
        json.dumps(
            {
                "status": payload["scientific_status"],
                "models": len(payload["models"]),
                "candidates": payload["candidate_ids"],
                "sequences": len(payload["selection"]["rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

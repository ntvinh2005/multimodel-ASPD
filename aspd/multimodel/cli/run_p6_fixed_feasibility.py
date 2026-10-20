"""Run the bounded 18-sequence teacher-forced ASPD P6 feasibility pilot."""

from __future__ import annotations

import argparse
import json

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
    args = parser.parse_args()
    payload = run_fixed_p6_feasibility(
        load_experiment_config(args.config),
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        localization_path=args.localization,
        posthoc_path=args.posthoc,
        output_dir=args.output_dir,
        device_name=args.device,
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

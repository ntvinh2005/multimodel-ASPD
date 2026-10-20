"""Run the bounded prefix-causal ASPD P6 greedy-generation feasibility pilot."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.p6_generation import run_prefix_generation_feasibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--localization", required=True)
    parser.add_argument("--fixed-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    payload = run_prefix_generation_feasibility(
        load_experiment_config(args.config),
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        localization_path=args.localization,
        fixed_result_path=args.fixed_result,
        output_dir=args.output_dir,
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        json.dumps(
            {
                "status": payload["scientific_status"],
                "arms": len(payload["arms"]),
                "candidates": payload["candidate_ids"],
                "sequences": payload["selection"]["sequences"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

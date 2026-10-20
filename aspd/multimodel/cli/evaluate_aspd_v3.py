"""Evaluate one ASPD checkpoint on frozen V3 exact-token R/X inputs."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.sports_risk import evaluate_v3_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--matrix-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--component-output", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-sequences", type=int, default=2)
    args = parser.parse_args()
    result = evaluate_v3_checkpoint(
        load_experiment_config(args.config),
        args.checkpoint,
        checkpoint_cfg=load_experiment_config(args.checkpoint_config),
        matrix_cache_dir=args.matrix_cache,
        output_json=args.output,
        output_components=args.component_output,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        device_name=args.device,
        batch_size_sequences=args.batch_size_sequences,
    )
    print(
        json.dumps(
            {
                "checkpoint_step": result["checkpoint"]["step"],
                "groups": len(result["groups"]),
                "output": args.output,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

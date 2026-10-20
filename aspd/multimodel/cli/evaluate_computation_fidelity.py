from __future__ import annotations

import argparse

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.fidelity import evaluate_computation_fidelity, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ASPD computation fidelity from a checkpoint and paired cache"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--batch-size-sequences", type=int)
    parser.add_argument("--expected-cache-fingerprint")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--checkpoint-config",
        help=(
            "resolved training config for a checkpoint evaluated on a compatible new cache; "
            "its training-cache RMS normalization is retained"
        ),
    )
    args = parser.parse_args()
    result = evaluate_computation_fidelity(
        load_experiment_config(args.config),
        args.checkpoint,
        checkpoint_cfg=(
            load_experiment_config(args.checkpoint_config)
            if args.checkpoint_config
            else None
        ),
        device_name=args.device,
        split=args.split,
        batch_size_sequences=args.batch_size_sequences,
        expected_cache_fingerprint=args.expected_cache_fingerprint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    output = write_json_immutable(args.output, result)
    print(output)


if __name__ == "__main__":
    main()

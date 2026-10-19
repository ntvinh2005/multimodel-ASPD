from __future__ import annotations

import argparse

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.fidelity import write_json_immutable
from aspd.multimodel.tiny_overfit import run_tiny_overfit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fail-closed ASPD fixed-batch overfit check")
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--fixed-batches", type=int, choices=(1, 2), default=1)
    parser.add_argument("--minimum-relative-internal-improvement", type=float, default=0.01)
    parser.add_argument("--expected-cache-fingerprint")
    args = parser.parse_args()
    result = run_tiny_overfit(
        load_experiment_config(args.config),
        device_name=args.device,
        steps=args.steps,
        fixed_batches=args.fixed_batches,
        minimum_relative_internal_improvement=args.minimum_relative_internal_improvement,
        expected_cache_fingerprint=args.expected_cache_fingerprint,
    )
    output = write_json_immutable(args.output, result)
    print(output)
    if not result["passed"]:
        raise SystemExit("ASPD tiny-overfit failed its pre-specified integrity criteria")


if __name__ == "__main__":
    main()

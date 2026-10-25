from __future__ import annotations

import argparse

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_streaming import run_three_gpu_streaming


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen three-GPU two-model original-ASPD stream"
    )
    parser.add_argument("config")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--benchmark-steps", type=int)
    args = parser.parse_args()
    run_three_gpu_streaming(
        load_experiment_config(args.config),
        output_dir=args.output_dir,
        resume=args.resume,
        benchmark_steps=args.benchmark_steps,
    )


if __name__ == "__main__":
    main()

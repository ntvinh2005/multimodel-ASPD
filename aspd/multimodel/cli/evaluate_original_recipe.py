from __future__ import annotations

import argparse

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_evaluation import evaluate_original_recipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a streamed two-model original-ASPD checkpoint"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint_rank0")
    parser.add_argument("output_dir")
    parser.add_argument("--checkpoint-rank1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-sequences", type=int, default=8)
    parser.add_argument("--max-validation-sequences", type=int)
    args = parser.parse_args()
    evaluate_original_recipe(
        load_experiment_config(args.config),
        args.checkpoint_rank0,
        args.output_dir,
        checkpoint_rank1=args.checkpoint_rank1,
        device_name=args.device,
        batch_size_sequences=args.batch_size_sequences,
        max_validation_sequences=args.max_validation_sequences,
    )


if __name__ == "__main__":
    main()

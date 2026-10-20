"""Run exact-LoRA positive control or one coarse localization generation shard."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.exact_lora_localization import (
    run_causal_intersection_generation,
    run_coarse_generation_shard,
    run_positive_control_preflight,
    run_refinement_generation_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "coarse", "refinement", "intersection"), required=True
    )
    parser.add_argument("--preflight")
    parser.add_argument("--coarse-scores")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    cfg = load_experiment_config(args.config)
    if args.mode == "preflight":
        if args.preflight is not None or args.coarse_scores is not None or args.shard is not None:
            parser.error("preflight mode does not accept prior-artifact or shard arguments")
        report = run_positive_control_preflight(
            cfg, args.prompts, args.output, device_name=args.device,
            max_new_tokens=args.max_new_tokens or 256,
        )
    elif args.mode == "coarse":
        if args.preflight is None or args.shard is None:
            parser.error("coarse mode requires --preflight and --shard")
        if args.coarse_scores is not None:
            parser.error("coarse mode does not accept --coarse-scores")
        report = run_coarse_generation_shard(
            cfg, args.prompts, args.preflight, args.output,
            shard=args.shard, device_name=args.device,
            max_new_tokens=args.max_new_tokens or 512, batch_size=args.batch_size,
        )
    elif args.mode == "refinement":
        if args.preflight is None or args.coarse_scores is None or args.shard is None:
            parser.error("refinement mode requires --preflight, --coarse-scores, and --shard")
        report = run_refinement_generation_shard(
            cfg, args.prompts, args.preflight, args.coarse_scores, args.output,
            shard=args.shard, device_name=args.device,
            max_new_tokens=args.max_new_tokens or 1024, batch_size=args.batch_size,
        )
    else:
        if args.preflight is None:
            parser.error("intersection mode requires --preflight")
        if args.coarse_scores is not None or args.shard is not None:
            parser.error("intersection mode does not accept --coarse-scores or --shard")
        report = run_causal_intersection_generation(
            cfg, args.prompts, args.preflight, args.output,
            device_name=args.device, max_new_tokens=args.max_new_tokens or 1024,
            batch_size=args.batch_size,
        )
    print(json.dumps({
        "status": report["scientific_status"],
        "output": args.output,
        "shard": report.get("shard"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

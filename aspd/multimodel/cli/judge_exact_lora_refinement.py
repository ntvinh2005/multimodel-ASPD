"""Blindly score the four bounded exact-LoRA refinement generation shards."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.exact_lora_judge import judge_exact_lora_refinement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and max new tokens must be positive")
    report = judge_exact_lora_refinement(
        args.shards, args.output, model_id=args.model, revision=args.revision,
        device_name=args.device, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({
        "status": report["scientific_status"],
        "attempted": report["rows_attempted"],
        "valid": report["valid_rows"],
        "parse_failures": report["parse_failures"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

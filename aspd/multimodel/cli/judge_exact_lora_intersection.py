"""Blindly score the exact up_proj × layers-7--13 intersection experiment."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.exact_lora_judge import judge_exact_lora_intersection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    report = judge_exact_lora_intersection(
        args.generation, args.output, model_id=args.model, revision=args.revision,
        device_name=args.device, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({
        "status": report["scientific_status"], "attempted": report["rows_attempted"],
        "valid": report["valid_rows"], "parse_failures": report["parse_failures"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

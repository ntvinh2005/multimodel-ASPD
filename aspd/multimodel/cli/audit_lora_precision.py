"""Audit ideal FP32 LoRA deltas against effective BF16 ASPD snapshots."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.diagnostics.lora_precision import audit_lora_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = audit_lora_precision(args.config, args.output, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

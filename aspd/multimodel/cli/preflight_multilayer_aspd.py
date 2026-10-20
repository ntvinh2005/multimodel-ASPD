"""Run the real C8192/K32 multi-layer ASPD memory and optimizer preflight."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.memory_preflight import run_memory_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    report = run_memory_preflight(
        load_experiment_config(args.config), args.cache_root, args.output,
        device_name=args.device, steps=args.steps,
    )
    print(json.dumps({
        "status": report["status"],
        "trainable_parameters": report["parameter_count"]["trainable"],
        "peak_reserved_gib": report["cuda"]["peak_reserved_bytes"] / 2**30,
        "headroom_gib": report["cuda"]["headroom_bytes"] / 2**30,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

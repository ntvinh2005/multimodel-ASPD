"""Validate the opt-in 100M P6 pilot specification without model loading."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.p6_original_causal import validate_p6_pilot_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = validate_p6_pilot_config(args.config, require_ready=args.require_ready)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

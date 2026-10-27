"""Static fail-closed audit for the common Mix3 ASPD/Delta preparation."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.mix3 import validate_delta_mix3_config, validate_mix3_recipe_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aspd_config")
    parser.add_argument("delta_config")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = {
        "aspd": validate_mix3_recipe_config(
            load_experiment_config(args.aspd_config), require_ready=args.require_ready
        ),
        "delta": validate_delta_mix3_config(load_delta_crosscoder_config(args.delta_config)),
        "execution": "STATIC_LOCAL_AUDIT_ONLY",
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

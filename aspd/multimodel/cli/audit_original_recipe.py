from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_recipe import (
    audit_cached_model_pair,
    resource_estimate,
    validate_dataset_manifest,
    validate_original_recipe_config,
)
from aspd.multimodel.original_steering import steering_compatibility_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen two-model original-ASPD recipe without loading model weights"
    )
    parser.add_argument("config")
    parser.add_argument("--require-dataset-ready", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--hf-hub-cache")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    report = {
        "method": validate_original_recipe_config(cfg),
        "dataset": validate_dataset_manifest(
            cfg, require_ready=args.require_dataset_ready
        ),
        "resources": resource_estimate(cfg),
        "steering": steering_compatibility_report(),
        "execution_status": "LOCAL_IMPLEMENTATION_VERIFIED / HPC_NOT_RUN",
    }
    if args.hf_hub_cache:
        report["cached_model_pair"] = audit_cached_model_pair(args.hf_hub_cache)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

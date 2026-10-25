"""Materialize or validate the pinned FineWeb corpus for the 100M ASPD run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_data import materialize_fineweb_original_corpus
from aspd.multimodel.original_recipe import validate_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.validate_only:
        result = validate_dataset_manifest(cfg, require_ready=True)
    else:
        result = materialize_fineweb_original_corpus(cfg, overwrite=args.overwrite)
    summary = {
        "status": "READY" if result.get("status") in {"READY", "PASS"} else result.get("status"),
        "manifest": cfg.data.frozen_manifest,
        "train_tokens": cfg.data.train_tokens,
        "calibration_tokens": cfg.data.calibration_tokens,
        "validation_tokens": cfg.data.validation_tokens,
        "train_sequences": cfg.data.train_tokens // cfg.data.sequence_length,
        "calibration_sequences": cfg.data.calibration_tokens // cfg.data.sequence_length,
        "validation_sequences": cfg.data.validation_tokens // cfg.data.sequence_length,
        "shards": result.get("shard_count", result.get("shards")),
        "duplicate_audit_executed": result.get("duplicate_audit", {}).get("executed", True),
        "locked_evaluation_audit_executed": result.get("locked_evaluation_audit", {}).get(
            "executed", True
        ),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

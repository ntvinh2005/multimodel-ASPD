"""Materialize or validate the pinned FineWeb corpus for the 100M ASPD run."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_data import materialize_fineweb_original_corpus
from aspd.multimodel.original_recipe import validate_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.validate_only:
        result = validate_dataset_manifest(cfg, require_ready=True)
    else:
        result = materialize_fineweb_original_corpus(cfg, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

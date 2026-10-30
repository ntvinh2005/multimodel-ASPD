"""Validate ASPD v3 infrastructure and print its fail-closed readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.v3.config import V3HistoricalRegistry, _load, load_v3_config_bundle
from aspd.multimodel.v3.data import write_json_immutable
from aspd.multimodel.v3.feasibility import estimate_v3_feasibility
from aspd.multimodel.v3.registry import build_model_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved", default="configs/multimodel/aspd_v3/approved.yaml"
    )
    parser.add_argument(
        "--candidates", default="configs/multimodel/aspd_v3/candidates.yaml"
    )
    parser.add_argument(
        "--runtime", default="configs/multimodel/aspd_v3/runtime_template.yaml"
    )
    parser.add_argument(
        "--historical", default="configs/multimodel/aspd_v3/historical_registry.yaml"
    )
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    bundle = load_v3_config_bundle(args.approved, args.candidates, args.runtime)
    historical = V3HistoricalRegistry.model_validate(_load(args.historical))
    missing_historical = {
        name: path
        for name, path in historical.experiments.items()
        if not Path(path).is_file()
    }
    if missing_historical:
        raise FileNotFoundError(f"historical v3 registry paths are missing: {missing_historical}")
    registry = build_model_registry(bundle.approved)
    report = bundle.report()
    report["registry"] = {
        "models": list(registry),
        "logical_matrix_targets": sum(len(layout.targets()) for layout in registry.values()),
        "historical_paths_verified": len(historical.experiments),
    }
    report["feasibility"] = estimate_v3_feasibility(bundle, registry)
    if args.output:
        write_json_immutable(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.require_ready:
        bundle.require_ready()


if __name__ == "__main__":
    main()

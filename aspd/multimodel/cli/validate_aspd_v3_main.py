"""Validate frozen ASPD v3 main-v1 science separately from execution evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.v3.config import V3ApprovedConfig, _load
from aspd.multimodel.v3.feasibility import estimate_main_v1_feasibility
from aspd.multimodel.v3.main_config import load_main_v1_config
from aspd.multimodel.v3.registry import build_model_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("configs/multimodel/aspd_v3/main_v1.yaml"),
    )
    parser.add_argument("--require-executable", action="store_true")
    args = parser.parse_args()
    config = load_main_v1_config(args.config)
    report = config.structural_report()
    approved = V3ApprovedConfig.model_validate(_load(config.approved_config_path))
    report["feasibility"] = estimate_main_v1_feasibility(build_model_registry(approved))
    execution_blockers = config.execution_blockers()
    report["execution_status"] = "READY" if not execution_blockers else "BLOCKED"
    report["execution_blockers"] = execution_blockers
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "STRUCTURALLY_READY" or (
        args.require_executable and execution_blockers
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

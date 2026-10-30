"""Run one isolated ASPD-100M P6 stage; never submits a follow-up job."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.p6_original_causal import run_p6_original_causal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--stage", choices=("smoke", "discovery", "confirmation"), required=True)
    parser.add_argument("--preflight-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage != "smoke" and not args.preflight_report:
        parser.error("discovery/confirmation requires --preflight-report from smoke GO")
    if args.stage == "smoke" and args.preflight_report:
        parser.error("smoke does not accept --preflight-report")
    internal_stage = {
        "smoke": "preflight",
        "discovery": "pilot",
        "confirmation": "confirmation",
    }[args.stage]
    result = run_p6_original_causal(
        args.config,
        args.output_dir,
        device=args.device,
        verify_all_artifact_hashes=True,
        stage=internal_stage,
        preflight_report=args.preflight_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    expected = "GPU_PREFLIGHT_GO" if args.stage == "smoke" else "JOB_COMPLETED_UNVERIFIED"
    if result["status"] != expected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

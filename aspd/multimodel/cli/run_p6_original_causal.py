"""Execute exactly one frozen original-ASPD 100M P6 stage."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.p6_original_causal import run_p6_original_causal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="frozen P6 pilot YAML")
    parser.add_argument("--stage", choices=("preflight", "pilot"), required=True)
    parser.add_argument(
        "--preflight-report",
        help="required only for pilot; machine-readable report from separate Stage A",
    )
    parser.add_argument("--output-dir", required=True, help="fresh job-specific directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-full-package-rehash",
        action="store_true",
        help="development only; Slurm must leave the complete package audit enabled",
    )
    args = parser.parse_args()
    result = run_p6_original_causal(
        args.config,
        args.output_dir,
        device=args.device,
        verify_all_artifact_hashes=not args.skip_full_package_rehash,
        stage=args.stage,
        preflight_report=args.preflight_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    expected = (
        "GPU_PREFLIGHT_GO" if args.stage == "preflight" else "JOB_COMPLETED_UNVERIFIED"
    )
    if result["status"] != expected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Audit or execute the fail-closed ASPD 100M risk-attribution stages."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.risk_attribution import (
    audit_risk_attribution_contract,
    finalize_readout_validation,
    run_attribution_screening,
    run_exact_intervention_stage,
    run_readout_measurement,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Frozen risk-attribution YAML contract",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "readout",
            "finalize-readout",
            "screen",
            "exact-feasibility",
            "exact-pilot",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--readout-validation-report")
    parser.add_argument("--readout-measurement-dir")
    parser.add_argument("--independent-review-file")
    parser.add_argument("--screening-dir")
    parser.add_argument("--feasibility-report")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.stage == "audit":
        if args.readout_validation_report is not None:
            parser.error("audit does not accept --readout-validation-report")
        report = audit_risk_attribution_contract(args.config, output=args.output)
    elif args.stage == "readout":
        if args.readout_validation_report is not None:
            parser.error("readout does not accept --readout-validation-report")
        report = run_readout_measurement(
            args.config,
            args.output,
            device=args.device,
        )
    elif args.stage == "finalize-readout":
        if args.readout_measurement_dir is None or args.independent_review_file is None:
            parser.error(
                "finalize-readout requires --readout-measurement-dir and "
                "--independent-review-file"
            )
        report = finalize_readout_validation(
            args.config,
            args.readout_measurement_dir,
            args.independent_review_file,
            args.output,
        )
    elif args.stage == "screen":
        if args.readout_validation_report is None:
            parser.error("screen requires --readout-validation-report")
        report = run_attribution_screening(
            args.config,
            args.readout_validation_report,
            args.output,
            device=args.device,
        )
    else:
        if args.readout_validation_report is None or args.screening_dir is None:
            parser.error(
                f"{args.stage} requires --readout-validation-report and --screening-dir"
            )
        if args.stage == "exact-pilot" and args.feasibility_report is None:
            parser.error("exact-pilot requires --feasibility-report")
        report = run_exact_intervention_stage(
            args.config,
            args.readout_validation_report,
            args.screening_dir,
            args.output,
            device=args.device,
            feasibility_only=args.stage == "exact-feasibility",
            feasibility_report=args.feasibility_report,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

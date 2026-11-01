"""Run the frozen RCCAS Stage 4 audit, preflight, held-out grid, or analysis."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.risk_heldout import (
    analyze_heldout_confirmation,
    run_heldout_confirmation,
    verify_heldout_preflight,
    write_decision_flip_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("audit-stage3", "preflight", "run", "analyze"))
    parser.add_argument("--config")
    parser.add_argument("--contract")
    parser.add_argument("--readout-validation-report")
    parser.add_argument("--screening-dir")
    parser.add_argument("--stage3-report")
    parser.add_argument("--stage3-raw-report")
    parser.add_argument("--stage3-records")
    parser.add_argument("--records")
    parser.add_argument("--output")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-markdown")
    parser.add_argument("--stage3-metrics-csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.stage == "audit-stage3":
        if not args.stage3_records or not args.output_csv or not args.output_markdown:
            parser.error("audit-stage3 requires --stage3-records, --output-csv, and --output-markdown")
        report = write_decision_flip_audit(
            args.stage3_records, args.output_csv, args.output_markdown
        )
    elif args.stage == "analyze":
        if not args.records or not args.contract or not args.output:
            parser.error("analyze requires --records, --contract, and --output")
        report = analyze_heldout_confirmation(
            args.records, args.contract, args.output,
            stage3_metrics_csv=args.stage3_metrics_csv,
        )
    else:
        required = {
            "config": args.config,
            "contract": args.contract,
            "readout-validation-report": args.readout_validation_report,
            "screening-dir": args.screening_dir,
            "stage3-report": args.stage3_report,
            "stage3-raw-report": args.stage3_raw_report,
            "stage3-records": args.stage3_records,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"{args.stage} missing required arguments: {', '.join(missing)}")
        if args.stage == "preflight":
            report = verify_heldout_preflight(
                args.config, args.contract, args.readout_validation_report,
                args.screening_dir, args.stage3_report, args.stage3_raw_report,
                args.stage3_records, output=args.output,
            )
        else:
            if not args.output:
                parser.error("run requires --output")
            report = run_heldout_confirmation(
                args.config, args.contract, args.readout_validation_report,
                args.screening_dir, args.stage3_report, args.stage3_raw_report,
                args.stage3_records, args.output, device=args.device,
            )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

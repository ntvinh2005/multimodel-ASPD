"""Serve or verify the local Multi-model ASPD P1--P5 research dashboard."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.dashboard import AnalysisDashboardStore, build_dashboard_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir")
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--notes")
    parser.add_argument("--low-support-threshold", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.low_support_threshold < 1:
        parser.error("--low-support-threshold must be positive")

    store = AnalysisDashboardStore(
        args.analysis_dir,
        config_path=args.config,
        run_dir=args.run_dir,
        notes_path=args.notes,
        low_support_threshold=args.low_support_threshold,
    )
    print(json.dumps(store.verification_summary(), indent=2, sort_keys=True), flush=True)
    if args.verify_only:
        return

    import uvicorn

    app = build_dashboard_app(store)
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

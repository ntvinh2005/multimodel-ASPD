"""Serve or verify the local Multi-model ASPD P1--P5 research dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.crosscoder_dashboard import (
    CrosscoderDashboardStore,
    ResearchComparisonStore,
)
from aspd.multimodel.dashboard import AnalysisDashboardStore, build_dashboard_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir")
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--notes")
    parser.add_argument(
        "--crosscoder-analysis",
        help="Crosscoder analysis directory; auto-discovers the completed R1 path when present",
    )
    parser.add_argument(
        "--research-annotations",
        help="Tracked method-specific post-ranking annotation JSON",
    )
    parser.add_argument(
        "--research-comparison",
        help="Tracked phenomenon-level Crosscoder/ASPD comparison JSON",
    )
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
    repo_root = Path(__file__).resolve().parents[3]
    default_crosscoder = (
        repo_root / "out/multimodel/baselines/crosscoder/r1_qwen3_full/analysis"
    )
    default_annotations = (
        repo_root / "docs/multimodel/results/qwen3_general_diff_annotations.json"
    )
    default_comparison = (
        repo_root / "docs/multimodel/results/qwen3_crosscoder_aspd_phenomena.json"
    )
    crosscoder_path = Path(args.crosscoder_analysis) if args.crosscoder_analysis else default_crosscoder
    annotations_path = (
        Path(args.research_annotations) if args.research_annotations else default_annotations
    )
    comparison_path = (
        Path(args.research_comparison) if args.research_comparison else default_comparison
    )
    crosscoder_store = (
        CrosscoderDashboardStore(crosscoder_path, annotations_path=annotations_path)
        if crosscoder_path.exists()
        else None
    )
    comparison_store = (
        ResearchComparisonStore(
            comparison_path,
            aspd_store=store,
            crosscoder_store=crosscoder_store,
            annotations_path=annotations_path,
        )
        if comparison_path.exists()
        else None
    )
    verification = store.verification_summary()
    verification["crosscoder"] = (
        crosscoder_store.verification_summary()
        if crosscoder_store is not None
        else {
            "available": False,
            "analysis_dir": str(crosscoder_path.resolve()),
            "diagnostics": ["Crosscoder analysis directory not found"],
        }
    )
    verification["research_comparison"] = (
        comparison_store.verification_summary()
        if comparison_store is not None
        else {
            "available": False,
            "path": str(comparison_path.resolve()),
            "diagnostics": ["Structured research comparison artifact not found"],
        }
    )
    print(json.dumps(verification, indent=2, sort_keys=True), flush=True)
    if args.verify_only:
        return

    import uvicorn

    app = build_dashboard_app(
        store,
        crosscoder_store=crosscoder_store,
        comparison_store=comparison_store,
    )
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

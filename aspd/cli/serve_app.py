"""Serve the circuit app: the component browser and per-prompt attribution graphs of one run.

    python -m aspd.cli.serve_app --run gpt2_all_aspd [--port 8055]

Graphs are computed on demand on the GPU and cached under `$PARAM_DECOMP_OUT_DIR/app`. Two
attribution methods, chosen per request with `?method=`: `lab` (gradients on the component model)
and `err` (the exact forward, with error nodes for what the decomposition does not explain).
Environment: `LM_INTERP_NODE_CAP` (components drawn, default 500), `LM_INTERP_MAX_DENSITY` (drop
components denser than this before the cap, default 1.0 = off), `LM_INTERP_COMPONENT_TARGETS`
(components the `err` method explains, default 0 = output targets only).
"""

import argparse
import os
from pathlib import Path

import aspd
from aspd.analysis.app.methods import install_all_methods
from aspd.analysis.app.patch import (
    install_attribution_mask_is_ci,
    install_component_data_cache,
    install_graph_edge_cap,
    install_node_ranking_api,
    install_output_influence_pruning,
    install_sparse_node_ci_vals,
    install_stored_graph_cache,
)
from aspd.lab_compat import widen_lab_config_parsing

NODE_CAP = int(os.environ.get("LM_INTERP_NODE_CAP", "500"))
MAX_DENSITY = float(os.environ.get("LM_INTERP_MAX_DENSITY", "1.0"))
# Read at import time by aspd.analysis.circuits.lab_api; echoed here so the banner can show them.
COMPONENT_TARGETS = int(os.environ.get("LM_INTERP_COMPONENT_TARGETS", "0"))
EDGES_PER_TARGET = int(os.environ.get("LM_INTERP_EDGES_PER_TARGET", "64"))

widen_lab_config_parsing()

import param_decomp_lab
import uvicorn
from fastapi.responses import FileResponse
from param_decomp_lab.app.backend.server import app

install_graph_edge_cap()
install_sparse_node_ci_vals()
install_attribution_mask_is_ci()
install_output_influence_pruning(node_cap=NODE_CAP, max_density=MAX_DENSITY)
install_node_ranking_api()
install_stored_graph_cache()
install_component_data_cache()

# Last, so its `save_graph` wrapper is outermost: it stamps the row id the rest of the chain made.
install_all_methods()

_FORK_DIST = Path(aspd.__file__).resolve().parent.parent / "frontend" / "dist"
_STOCK_DIST = Path(param_decomp_lab.__file__).parent / "app" / "frontend" / "dist"
_DIST = _FORK_DIST if _FORK_DIST.exists() else _STOCK_DIST


def open_on_start(run_id: str) -> str:
    """Link `runs/<p-id>` to the run (the id the lab addresses runs by) and load it at startup."""
    from param_decomp_lab.app.backend import server

    from aspd.cli.harvest import link_downstream_id
    from aspd.paths import RUNS_DIR

    run_dir = RUNS_DIR / run_id
    assert (run_dir / "experiment_config.yaml").exists(), f"no run at {run_dir}"
    did = link_downstream_id(run_dir)
    server.PARAM_DECOMP_APP_DEFAULT_RUN = f"local/local/{did}"
    return did


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run id under $PARAM_DECOMP_OUT_DIR/runs to open at startup")
    ap.add_argument("--port", type=int, default=8055)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    if args.run is not None:
        print(f"[serve_app] opening {args.run} as local/local/{open_on_start(args.run)}", flush=True)
    assert _DIST.exists(), f"frontend not built: {_DIST} missing (see module docstring)"
    if _DIST is _STOCK_DIST:
        print("[serve_app] WARNING: serving the stock frontend; run frontend/build.sh", flush=True)
    index = _DIST / "index.html"

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)

    print(f"[serve_app] UI+API on http://{args.host}:{args.port}  (dist={_DIST})", flush=True)
    print(
        f"[serve_app] methods: ?method=lab (default) | ?method=err  "
        f"component targets={COMPONENT_TARGETS} edges/target={EDGES_PER_TARGET}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

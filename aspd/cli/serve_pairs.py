"""Serve the pair viewer and the per-prompt page for one decomposition run."""

import argparse
import os
import re
from pathlib import Path

import yaml

from aspd.paths import RUNS_DIR

_STEP_RE = re.compile(r"^model_(\d+)\.pth$")


def latest_checkpoint(run_dir: Path, step: int | None) -> Path:
    if step is not None:
        path = run_dir / f"model_{step}.pth"
        assert path.exists(), f"no checkpoint {path}"
        return path
    steps = [int(m.group(1)) for p in run_dir.iterdir() if (m := _STEP_RE.match(p.name))]
    assert steps, f"no model_*.pth in {run_dir}"
    return run_dir / f"model_{max(steps)}.pth"


def latest_harvest(run_dir: Path, sub: str | None) -> Path | None:
    """The harvest db to serve, or None -- a run with no harvest still serves pair scores."""
    root = run_dir / "harvest"
    if not root.exists():
        return None
    if sub is not None:
        db = root / sub / "harvest.db"
        assert db.exists(), f"no harvest db {db}"
        return db
    subs = sorted(p for p in root.iterdir() if (p / "harvest.db").exists())
    return (subs[-1] / "harvest.db") if subs else None


def run_target(run_dir: Path) -> tuple[str, str]:
    """`(model_name, tokenizer_name)` from the run's config, parsed as plain YAML."""
    cfg = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
    model_name = cfg["target"]["spec"]["params"]["model_name"]
    return model_name, cfg["data"]["tokenizer_name"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id under $PARAM_DECOMP_OUT_DIR/runs")
    ap.add_argument("--runs-root", default=str(RUNS_DIR))
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--harvest-sub", default=None, help="harvest subrun id (default: latest)")
    ap.add_argument("--interp-db", default=None, help="autointerp db to read labels from / write to")
    ap.add_argument("--pair-db", default=None, help="sidecar db of data-pass pair metrics")
    ap.add_argument("--kappa", default=None,
                    help="co-activation coefficients for dot_coact "
                         "(default: <run>/harvest/pair_coactivation.pt, if it exists; "
                         "'none' to serve without it)")
    ap.add_argument("--device", default="cpu",
                    help="device for the prompt tab's ComponentModel (loaded lazily, on the "
                         "first /api/prompt request -- the pair viewer itself loads no model)")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--threads", type=int, default=32, help="torch intra-op threads")
    ap.add_argument("--sae-cache-bytes", type=int, default=6 << 30,
                    help="in-memory budget for loaded SAE direction blocks")
    ap.add_argument("--qk-edit", choices=("gated", "weight"), default="gated",
                    help="what 'without this pair' means on the QK panel: 'gated' subtracts the "
                         "pair's own term (what it contributed, where both gates were open), "
                         "'weight' deletes the two rank-1 components from Q and K and recomputes "
                         "the score. Per-request `edit=` overrides it.")
    ap.add_argument("--no-neuronpedia", action="store_true",
                    help="do not look up feature labels/examples on neuronpedia.org")
    args = ap.parse_args()

    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from aspd.analysis.pairs.features import SaeStore, TargetNorms, catalogue
    from aspd.analysis.pairs.server import build_app
    from aspd.analysis.pairs.store import HarvestStore, InterpStore, KappaStore, PairScoreStore
    from aspd.analysis.pairs.weights import RunWeights
    from aspd.analysis.prompt.api import install_prompt_api
    from aspd.analysis.prompt.engine import PromptEngine
    from aspd.eval.tokens import decode_with_spaces

    torch.set_num_threads(min(args.threads, os.cpu_count() or args.threads))

    run_dir = Path(args.runs_root) / args.run
    assert run_dir.exists(), f"no run dir {run_dir}"
    model_name, tokenizer_name = run_target(run_dir)
    ckpt = latest_checkpoint(run_dir, args.step)
    harvest_db = latest_harvest(run_dir, args.harvest_sub)

    kappa_path = (None if args.kappa == "none"
                  else Path(args.kappa) if args.kappa
                  else run_dir / "harvest" / "pair_coactivation.pt")

    print(f"[pairs] run {args.run}  model {model_name}", flush=True)
    print(f"[pairs] checkpoint {ckpt}", flush=True)
    print(f"[pairs] harvest {harvest_db or '(none — scores only)'}", flush=True)

    decode = decode_with_spaces(AutoTokenizer.from_pretrained(tokenizer_name))
    weights = RunWeights(ckpt, model_name)
    print(f"[pairs] {len(weights.spaces)} decomposed matrices", flush=True)

    kappa = KappaStore(kappa_path if kappa_path and kappa_path.exists() else None)
    if kappa.available:
        m = kappa.meta
        print(f"[pairs] kappa {kappa_path.name}: {m.get('n_pairs')} module pairs, "
              f"{'every component' if m.get('full_pool') else str(m.get('per_module')) + '/module'}, "
              f"{m.get('n_tokens', 0):,} tokens"
              + (f", {m.get('n_shared_gate_pairs')} share an encoder"
                 if m.get('n_shared_gate_pairs') else ""), flush=True)
    else:
        print("[pairs] kappa (none — dot_coact unavailable; run slurm/app_coactivation.sbatch)",
              flush=True)

    saes = SaeStore(model_name, cache_bytes=args.sae_cache_bytes)
    norms = TargetNorms(model_name)
    n_rel = len(catalogue(model_name)) if saes.available else 0
    print(f"[pairs] SAE releases for {model_name}: {n_rel or 'none configured — weight × weight only'}",
          flush=True)

    harvest = HarvestStore(harvest_db, decode)
    app = build_app(
        run_name=args.run,
        weights=weights,
        harvest=harvest,
        interp=InterpStore(Path(args.interp_db) if args.interp_db else None),
        pair_scores=PairScoreStore(Path(args.pair_db) if args.pair_db else None),
        kappa=kappa,
        saes=saes,
        norms=norms,
        neuronpedia=not args.no_neuronpedia,
    )
    install_prompt_api(app, PromptEngine(run_dir, args.step, args.device), args.run,
                       saes=saes, norms=norms, harvest=harvest, qk_edit=args.qk_edit)

    print(f"[pairs] UI+API on http://{args.host}:{args.port}", flush=True)
    print(f"[pairs] prompt trace on http://{args.host}:{args.port}/prompt "
          f"(loads the model on first use, device={args.device})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

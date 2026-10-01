"""Serve the SAE latent viewer over a `harvest.db`."""

import argparse
from pathlib import Path


def main() -> None:
    import uvicorn
    from transformers import AutoTokenizer

    from aspd.config import LMInterpExperimentConfig
    from aspd.eval.feature_server import build_app
    from aspd.eval.harvest_path import resolve_harvest_db
    from aspd.eval.tokens import decode_with_spaces
    from aspd.sae.config import dictionary_experiment_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True, help="dir containing harvest.db")
    ap.add_argument("--sae-config", required=True, help="configs/sae/<target>.yaml or configs/transcoder/<target>.yaml (tokenizer only)")
    ap.add_argument("--interp-db", default=None, help="optional autointerp interp.db for labels")
    ap.add_argument("--logit-lens-json", default=None, help="optional logit_lens.json overlay")
    ap.add_argument("--port", type=int, default=8056)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n-intervals", type=int, default=5)
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    cfg = LMInterpExperimentConfig.from_file(dictionary_experiment_config(args.sae_config))
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)

    app = build_app(
        resolve_harvest_db(args.harvest_dir),
        decode_with_spaces(tok),
        interp_db=Path(args.interp_db) if args.interp_db else None,
        logit_lens_json=Path(args.logit_lens_json) if args.logit_lens_json else None,
        n_intervals=args.n_intervals,
        window=args.window,
    )

    import socket

    node = socket.gethostname()
    print(f"[serve] http://{args.host}:{args.port}  (SAE feature viewer)", flush=True)
    print(f"[serve] tunnel:  ssh -N -L {args.port}:localhost:{args.port} {node}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Intruder scores of an evaluation SAE's latents."""

import argparse
from pathlib import Path


def main() -> None:
    from param_decomp_lab.harvest.db import HarvestDB

    from aspd.config import LMInterpExperimentConfig
    from aspd.eval.harvest_path import resolve_harvest_db
    from aspd.eval.intruder_db import (
        eligible_density_entries,
        intruder_harvest_db,
    )
    from aspd.eval.judge import OpenAICompatProvider, add_judge_args, judge_config_from_args
    from aspd.sae.config import dictionary_experiment_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True, help="dir containing harvest.db")
    ap.add_argument("--sae-config", required=True, help="configs/sae/<target>.yaml or configs/transcoder/<target>.yaml (tokenizer only)")
    ap.add_argument("--sites", nargs="+", default=None,
                    help="site prefixes to score (default: all sites found in the db)")
    ap.add_argument("--n-subsample", type=int, default=200, help="cap per site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cross-site-donors", action="store_true",
                    help="draw intruder donors from all sites instead of the scored site only")
    add_judge_args(ap)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--density-tolerance", type=float, default=0.05)
    ap.add_argument("--window-tokens-per-side", type=int, default=None,
                    help="crop each example the judge sees to this half-window, centred on its "
                         "peak firing token (20 = parity with aspd.cli.harvest's stored windows); "
                         "default: show the stored example whole. Scores land under "
                         "score_type='intruder_w<n>' in a separate intruder_summary_*_w<n>.json")
    ap.add_argument("--cost-limit-usd", type=float, default=None)
    args = ap.parse_args()

    cfg = LMInterpExperimentConfig.from_file(dictionary_experiment_config(args.sae_config))
    tokenizer_name = cfg.data.tokenizer_name
    harvest_dir = Path(args.harvest_dir)
    harvest_db = resolve_harvest_db(harvest_dir)

    if args.sites is None:
        db = HarvestDB(harvest_db, readonly=True)
        entries = eligible_density_entries(db, min_examples=args.n_real + 1)
        db.close()
        sites = sorted({k.rsplit(":", 1)[0] for k, _ in entries})
    else:
        sites = args.sites
    assert sites, f"no eligible sites in {harvest_dir}"

    llm_config = judge_config_from_args(args)
    provider = OpenAICompatProvider(llm_config)

    for site in sites:
        print(f"[sae-intruder] === site {site} ===", flush=True)
        intruder_harvest_db(
            harvest_dir,
            tokenizer_name,
            llm_config=llm_config,
            n_real=args.n_real,
            n_trials=args.n_trials,
            density_tolerance=args.density_tolerance,
            n_subsample=args.n_subsample,
            seed=args.seed,
            key_prefix=f"{site}:",
            restrict_donors_to_prefix=not args.cross_site_donors,
            window_tokens_per_side=args.window_tokens_per_side,
            cost_limit_usd=args.cost_limit_usd,
            provider=provider,
        )


if __name__ == "__main__":
    main()

"""Multiple-feature weight editing of a run against the output SAE."""

import argparse
from pathlib import Path

import torch

from aspd.cli.editing import (
    annotate_transcoder_report,
    default_feature_harvest,
    parse_top_k,
)
from aspd.eval.adapters.transcoder import resolve_transcoder_steps
from aspd.eval.artifacts import (
    experiment_config_from_sae_dir,
    load_experiment_config,
    module_from_sae_dir,
)
from aspd.eval.batches import collect_batches, stored_stream
from aspd.eval.editing import measure
from aspd.eval.editing.multi import (
    SETUPS,
    MultiConfig,
    default_control_combos,
    default_top_k,
    prepare_multi,
    run_step_multi,
)
from aspd.eval.editing.report import (
    MultiAttrEditReport,
    summarize_multi,
    write_multi_report,
    write_multi_rows_csv,
    write_multi_sweep,
)
from aspd.eval.editing.sample import (
    DEFAULT_MAX_DENSITY,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_N_COMBINATIONS,
    DEFAULT_N_FEATURES,
    DEFAULT_POOL_SIZE,
    DEFAULT_SEED,
    dictionary_fingerprint,
    draw_combinations,
    draw_pool,
    draw_sample,
    harvest_densities,
    load_pool,
    load_sample,
    write_pool,
    write_sample,
)
from aspd.eval.plots import safe_plot
from aspd.eval.plots.editing_multi import plot_arm_dir


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="the decomposition. Required unless --pool-only")
    ap.add_argument("--sae-dir", required=True,
                    help="the frozen pair; its OUTPUT half supplies the features and the pool")
    ap.add_argument("--m", type=int, default=5,
                    help="output features targeted at once (m = |J|)")
    ap.add_argument("--setup", default="cond", choices=list(SETUPS),
                    help="'cond' averages the attribution over A_j; 'global' over every "
                         "unmasked token. Rank-1 arms and transcoders only for 'global'")
    ap.add_argument("--steps", default="all",
                    help="'all' or a list separated by , : or space. COLONS under sbatch --export")
    ap.add_argument("--step", type=int, default=None, help="single-checkpoint shorthand")
    ap.add_argument("--pool", default=None,
                    help="a feature_pool.json to reuse. Default: <sae-dir>/attr_edit/"
                         "feature_pool.json, drawn on first use and reused after")
    ap.add_argument("--features", default=None,
                    help="the sample that leads the pool. Default: <sae-dir>/attr_edit/"
                         "sampled_features.json")
    ap.add_argument("--pool-only", action="store_true",
                    help="draw the pool and exit; no decomposition is loaded")
    ap.add_argument("--redraw", action="store_true",
                    help="draw a fresh pool even if the file exists. Every arm already measured "
                         "against it becomes incomparable — say so in the log")
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--n-combinations", type=int, default=DEFAULT_N_COMBINATIONS)
    ap.add_argument("--n-control-combos", type=int, default=0,
                    help="combinations the CONTROLS are measured against; 0 picks 500/m, clamped "
                         "to [10, --n-combinations]. A cap below --n-combinations is logged and "
                         "recorded in the report, never silent")
    ap.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES,
                    help="size of the sample leading the pool, if it has to be drawn")
    ap.add_argument("--max-density", type=float, default=DEFAULT_MAX_DENSITY)
    ap.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    ap.add_argument("--feature-harvest", default=None,
                    help="harvest.db supplying firing_density. Default: discovered under --sae-dir")
    ap.add_argument("--top-k", default=None,
                    help="components per FEATURE; the edit removes their union. Default: "
                         "1:5:10:20:50 at m=1, 1:5:10 otherwise")
    ap.add_argument("--n-control-reps", type=int, default=5)
    ap.add_argument("--n-tokens", type=int, default=1_000_000)
    ap.add_argument("--n-tokens-global", type=int, default=200_000)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--estimator", default="auto", choices=["auto", "analytic", "autograd"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cache-device", default=None,
                    help="where the cached baseline activations live; default: the compute "
                         "device. A 600-feature pool caches up to the whole token budget "
                         "(~12 GB fp32) — pass 'cpu' if the GPU cannot hold it")
    ap.add_argument("--source", choices=["components", "transcoder"], default="components")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run-dir>/attr_edit_multi/m<m>_<setup>")
    ap.add_argument("--no-cache-module-inputs", action="store_true",
                   help="measure every edit through a full model forward instead of applying the "
                        "patched module to its cached input. ~60x slower and bitwise identical -- "
                        "the reference path --no-verify-cached-forward stops checking against")
    ap.add_argument("--no-verify-cached-forward", action="store_true",
                   help="skip the one-cell exact parity check between the two paths (~0.6%% of the "
                        "arm). Only worth it if the check has already been run on this target")
    return ap


def main() -> None:
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_target
    from transformers import AutoTokenizer

    from aspd.checkpoints import load_run_model, resolve_steps
    from aspd.eval.adapters.dictionary import assert_deterministic_encode
    from aspd.lab_compat import widen_lab_config_parsing
    from aspd.sae.setup import sites_for_run, train_or_load

    args = arg_parser().parse_args()
    widen_lab_config_parsing()
    assert args.run_dir or args.pool_only, "pass --run-dir, or --pool-only to just draw the pool"

    sae_dir = Path(args.sae_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None
    cfg_path = (run_dir / "experiment_config.yaml") if run_dir else \
        experiment_config_from_sae_dir(sae_dir)
    cfg_run = load_experiment_config(cfg_path)
    module = (cfg_run.pd.decomposition_targets[0].module_pattern if run_dir
              else module_from_sae_dir(sae_dir))
    device = get_device()
    cache_device = torch.device(args.cache_device) if args.cache_device else device

    target_model = build_target(cfg_run.target).to(device).eval()
    saes = train_or_load(target_model, module, iter([]), sae_dir, device=device)
    sites = sites_for_run(module, sae_dir)
    sae = saes["output"].eval()
    assert_deterministic_encode(sae)
    assert sites.output_site == module, (
        f"the pair's output site is {sites.output_site!r} but the decomposed module is {module!r}; "
        "the output SAE must read the decomposed module's own output"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg_run.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    top_k = parse_top_k(args.top_k) if args.top_k else default_top_k(args.m)
    cfg = MultiConfig(
        m=args.m,
        setup=args.setup,
        top_k=top_k,
        n_combinations=args.n_combinations,
        n_control_combos=args.n_control_combos,
        n_control_reps=args.n_control_reps,
        n_tokens_global=args.n_tokens_global,
        batch_size=args.batch_size,
        seed=args.seed,
        estimator=args.estimator,
        cache_module_inputs=not args.no_cache_module_inputs,
        verify_cached_forward=not args.no_verify_cached_forward,
    )

    batches, seen = collect_batches(
        cfg_run, split=args.split, device=str(device),
        n_tokens=args.n_tokens, batch_size=args.batch_size,
        stream=stored_stream(args.sae_dir),
    )
    plan = measure.build_token_plan(batches, tokenizer)
    del batches
    print(f"[attr_edit_multi] {plan.tokens.shape[0]} sequences · {plan.n_tokens_total:,} unmasked "
          f"of {seen:,} drawn tokens", flush=True)

    fingerprint = dictionary_fingerprint(sae.W_enc, sae.W_dec, sae.threshold)
    pool_path = Path(args.pool) if args.pool else sae_dir / "attr_edit" / "feature_pool.json"
    if pool_path.exists() and not args.redraw:
        pool = load_pool(pool_path, fingerprint=fingerprint, site=sites.output_site)
        print(f"[attr_edit_multi] reusing {pool_path}: {len(pool.feature_ids)} features "
              f"({pool.n_seed_features} from the sample, seed {pool.seed})", flush=True)
    else:
        pool = _draw_pool(args, sae_dir, sae, plan, sites, target_model, module,
                          fingerprint=fingerprint, device=device, pool_path=pool_path)
    pool.source["path"] = str(pool_path.resolve())

    if args.pool_only:
        print("[attr_edit_multi] --pool-only: nothing else to do")
        return

    combos = draw_combinations(
        pool, cfg.m, n_combinations=cfg.n_combinations, seed=cfg.seed
    )
    touched = sorted({j for c in combos for j in c})
    print(f"[attr_edit_multi] m={cfg.m} setup={cfg.setup}: {len(combos)} combinations over "
          f"{len(touched)} distinct features · top_k={list(cfg.top_k)} · controls on "
          f"{cfg.control_combos}/{cfg.n_combinations} combinations "
          f"(auto default {default_control_combos(cfg.m, cfg.n_combinations)})", flush=True)

    assert run_dir is not None
    out_dir = Path(args.out_dir) if args.out_dir else \
        run_dir / "attr_edit_multi" / f"m{cfg.m}_{cfg.setup}"
    steps = (
        [args.step]
        if args.step is not None
        else resolve_transcoder_steps(run_dir, args.steps)
        if args.source == "transcoder"
        else resolve_steps(run_dir, args.steps)
    )

    tc_sites = None
    if args.source == "transcoder":
        from aspd.sae.transcoder_setup import sites_for_transcoder

        tc_sites = sites_for_transcoder(module, run_dir)
        exact = tc_sites.input_take == "output"
        print(f"[attr_edit_multi] --source transcoder: encoder on {tc_sites.input_site} "
              f"(take={tc_sites.input_take}); edit_is_exact={exact}. The two uncorrected "
              "differences apply to every m unchanged", flush=True)
        assert args.estimator != "autograd" or exact, (
            "--estimator autograd recentres the mask on `y(g)` computed from the DECOMPOSED "
            "MODULE's input, which is not what this transcoder's encoder reads"
        )

    base = prepare_multi(
        target_model, module, sae, plan, pool, combos, cfg,
        device=device, cache_device=cache_device,
    )
    union_sizes = torch.tensor([float(c.positions.numel()) for c in base.combinations])
    print(f"[attr_edit_multi] baseline ready: {base.global_group.n_positions:,} global positions · "
          f"|A_J| median {int(union_sizes.median()):,}, max {int(union_sizes.max()):,}", flush=True)

    reports: list[MultiAttrEditReport] = []
    for step in steps:
        if tc_sites is not None:
            from aspd.eval.adapters.transcoder import TranscoderComponentModel
            from aspd.sae.transcoder import load_transcoder

            model = TranscoderComponentModel(
                load_transcoder(run_dir, step, device=str(device)),
                target_model, module, tc_sites, tokenizer,
            )
        else:
            model, _ = load_run_model(
                run_dir, step, str(device), target_model=target_model
            )
        report = run_step_multi(
            target_model, model, module, sae, base, cfg,
            step=step, run_dir=run_dir, sae_dir=sae_dir, site=sites.output_site, device=device,
        )
        if tc_sites is not None:
            annotate_transcoder_report(report, model, cfg)
        print(summarize_multi(report))
        write_multi_report(report, out_dir / f"attr_edit_multi_step{step}.json")
        write_multi_rows_csv(report, out_dir / f"attr_edit_multi_step{step}.csv")
        reports.append(report)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    write_multi_sweep(reports, out_dir / "sweep_attr_edit_multi.json")
    safe_plot(plot_arm_dir, out_dir)


def _draw_pool(args, sae_dir, sae, plan, sites, target_model, module, *,
               fingerprint, device, pool_path):
    if pool_path.exists():
        print(f"[attr_edit_multi] --redraw: {pool_path} is being REPLACED. Every arm already "
              "measured against the old pool describes different combinations", flush=True)
    harvest = Path(args.feature_harvest) if args.feature_harvest else \
        default_feature_harvest(sae_dir)
    density = harvest_densities(harvest, sites.output_site, sae.cfg.n_features)
    support = measure.feature_support(
        target_model, module, sae, plan, batch_size=args.batch_size, device=device
    )
    sample_path = Path(args.features) if args.features else \
        sae_dir / "attr_edit" / "sampled_features.json"
    if sample_path.exists():
        sample = load_sample(sample_path, fingerprint=fingerprint, site=sites.output_site)
    else:
        print(f"[attr_edit_multi] no {sample_path}; drawing the sample first", flush=True)
        sample = draw_sample(
            density=density, support=support, site=sites.output_site, fingerprint=fingerprint,
            n_features=args.n_features, seed=args.seed, max_density=args.max_density,
            min_support=args.min_support, n_tokens=plan.n_tokens_total,
            source={"harvest": str(harvest.resolve()), "sae_dir": str(sae_dir.resolve()),
                    "split": args.split, "path": str(sample_path.resolve())},
        )
        write_sample(sample, sample_path)
    pool = draw_pool(
        density=density, support=support, site=sites.output_site, fingerprint=fingerprint,
        seed_features=sample.feature_ids, pool_size=args.pool_size, seed=args.seed,
        max_density=args.max_density, min_support=args.min_support,
        n_tokens=plan.n_tokens_total,
        source={"harvest": str(harvest.resolve()), "sae_dir": str(sae_dir.resolve()),
                "split": args.split, "sample": str(sample_path.resolve()),
                "path": str(pool_path.resolve())},
    )
    write_pool(pool, pool_path)
    print(f"[attr_edit_multi] {pool.n_density_eligible} features pass the density band, "
          f"{pool.n_eligible} also reach support {args.min_support}", flush=True)
    return pool


if __name__ == "__main__":
    main()

"""Single-feature weight editing of a run against the output SAE."""

import argparse
from pathlib import Path

import torch

from aspd.eval.adapters.transcoder import resolve_transcoder_steps
from aspd.eval.artifacts import (
    experiment_config_from_sae_dir,
    load_experiment_config,
    module_from_sae_dir,
)
from aspd.eval.batches import collect_batches, stored_stream
from aspd.eval.editing import measure
from aspd.eval.editing.report import (
    AttrEditReport,
    summarize,
    write_report,
    write_rows_csv,
    write_sweep,
)
from aspd.eval.editing.run import DEFAULT_TOP_K, AttrEditConfig, prepare, run_step
from aspd.eval.editing.sample import (
    DEFAULT_MAX_DENSITY,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_N_FEATURES,
    DEFAULT_SEED,
    dictionary_fingerprint,
    draw_sample,
    harvest_densities,
    load_sample,
    write_sample,
)
from aspd.eval.plots import safe_plot
from aspd.eval.plots.editing import plot_dir


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="the decomposition. Required unless --sample-only")
    ap.add_argument("--sae-dir", required=True,
                    help="the frozen pair. Its OUTPUT half supplies the features being edited "
                         "toward, and its harvest supplies the density filter")
    ap.add_argument("--steps", default="all",
                    help="'all' or a list separated by , : or space. Use COLONS under "
                         "sbatch --export, which splits on commas")
    ap.add_argument("--step", type=int, default=None, help="single-checkpoint shorthand")
    ap.add_argument("--features", default=None,
                    help="a sampled_features.json to reuse. Default: <sae-dir>/attr_edit/"
                         "sampled_features.json, drawn on first use and reused after")
    ap.add_argument("--sample-only", action="store_true",
                    help="draw the feature sample and exit; no decomposition is loaded")
    ap.add_argument("--redraw", action="store_true",
                    help="draw a fresh sample even if the default file exists. Every arm already "
                         "measured against this pair becomes incomparable — say so in the log")
    ap.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    ap.add_argument("--max-density", type=float, default=DEFAULT_MAX_DENSITY,
                    help="upper end of the eligibility band; a latent above it is a bias "
                         "direction, not a feature")
    ap.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT,
                    help="|A_j| a feature must reach on THIS batch. Every reported mean is over "
                         "it; at GPT2-small's output dictionary the median latent reaches ~33")
    ap.add_argument("--feature-harvest", default=None,
                    help="harvest.db supplying firing_density. Default: discovered under --sae-dir")
    ap.add_argument("--top-k", default=":".join(str(k) for k in DEFAULT_TOP_K),
                    help="the nested edit sizes. Colons, not commas, under sbatch --export")
    ap.add_argument("--n-control-reps", type=int, default=5,
                    help="random / norm-matched draws per k. Shared across features, so each "
                         "draw is one extra edit rather than one per feature")
    ap.add_argument("--n-tokens", type=int, default=1_000_000,
                    help="evaluation tokens; A_j is collected over all of them")
    ap.add_argument("--n-tokens-global", type=int, default=200_000,
                    help="the fixed global sample the off-A_j effect and the collateral use")
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--batch-size", type=int, default=16,
                    help="sequences per forward, and the size of the [b*L, F] encode block")
    ap.add_argument("--estimator", default="auto", choices=["auto", "analytic", "autograd"],
                    help="'auto' is analytic on a rank-1 arm and autograd on every other one")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cache-device", default=None,
                    help="where the cached baseline activations live; default: the compute "
                         "device (~2.7 GB at the default budgets)")
    ap.add_argument("--source", choices=["components", "transcoder"], default="components",
                    help="`transcoder` treats --run-dir as a TRANSCODER directory and edits "
                         "`dW = sum_j W_dec[j] (x) W_enc[:, j]`. The estimator is analytic there "
                         "-- a transcoder's (W_enc, W_dec) IS core's rank-1 form -- and the "
                         "report records whether the edit is exact at that encoder site")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/attr_edit")
    ap.add_argument("--no-cache-module-inputs", action="store_true",
                    help="measure every edit through a full model forward instead of applying the "
                         "patched module to its cached input. Far slower and bitwise identical")
    ap.add_argument("--no-verify-cached-forward", action="store_true",
                    help="skip the one-cell exact parity check between the two paths")
    return ap


def parse_top_k(spec: str) -> tuple[int, ...]:
    import re

    values = tuple(int(s) for s in re.split(r"[,:\s]+", spec.strip()) if s)
    assert values, f"could not parse --top-k {spec!r}"
    assert values == tuple(sorted(values)), f"--top-k must be ascending; got {values}"
    return values


def annotate_transcoder_report(report, model, cfg) -> None:
    """Record the two ways a transcoder's edit differs from a component's. Neither is corrected."""
    from aspd.eval.adapters.transcoder import bias_residue

    ranked = [
        row for row in report.edits
        if row.kind == "ranked" and getattr(row, "block", "union") == "union"
    ]
    per_k: dict[str, dict[str, float]] = {}
    for k in cfg.top_k:
        rows = [row for row in ranked if row.k == k]
        if not rows:
            continue
        norms = [
            float(
                bias_residue(
                    model.transcoder,
                    torch.tensor(row.selection, device=model.transcoder.W_enc.device),
                ).norm()
            )
            for row in rows
        ]
        per_k[str(k)] = {
            "mean_residue_norm": sum(norms) / len(norms),
            "max_residue_norm": max(norms),
            "mean_edit_norm": sum(row.edit_norm for row in rows) / len(rows),
        }
    report.meta |= {
        "source": "transcoder",
        "encoder_site": model.sites.input_site,
        "encoder_take": model.sites.input_take,
        "edit_is_exact": model.edit_is_exact,
        "edit_exactness_note": (
            "encoder adjacent to the decomposed matrix; dW removes exactly what latent j wrote "
            "(up to bias_residue)"
            if model.edit_is_exact
            else "encoder is one LayerNorm UPSTREAM of the tensor c_fc.weight multiplies, so dW "
            "removes a different map built from the same numbers; compare against the adjacent "
            "transcoder rather than reading this in isolation"
        ),
        "bias_residue": per_k,
        "b_dec_norm": float(model.transcoder.b_dec.detach().norm()),
    }


def annotate_tc_arm_report(report, model, module_path: str, cfg) -> None:
    """The part of `annotate_transcoder_report` that PD Transcoder and ASPD runs also need."""
    from aspd.eval.editing.attribution import bias_residue_for

    components = model.components[module_path]
    if getattr(components, "b_dec", None) is None:
        return  # every non-transcoder arm

    ranked = [
        row for row in report.edits
        if row.kind == "ranked" and getattr(row, "block", "union") == "union"
    ]
    per_k: dict[str, dict[str, float]] = {}
    for k in cfg.top_k:
        rows = [row for row in ranked if row.k == k]
        if not rows:
            continue
        norms = [
            float(
                bias_residue_for(
                    components,
                    torch.tensor(row.selection, device=components.V.device),
                ).norm()
            )
            for row in rows
        ]
        per_k[str(k)] = {
            "mean_residue_norm": sum(norms) / len(norms),
            "max_residue_norm": max(norms),
            "mean_edit_norm": sum(row.edit_norm for row in rows) / len(rows),
        }
    report.meta |= {
        "component_arch": "transcoder",
        "bias_residue": per_k,
        "b_dec_norm": float(components.b_dec.detach().norm()),
    }


def default_feature_harvest(sae_dir: Path) -> Path:
    """The pair's `harvest.db`, by the rule `p2_matching` and `p2_plan` already use."""
    flat = sae_dir / "harvest" / "harvest.db"
    if flat.exists():
        return flat
    found = sorted((sae_dir / "harvest").glob("*/harvest.db"))
    assert len(found) == 1, (
        f"expected exactly one harvest.db under {sae_dir / 'harvest'}, found {len(found)}. It "
        "supplies the firing densities the eligibility band is read off; pass --feature-harvest"
    )
    return found[0]


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
    assert args.run_dir or args.sample_only, "pass --run-dir, or --sample-only to just draw"

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

    cfg = AttrEditConfig(
        top_k=parse_top_k(args.top_k),
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
    print(f"[attr_edit] {plan.tokens.shape[0]} sequences · {plan.n_tokens_total:,} unmasked of "
          f"{seen:,} drawn tokens", flush=True)

    fingerprint = dictionary_fingerprint(sae.W_enc, sae.W_dec, sae.threshold)
    sample_path = Path(args.features) if args.features else \
        sae_dir / "attr_edit" / "sampled_features.json"
    if sample_path.exists() and not args.redraw:
        sample = load_sample(sample_path, fingerprint=fingerprint, site=sites.output_site)
        print(f"[attr_edit] reusing {sample_path}: {len(sample.feature_ids)} features "
              f"(seed {sample.seed}, support floor {sample.min_support})", flush=True)
    else:
        if sample_path.exists():
            print(f"[attr_edit] --redraw: {sample_path} is being REPLACED. Every result already "
                  "measured against the old sample describes different features", flush=True)
        harvest = Path(args.feature_harvest) if args.feature_harvest else \
            default_feature_harvest(sae_dir)
        density = harvest_densities(harvest, sites.output_site, sae.cfg.n_features)
        support = measure.feature_support(
            target_model, module, sae, plan, batch_size=args.batch_size, device=device
        )
        sample = draw_sample(
            density=density, support=support, site=sites.output_site, fingerprint=fingerprint,
            n_features=args.n_features, seed=args.seed, max_density=args.max_density,
            min_support=args.min_support, n_tokens=plan.n_tokens_total,
            source={"harvest": str(harvest.resolve()), "sae_dir": str(sae_dir.resolve()),
                    "split": args.split, "path": str(sample_path.resolve())},
        )
        write_sample(sample, sample_path)
        print(f"[attr_edit] {sample.n_density_eligible} features pass the density band, "
              f"{sample.n_eligible} also reach support {args.min_support}", flush=True)
    sample.source["path"] = str(sample_path.resolve())

    if args.sample_only:
        print("[attr_edit] --sample-only: nothing else to do")
        return

    assert run_dir is not None
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "attr_edit"
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
        print(
            f"[attr_edit] --source transcoder: encoder on {tc_sites.input_site} "
            f"(take={tc_sites.input_take}); edit_is_exact={exact}."
            + (
                ""
                if exact
                else " The encoder reads one LayerNorm UPSTREAM of the tensor c_fc.weight "
                "multiplies, so `W_dec[j] (x) W_enc[:, j]` removes a different map built from "
                "the same numbers. It is run and reported with edit_is_exact=false; read it "
                "against the adjacent transcoder's faithfulness panel, not on its own."
            ),
            flush=True,
        )
        assert args.estimator != "autograd" or exact, (
            "--estimator autograd recentres the mask on `y(g)` computed from the DECOMPOSED "
            "MODULE's input, which is not what this transcoder's encoder reads. Use the analytic "
            "path (the default here: a transcoder's components are core's own rank-1 class)."
        )

    base = prepare(
        target_model, module, sae, plan, sample, cfg,
        device=device, cache_device=cache_device,
    )
    print(f"[attr_edit] baseline ready: {sum(p.numel() for p in base.positions.values()):,} A_j "
          f"positions, {base.global_group.n_positions:,} global positions", flush=True)

    reports: list[AttrEditReport] = []
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
        report = run_step(
            target_model, model, module, sae, base, cfg,
            step=step, run_dir=run_dir, sae_dir=sae_dir, site=sites.output_site, device=device,
        )
        if tc_sites is not None:
            annotate_transcoder_report(report, model, cfg)
        else:
            annotate_tc_arm_report(report, model, module, cfg)
        print(summarize(report))
        write_report(report, out_dir / f"attr_edit_step{step}.json")
        write_rows_csv(report, out_dir / f"attr_edit_step{step}.csv")
        reports.append(report)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    write_sweep(reports, out_dir / "sweep_attr_edit.json")
    safe_plot(plot_dir, out_dir)


if __name__ == "__main__":
    main()

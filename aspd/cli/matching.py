"""Meaning localization (matching) of a run's components against the output SAE."""

import argparse
from pathlib import Path

import torch
from param_decomp_lab.harvest.db import HarvestDB

from aspd.eval.adapters.component import VPDComponentSource
from aspd.eval.adapters.transcoder import resolve_transcoder_steps
from aspd.eval.batches import collect_batches, stored_stream
from aspd.eval.judge import add_judge_args, judge_config_from_args
from aspd.eval.matching import DEFAULT_MODE, MODES
from aspd.eval.matching.examples import example_counts
from aspd.eval.matching.run import (
    accumulate_alignment,
    assert_input_pair_adjacent,
    build_prompts_c2o,
    build_prompts_i2o,
    input_hook_site,
    run_judging,
    select_pairings,
    write_report,
)
from aspd.eval.plots import plot_matching_dir, safe_plot


def assert_harvest_provenance(harvest: Path, run_dir: Path, step: int) -> None:
    """The component harvest must belong to THIS run and THIS step."""
    assert harvest.is_relative_to(run_dir), (
        f"--component-harvest {harvest} is not under --run-dir {run_dir}"
    )
    stamped = [p.name for p in harvest.parents if p.name.startswith("h-step")]
    assert stamped, (
        f"{harvest} has no `h-step<N>` directory in its path, so its checkpoint step cannot be "
        "verified against --step"
    )
    found = int(stamped[0].removeprefix("h-step"))
    assert found == step, (
        f"--step {step} but the harvest is {stamped[0]} (step {found}); the alignment would be "
        "computed from one checkpoint and the examples read from another"
    )


def default_feature_harvest(sae_dir: Path) -> Path:
    """The output dictionary's `harvest.db` inside `--sae-dir`, discovered rather than declared."""
    flat = sae_dir / "harvest" / "harvest.db"
    if flat.exists():
        return flat
    found = sorted((sae_dir / "harvest").glob("*/harvest.db"))
    assert found, (
        f"no harvest.db under {sae_dir / 'harvest'}, so the output dictionary's features have no "
        "captions to judge. Harvest the pair first, or pass --feature-harvest explicitly"
    )
    assert len(found) == 1, (
        f"{len(found)} candidate harvests under {sae_dir / 'harvest'} ({[str(p) for p in found]}); "
        "pass --feature-harvest to say which dictionary's features to judge"
    )
    return found[0]


def default_component_harvest(run_dir: Path, step: int) -> Path:
    """`<run-dir>/harvest/h-step<NNNNNN>/harvest.db`, the layout the harvest pipeline writes."""
    return run_dir / "harvest" / f"h-step{step:06d}" / "harvest.db"


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", default="all",
                    help="'all' (every model_<step>.pth in the run dir) or a list separated by "
                         ", : or space. Use COLONS under sbatch --export, which splits on commas")
    ap.add_argument("--step", type=int, default=None, help="single-checkpoint shorthand")
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--component-harvest", default=None,
                    help="the RUN's harvest.db; derived per step by default, and only accepted "
                         "explicitly for a single-step run")
    ap.add_argument("--feature-harvest", default=None,
                    help="the OUTPUT dictionary's harvest.db; default: found under --sae-dir")
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--n-tokens", type=int, default=1_000_000, help="evaluation tokens for the effect matrix")
    ap.add_argument("--n-subsample", type=int, default=200,
                    help="pairs judged per configuration; the reported mean depends on it")
    ap.add_argument("--min-examples", type=int, default=10,
                    help="harvested windows required on BOTH sides for a pair to be judged")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["components", "transcoder"], default="components",
                    help="`transcoder` treats --run-dir as a TRANSCODER directory and matches its "
                         "latents against the output dictionary. Same alignment, same judge, same "
                         "control, using its (W_enc, W_dec) as (V, U)")
    ap.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE,
                    help="`c2o` (default) judges the COMPONENT against its matched output "
                         "feature; `i2o` chains an input feature on and judges the two "
                         "dictionary latents. Reports are named per mode, so both can coexist")
    ap.add_argument("--component-activation-key", default=None,
                    choices=["causal_importance", "component_activation", "activation"],
                    help="which harvested series the judge sees for a component on `c2o`. Default "
                         "follows --source: `causal_importance` for a decomposition (matching "
                         "p2.component_feature_server, so the judge reads the same object the app "
                         "shows) and `activation` for a transcoder, whose harvest is written by "
                         "`harvest_dictionaries` and stores a dictionary's own feature value")
    ap.add_argument("--feature-activation-key", default="activation",
                    help="the harvested series the judge reads for a DICTIONARY latent -- the "
                         "output feature in both modes, and the input feature on `i2o`")
    ap.add_argument("--dry-run", action="store_true",
                    help="stop after building prompts and dump the first few; validates the "
                         "whole pipeline without standing up the judge")
    add_judge_args(ap)
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/matching")
    return ap


def main() -> None:
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_target
    from transformers import AutoTokenizer

    from aspd.checkpoints import load_run_model, resolve_steps
    from aspd.eval.artifacts import load_experiment_config
    from aspd.lab_compat import widen_lab_config_parsing
    from aspd.sae.setup import sites_for_run, train_or_load

    args = arg_parser().parse_args()
    widen_lab_config_parsing()

    run_dir = Path(args.run_dir)
    cfg = load_experiment_config(run_dir / "experiment_config.yaml")
    module = cfg.pd.decomposition_targets[0].module_pattern
    device = get_device()
    out_dir = Path(args.out_dir or run_dir / "matching")
    is_transcoder = args.source == "transcoder"
    chained = args.mode == "i2o"
    activation_key = args.component_activation_key or (
        "activation" if is_transcoder else "causal_importance"
    )
    steps = (
        [args.step]
        if args.step is not None
        else resolve_transcoder_steps(run_dir, args.steps)
        if is_transcoder
        else resolve_steps(run_dir, args.steps)
    )
    assert args.component_harvest is None or len(steps) == 1, (
        f"--component-harvest names one harvest but {len(steps)} steps were requested; the "
        "alignment and the judged examples would come from different checkpoints. Drop it and "
        "let each step derive its own, or sweep one step at a time."
    )

    target_model = build_target(cfg.target).to(device).eval()
    saes = train_or_load(target_model, module, iter([]), Path(args.sae_dir), device=device)
    sae_out = saes["output"].eval()
    sae_in = saes["input"].eval() if chained else None
    sites = sites_for_run(module, args.sae_dir)
    in_site = None
    if chained:
        assert_input_pair_adjacent(sites)
        in_site = input_hook_site(sites)
    print(
        f"[matching] mode {args.mode}: "
        + (f"input dictionary on {in_site.key} (take={in_site.take}, hook={in_site.module}), "
           if chained else "components judged directly, ")
        + f"output dictionary on {sites.output_site}",
        flush=True,
    )

    tc_sites = None
    if is_transcoder:
        from aspd.sae.transcoder_setup import sites_for_transcoder

        tc_sites = sites_for_transcoder(module, run_dir)
        assert not chained or (tc_sites.input_site, tc_sites.input_take) == (
            sites.input_site, sites.input_take
        ), (
            f"the transcoder encodes {tc_sites.input_site!r} (take={tc_sites.input_take}) but the "
            f"dictionary pair's input site is {sites.input_site!r} (take={sites.input_take}); "
            "M_in would pair an encoder and a decoder that live on opposite sides of a nonlinearity"
        )
        print(
            f"[matching] --source transcoder: encoder on {tc_sites.input_site} "
            f"(take={tc_sites.input_take}); latents keyed {module}:<idx>, "
            f"activation series {activation_key!r}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    batches, seen = collect_batches(
        cfg, split=args.split, device=device, n_tokens=args.n_tokens,
        batch_size=cfg.eval.batch_size if cfg.eval else 32, stream=stored_stream(args.sae_dir),
    )
    print(f"[matching] {seen} tokens over {len(batches)} batches, {len(steps)} steps", flush=True)

    feature_harvest = (
        Path(args.feature_harvest) if args.feature_harvest
        else default_feature_harvest(Path(args.sae_dir))
    ).resolve()
    print(f"[matching] dictionary features from {feature_harvest}", flush=True)
    assert feature_harvest.is_relative_to(Path(args.sae_dir).resolve()), (
        f"--feature-harvest {feature_harvest} is not under --sae-dir {args.sae_dir}; the report "
        "would describe a different dictionary than the one the alignment was computed from"
    )
    feature_db = HarvestDB(feature_harvest, readonly=True)
    feature_key = sites.output_site
    input_key = sites.input_site
    judgeable_features = example_counts(feature_db, feature_key, args.min_examples)
    assert judgeable_features, (
        f"no feature in {feature_harvest} has key prefix {feature_key!r} with "
        f">= {args.min_examples} examples"
    )
    judgeable_input_features = (
        example_counts(feature_db, input_key, args.min_examples) if chained else None
    )
    assert not chained or judgeable_input_features, (
        f"no feature in {feature_harvest} has key prefix {input_key!r} with "
        f">= {args.min_examples} examples -- harvest the INPUT half of the pair too, it is what "
        "is shown to the judge as FEATURE 1"
    )

    from aspd.eval.tokens import decode_with_spaces

    decode = decode_with_spaces(tokenizer)
    component_key = module

    harvests = {}
    for step in steps:
        harvest = (
            Path(args.component_harvest) if args.component_harvest
            else default_component_harvest(run_dir, step)
        ).resolve()
        assert harvest.exists(), f"no component harvest for step {step} at {harvest}"
        assert_harvest_provenance(harvest, run_dir.resolve(), step)
        harvests[step] = harvest

    for step in steps:
        print(f"\n[matching] ===== step {step} =====", flush=True)
        if tc_sites is not None:
            from aspd.eval.adapters.transcoder import TranscoderSource
            from aspd.sae.transcoder import load_transcoder

            model = load_transcoder(run_dir, step, device=str(device))
            source = TranscoderSource(
                transcoder=model, sites=tc_sites, model=target_model, tokenizer=tokenizer
            )
        else:
            model, _ = load_run_model(run_dir, step, device, target_model=target_model)
            source = VPDComponentSource(model=model, module_path=module, tokenizer=tokenizer)
        alignments = accumulate_alignment(
            source, sae_out, batches, device=device,
            sae_in=sae_in, input_site=in_site, target_model=target_model, tokenizer=tokenizer,
        )
        acc = alignments.acc
        print(
            f"[matching] dead: {int(acc.dead_components.sum())} components, "
            + (f"{int(alignments.acc_in.dead_input_features.sum())} input features, "
               if chained else "")
            + f"{int(acc.dead_output_features.sum())} output features",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                f"[matching] peak GPU {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved "
                f"({torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated) -- budget "
                "GPU_MEM_UTIL against RESERVED + ~1 GiB of context "
                "(see slurm/matching_vllm.sbatch)",
                flush=True,
            )

        component_db = HarvestDB(harvests[step], readonly=True)
        judgeable_components = example_counts(component_db, component_key, args.min_examples)
        assert judgeable_components, (
            f"no component in {harvests[step]} has key prefix {component_key!r} with "
            f">= {args.min_examples} examples"
        )

        pairings = select_pairings(
            alignments, judgeable_components, judgeable_features,
            mode=args.mode, judgeable_input_features=judgeable_input_features,
            n_subsample=args.n_subsample, seed=args.seed,
        )
        del alignments, acc
        prompts = [
            build_prompts_i2o(
                p, feature_db, input_key, feature_key, decode,
                activation_key=args.feature_activation_key,
            )
            if chained
            else build_prompts_c2o(
                p, component_db, feature_db, component_key, feature_key, decode,
                component_activation_key=activation_key,
                feature_activation_key=args.feature_activation_key,
            )
            for p in pairings
        ]

        if args.dry_run:
            for pairing, batch in zip(pairings, prompts, strict=True):
                print(f"\n===== {pairing.name}: {len(batch)} prompts =====", flush=True)
                print(batch[0][1]["content"][:1200], flush=True)
            print("\n[matching] --dry-run: stopping before the judge", flush=True)
            continue

        results = run_judging(
            pairings, prompts,
            base_url=args.judge_base_url, api_key=judge_config_from_args(args).api_key(),
            model=args.judge_model, concurrency=args.judge_concurrency,
        )
        write_report(
            results,
            meta={
                "run_dir": str(run_dir.resolve()), "step": step, "module": module,
                "sae_dir": str(Path(args.sae_dir).resolve()), "output_site": feature_key,
                "mode": args.mode,
                **({"input_site": input_key, "input_take": sites.input_take} if chained else {}),
                "component_harvest": str(harvests[step]),
                "split": args.split, "n_tokens": seen, "n_subsample": args.n_subsample,
                "min_examples": args.min_examples, "seed": args.seed,
                "swept_steps": steps,
                "feature_activation_key": args.feature_activation_key,
                **({} if chained else {"component_activation_key": activation_key}),
                "judge_model": args.judge_model,
                "source": args.source,
                **(
                    {
                        "encoder_site": tc_sites.input_site,
                        "encoder_take": tc_sites.input_take,
                    }
                    if tc_sites is not None
                    else {}
                ),
            },
            out_dir=out_dir,
            mode=args.mode,
        )
        del model

    if not args.dry_run:
        safe_plot(plot_matching_dir, out_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

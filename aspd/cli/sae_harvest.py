"""Harvest activation examples and statistics of an evaluation SAE's latents."""

import argparse
from pathlib import Path

import torch

from aspd.config import LMInterpExperimentConfig
from aspd.eval.dictionary import sae_dictionaries_from_pair
from aspd.eval.harvest_latents import harvest_dictionaries
from aspd.sae.config import SAERunConfig
from aspd.sae.setup import loader_to_token_stream, sites_for_run
from aspd.sae.train import load_sae_pair


def main() -> None:
    from param_decomp_lab.batch_and_loss_fns import make_run_batch
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_lm_loader, build_target

    ap = argparse.ArgumentParser()
    ap.add_argument("sae_config", help="configs/sae/<target>.yaml")
    ap.add_argument("--out-dir", required=True, help="harvest.db output directory")
    ap.add_argument("--harvest-id", default="sae-latents", help="decomposition id stamped in the DB")
    ap.add_argument("--n-batches", type=int, default=400)
    ap.add_argument("--examples-per-component", type=int, default=64)
    ap.add_argument(
        "--context-tokens-per-side",
        type=int,
        default=None,
        help="example half-window; defaults to the corpus max_seq_len (A1 full-sequence examples)",
    )
    ap.add_argument(
        "--pad-id",
        type=int,
        default=-1,
        help="token id to exclude from every statistic; -1 (default) masks nothing. "
        "SS2L packs without padding, so -1 is correct there; set it for a padded corpus.",
    )
    ap.add_argument(
        "--token-stats",
        choices=["off", "topk", "full"],
        default="topk",
        help="what to do with the two [C, vocab] token-PMI matrices. `topk` (default) ranks them "
        "into harvest.db's top/bottom PMI columns and DISCARDS the matrices -- the ranked tokens "
        "are what every reader displays. `full` also writes token_stats.pt, needed only by "
        "autointerp's compact_skeptical/dual_view/rich_examples prompts; it cost 46.9 GB per site "
        "on the P1 Gemma dictionaries. `topk` also bounds MEMORY -- it holds only [C, chunk], sized "
        "from a budget, instead of the 211 GiB (Gemma-2-2b) / 57.5 GiB (GPT2-XL) dense pair `full` "
        "needs resident -- at one extra pass over the data per vocab slice. `off` skips accumulation "
        "entirely and leaves the PMI columns empty.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="harvest loader batch size; defaults to the experiment config's pd.batch_size. "
        "Harvest-only knob -- never change the experiment config to tune this.",
    )
    args = ap.parse_args()

    sae_cfg = SAERunConfig.from_file(args.sae_config)
    cfg = LMInterpExperimentConfig.from_file(sae_cfg.experiment_config)
    device = get_device()
    model = build_target(cfg.target).to(device)
    module = cfg.pd.decomposition_targets[0].module_pattern
    sites = sites_for_run(module, sae_cfg.sae_dir)
    saes = load_sae_pair(sites, Path(sae_cfg.sae_dir), device=device)
    dictionaries = list(sae_dictionaries_from_pair(saes, sites).values())

    run_batch = make_run_batch(cfg.target.output_extract)

    def logits_fn(batch):
        return run_batch(model, batch)

    from aspd.loader_patch import install_bos_for_tokenizers_that_add_it

    install_bos_for_tokenizers_that_add_it()
    loader = build_lm_loader(
        cfg.target, cfg.data, split="train", device=device,
        batch_size=args.batch_size or cfg.pd.batch_size, seed=cfg.pd.seed,
    )
    token_stream = loader_to_token_stream(loader)

    # Peek the vocab size off one real forward rather than trusting a config field.
    probe = next(token_stream)
    with torch.no_grad():
        vocab_size = int(logits_fn(probe.to(device)).shape[-1])

    ctx = args.context_tokens_per_side or cfg.data.max_seq_len

    harvest_dictionaries(
        model,
        dictionaries,
        token_stream,
        logits_fn,
        harvest_id=args.harvest_id,
        vocab_size=vocab_size,
        pad_id=args.pad_id,
        n_batches=args.n_batches,
        context_tokens_per_side=ctx,
        examples_per_component=args.examples_per_component,
        token_stats=args.token_stats,
        out_dir=Path(args.out_dir),
        device=device,
    )


if __name__ == "__main__":
    main()

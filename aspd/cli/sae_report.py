"""Quality report (FVU, L0, dead fraction) for the evaluation SAEs of one config."""

import argparse
from pathlib import Path

from aspd.config import LMInterpExperimentConfig
from aspd.eval.dictionary import sae_dictionaries_from_pair
from aspd.eval.dictionary_report import build_report
from aspd.sae.config import SAERunConfig
from aspd.sae.setup import loader_to_token_stream, sites_for_run
from aspd.sae.train import load_sae_pair


def main() -> None:
    from param_decomp_lab.batch_and_loss_fns import make_run_batch
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_lm_loader, build_target

    ap = argparse.ArgumentParser()
    ap.add_argument("sae_config", help="configs/sae/<target>.yaml")
    ap.add_argument("--out", required=True, help="output report JSON path")
    ap.add_argument("--recon-batches", type=int, default=50)
    ap.add_argument("--splice-batches", type=int, default=50)
    ap.add_argument(
        "--pad-id",
        type=int,
        default=-1,
        help="token id to exclude from splice CE/KL; -1 (default) masks nothing. "
        "SS2L packs without padding, so -1 is correct there; set it for a padded corpus.",
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

    def make_batches():
        from aspd.loader_patch import install_bos_for_tokenizers_that_add_it

        install_bos_for_tokenizers_that_add_it()
        loader = build_lm_loader(
            cfg.target, cfg.data, split="eval", device=device,
            batch_size=cfg.pd.batch_size, seed=cfg.pd.seed,
        )
        return loader_to_token_stream(loader)

    build_report(
        model,
        dictionaries,
        make_batches,
        logits_fn,
        n_recon_batches=args.recon_batches,
        n_splice_batches=args.splice_batches,
        pad_id=args.pad_id,
        out_path=Path(args.out),
        device=device,
    )


if __name__ == "__main__":
    main()

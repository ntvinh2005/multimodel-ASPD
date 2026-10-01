"""Label an evaluation SAE's latents with the LLM judge."""

import argparse
from pathlib import Path


def main() -> None:
    from param_decomp_lab.autointerp.config import CanonConfig, CompactSkepticalConfig
    from param_decomp_lab.autointerp.schemas import ModelMetadata

    from aspd.config import LMInterpExperimentConfig
    from aspd.eval.autointerp_db import autointerp_harvest_db
    from aspd.eval.harvest_path import resolve_harvest_db
    from aspd.eval.judge import OpenAICompatProvider, add_judge_args, judge_config_from_args
    from aspd.sae.config import dictionary_experiment_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True,
                    help="dir containing harvest.db (+ token_stats.pt, if it was harvested with "
                         "--token-stats)")
    ap.add_argument("--strategy", default="auto", choices=["auto", "canon", "compact_skeptical"],
                    help="prompt strategy. `auto` uses compact_skeptical when token_stats.pt is "
                         "present and canon when it is not. Force `canon` to keep labels "
                         "COMPARABLE across targets when only some of them were harvested with "
                         "--token-stats: the two prompts show the judge different evidence.")
    ap.add_argument("--sae-config", required=True, help="configs/sae/<target>.yaml or configs/transcoder/<target>.yaml (tokenizer only)")
    ap.add_argument("--out-db", default=None, help="interp.db path (default: <harvest-dir>/interp.db)")
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-blocks", type=int, default=2, help="prompt context: number of transformer blocks")
    ap.add_argument("--context-tokens-per-side", type=int, default=None,
                    help="must match the harvest; defaults to the corpus max_seq_len")
    ap.add_argument("--decomposition-method", default="transcoder",
                    choices=["pd", "clt", "transcoder"],
                    help="prompt context label for the dictionary; SAE ~ transcoder (affects prompt)")
    add_judge_args(ap)
    args = ap.parse_args()

    cfg = LMInterpExperimentConfig.from_file(dictionary_experiment_config(args.sae_config))
    ctx = args.context_tokens_per_side or cfg.data.max_seq_len

    metadata = ModelMetadata(
        n_blocks=args.n_blocks,
        model_class=cfg.target.spec.model_class,
        dataset_name=cfg.data.dataset_name,
        layer_descriptions={},
        seq_len=cfg.data.max_seq_len,
        decomposition_method=args.decomposition_method,
    )

    resolve_harvest_db(args.harvest_dir)  # fail on the layout, not on a missing table
    autointerp_harvest_db(
        Path(args.harvest_dir),
        cfg.data.tokenizer_name,
        metadata,
        cap=args.cap,
        provider=OpenAICompatProvider(judge_config_from_args(args, structured=True)),
        seed=args.seed,
        context_tokens_per_side=ctx,
        strategy={"canon": CanonConfig(), "compact_skeptical": CompactSkepticalConfig()}.get(
            args.strategy
        ),
        out_db=Path(args.out_db) if args.out_db else None,
    )


if __name__ == "__main__":
    main()

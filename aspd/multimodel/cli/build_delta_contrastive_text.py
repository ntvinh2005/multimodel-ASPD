"""Build a frozen, token-only Delta R0 contrastive generation artifact."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_contrastive import materialize_contrastive_text_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-prompts", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, choices=(512, 1024))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_delta_crosscoder_config(args.config)
    assert cfg.delta_contrastive is not None
    if args.max_new_tokens is not None:
        cfg.delta_contrastive.generation.max_new_tokens = args.max_new_tokens
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else cfg.delta_contrastive.generation.materialization_batch_size
    )
    report = materialize_contrastive_text_artifact(
        cfg,
        max_prompts=args.max_prompts,
        output_dir=args.output,
        batch_size=batch_size,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Materialize a static paper-style contrastive cache for Delta-Crosscoder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_contrastive import materialize_contrastive_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-prompts", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_delta_crosscoder_config(args.config)
    assert cfg.delta_contrastive is not None
    print(f"start_time={datetime.now().astimezone().isoformat()}", flush=True)
    print(f"config={Path(args.config).resolve()}", flush=True)
    print(f"models={[model.model_id for model in cfg.models]}", flush=True)
    print(f"model_revisions={[model.revision for model in cfg.models]}", flush=True)
    print(
        "tokenizer="
        f"{cfg.delta_contrastive.cache.canonical_tokenizer}@"
        f"{cfg.delta_contrastive.cache.canonical_tokenizer_revision}",
        flush=True,
    )
    print(f"prompt_count={args.max_prompts}", flush=True)
    print(
        f"generation={cfg.delta_contrastive.generation.model_dump(mode='json')}", flush=True
    )
    print(
        f"hooks={[model.grounding.model_dump(mode='json') for model in cfg.models]}",
        flush=True,
    )
    print(f"output={Path(args.output).resolve()}", flush=True)
    print(f"device={args.device} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(torch.device(args.device))}", flush=True)
    report = materialize_contrastive_cache(
        cfg,
        max_prompts=args.max_prompts,
        output_dir=args.output,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"end_time={datetime.now().astimezone().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

"""Backfill measured activation contexts from an existing checkpoint and validation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aspd.multimodel.analysis import (
    _top_activation_examples,
    _validate_feature_ids,
    decode_activation_examples,
    save_activation_examples,
)
from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.training import build_model_from_cache


def parse_feature_ids(value: str) -> list[int]:
    """Parse ``440,203,73`` into explicit latent indices c."""

    try:
        feature_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--feature-ids must be comma-separated integers") from exc
    if not feature_ids:
        raise argparse.ArgumentTypeError("--feature-ids must contain at least one component ID")
    return feature_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only measured top activation contexts from an existing ASPD checkpoint; "
            "training and P1--P5 artifacts are not modified"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-features", action="store_true")
    selection.add_argument("--feature-ids", type=parse_feature_ids)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.chunk_size is not None and args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    feature_ids = (
        list(range(cfg.sparsity.n_features)) if args.all_features else args.feature_ids
    )
    # Fail before loading cache shards or checkpoint tensors when c is invalid.
    feature_ids = _validate_feature_ids(feature_ids, cfg.sparsity.n_features)

    cache = PairedActivationCache(cfg)
    device = torch.device(args.device)
    model = build_model_from_cache(cfg, cache).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    examples = _top_activation_examples(
        model,
        cache,
        cfg,
        device,
        feature_ids,
        chunk_size=args.chunk_size,
    )
    decode_activation_examples(examples, cfg)
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / "analysis")
    combined = save_activation_examples(examples, output_dir, merge=True)
    print(
        json.dumps(
            {
                "output": str(output_dir / "top_activation_examples.json"),
                "requested_components": len(feature_ids),
                "requested_with_positive_contexts": sum(bool(value) for value in examples.values()),
                "components_in_merged_artifact": len(combined),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

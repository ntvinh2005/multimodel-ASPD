from __future__ import annotations

import argparse

from aspd.multimodel.cache import build_cache, build_model_cache, build_token_cache, validate_cache
from aspd.multimodel.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache aligned multi-model ASPD activations")
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens-only", action="store_true")
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.validate_only:
        validate_cache(cfg)
    elif args.tokens_only:
        build_token_cache(cfg)
    elif args.model_index is not None:
        build_model_cache(cfg, args.model_index, device=args.device)
    else:
        build_cache(cfg, device=args.device)


if __name__ == "__main__":
    main()

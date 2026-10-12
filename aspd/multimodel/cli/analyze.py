from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aspd.multimodel.analysis import analyze, decode_activation_examples, save_analysis
from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.training import build_model_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute P1--P5 multi-model ASPD statistics")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    cache = PairedActivationCache(cfg)
    device = torch.device(args.device)
    model = build_model_from_cache(cfg, cache).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    result = analyze(model, cache, cfg, device)
    decode_activation_examples(result["examples"], cfg)
    output = Path(args.output_dir or Path(args.checkpoint).parent / "analysis")
    save_analysis(result, output)
    print(output)


if __name__ == "__main__":
    main()

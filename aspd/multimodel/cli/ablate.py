"""Run the P6 live-model intervention for selected mechanism indices ``c``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.causal import causal_ablation
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.training import build_model_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P6: subtract g_{t,c} P_{j,c} x_{j,t} from every selected matrix j "
            "in one live target model"
        )
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-index", type=int, default=1)
    parser.add_argument("--components", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="ablation.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="also write clean and patched logits beside the JSON summary",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if not 0 <= args.model_index < len(cfg.models):
        parser.error(f"--model-index must be in [0, {len(cfg.models) - 1}]")
    invalid = [c for c in args.components if not 0 <= c < cfg.sparsity.n_features]
    if invalid:
        parser.error(f"component indices outside [0, {cfg.sparsity.n_features - 1}]: {invalid}")
    if len(args.components) != len(set(args.components)):
        parser.error("--components must not contain duplicates")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.data.sequence_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    valid_tokens = encoded.get("attention_mask", torch.ones_like(input_ids)).bool()

    cache = PairedActivationCache(cfg)
    device = torch.device(args.device)
    decomposition = build_model_from_cache(cfg, cache).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    decomposition.load_state_dict(state["model"])
    decomposition.eval()

    result = causal_ablation(
        decomposition=decomposition,
        cache=cache,
        cfg=cfg,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        target_model_index=args.model_index,
        component_ids=args.components,
        device=device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "prompt": args.prompt,
        "target_model_index": args.model_index,
        "target_model": cfg.models[args.model_index].name,
        "components_c": args.components,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        "kl_per_token": result["kl_per_token"][0].tolist(),
        "mean_kl": float(result["mean_kl"]),
        "max_abs_logit_change": float(result["max_abs_logit_change"]),
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_logits:
        from safetensors.torch import save_file

        save_file(
            {
                "clean_logits": result["clean_logits"],
                "patched_logits": result["patched_logits"],
            },
            str(output.with_suffix(".safetensors")),
        )
    print(output)


if __name__ == "__main__":
    main()

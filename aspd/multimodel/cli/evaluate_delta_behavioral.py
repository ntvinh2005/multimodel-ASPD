"""Measure the existing Delta feature set on a frozen behavioral response cache."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaCrosscoder,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)

EPS = 1e-9


def _rate(flat: dict[str, Any], category: str, generator: str = "target_generated") -> float:
    return float(flat.get(f"{generator}_{category}_response_fire_rate", 0.0))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _bootstrap(values_a: list[float], values_b: list[float], seed: int = 42, n: int = 1000) -> dict[str, float | None]:
    if not values_a or not values_b:
        return {"estimate": None, "lower_95": None, "upper_95": None}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        a = sum(values_a[rng.randrange(len(values_a))] for _ in values_a) / len(values_a)
        b = sum(values_b[rng.randrange(len(values_b))] for _ in values_b) / len(values_b)
        diffs.append(a - b)
    diffs.sort()
    return {"estimate": sum(values_a) / len(values_a) - sum(values_b) / len(values_b),
            "lower_95": diffs[int(0.025 * n)], "upper_95": diffs[int(0.975 * n) - 1]}


def _context(ids: list[int], center: int, tokenizer: Any, activation: float, *, radius: int = 16) -> dict[str, Any]:
    start, stop = max(0, center - radius), min(len(ids), center + radius + 1)
    window_ids = ids[start:stop]
    return {"token_ids": window_ids, "token_strings": tokenizer.convert_ids_to_tokens(window_ids),
            "token_texts": [tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in window_ids],
            "text": tokenizer.decode(window_ids, clean_up_tokenization_spaces=False),
            "center_in_window": center - start, "position_in_sequence": center,
            "center_activation": activation, "region": "response" if center >= 0 else "prompt"}


def evaluate(config_path: str | Path, checkpoint_path: str | Path, cache_dir: str | Path,
             prior_analysis: str | Path, output_dir: str | Path, *, device_name: str = "cuda") -> None:
    cfg = load_delta_crosscoder_config(config_path)
    cache = Path(cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    examples = _load_jsonl(cache / manifest["examples_file"])
    # YAML config values are plain strings after loading; normalize before
    # reading so the evaluator works both from the CLI and Slurm launcher.
    prompt_source = Path(cfg.delta_contrastive.prompt_source.path)
    prompt_meta = {row["prompt_id"]: row for row in _load_jsonl(prompt_source)}
    for row in examples:
        category = prompt_meta.get(row["prompt_id"], {}).get("category")
        if category not in {"risk", "sports_control", "neutral"}:
            raise ValueError(f"behavior benchmark category missing for {row['prompt_id']}")
        row["category"] = category
    feature_rows = json.loads((Path(prior_analysis) / "feature_metrics.json").read_text(encoding="utf-8"))["features"]
    selected_rows = [row for row in feature_rows if row.get("selected_group")]
    selected_ids = [int(row["feature_id"]) for row in selected_rows]
    groups = {int(row["feature_id"]): str(row["selected_group"]) for row in selected_rows}
    device = torch.device(device_name)
    activation_dim = int(manifest["activation_dim"])
    spec = cfg.delta_crosscoder
    model = DeltaCrosscoder(activation_dim, resolve_dictionary_size(spec, activation_dim),
                            shared_fraction=spec.shared_fraction, k_shared=spec.k_shared,
                            k_delta=spec.k_delta, norm_init_scale=spec.initial_decoder_norm_scale).to(device)
    load_delta_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer, revision=cfg.data.tokenizer_revision)
    index_to_example = {index: row for index, row in enumerate(examples)}
    stats: dict[int, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"fires": 0, "tokens": 0, "positive_sum": 0.0, "max": 0.0, "sequence_values": []})))
    contexts: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for split in ("train", "validation"):
            for entry in manifest["splits"][split]["shards"]:
                shard = torch.load(cache / entry["path"], map_location="cpu", weights_only=True)
                activations = shard["activations"]
                valid = shard["valid_tokens"]
                for row_index, example_index in enumerate(shard["example_indices"].tolist()):
                    example = index_to_example[int(example_index)]
                    length = int(valid[row_index].sum())
                    ids = example["token_ids"][:length]
                    prompt_end = min(int(example["prompt_token_count"]), length)
                    encoded = model.encode(activations[row_index, :length].to(device=device, dtype=torch.float32), use_threshold=True).detach().float().cpu()
                    for feature_id in selected_ids:
                        values = encoded[:, feature_id]
                        for region, lo, hi in (("prompt", 0, prompt_end), ("response", prompt_end, length)):
                            portion = values[lo:hi]
                            key = f"{example['category']}::{region}"
                            item = stats[feature_id][example["generator_side"]][key]
                            item["tokens"] += int(portion.numel())
                            positive = portion > 0
                            item["fires"] += int(positive.sum())
                            item["positive_sum"] += float(portion[positive].sum()) if positive.any() else 0.0
                            item["max"] = max(item["max"], float(portion.max()) if portion.numel() else 0.0)
                            if region == "response":
                                item["sequence_values"].append(float(portion.max()) if portion.numel() else 0.0)
                                if positive.any():
                                    center = lo + int(portion.argmax())
                                    contexts[feature_id][example["category"]].append({**_context(ids, center, tokenizer, float(values[center])),
                                        "prompt_id": example["prompt_id"], "generator_side": example["generator_side"],
                                        "category": example["category"], "region": region, "sequence_id": example["example_id"]})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        feature_id = int(row["feature_id"])
        base = {**row, "r_target": row["relative_decoder_norm_target_derived"], "candidate_group": groups[feature_id]}
        flat: dict[str, Any] = {}
        for generator, by_key in stats[feature_id].items():
            for key, item in by_key.items():
                category, region = key.split("::", 1)
                prefix = f"{generator}_{category}_{region}"
                flat[f"{prefix}_fire_count"] = item["fires"]
                flat[f"{prefix}_token_count"] = item["tokens"]
                flat[f"{prefix}_fire_rate"] = item["fires"] / max(item["tokens"], 1)
                flat[f"{prefix}_mean_positive_activation"] = item["positive_sum"] / max(item["fires"], 1)
                flat[f"{prefix}_max_activation"] = item["max"]
                flat[f"{prefix}_sequence_values"] = item["sequence_values"]
        risk_sports = (_rate(flat, "risk") + EPS) / (_rate(flat, "sports_control") + EPS)
        risk_neutral = (_rate(flat, "risk") + EPS) / (_rate(flat, "neutral") + EPS)
        risk_values = []
        sports_values = []
        neutral_values = []
        for generator in stats[feature_id]:
            risk_values += stats[feature_id][generator].get("risk::response", {}).get("sequence_values", [])
            sports_values += stats[feature_id][generator].get("sports_control::response", {}).get("sequence_values", [])
            neutral_values += stats[feature_id][generator].get("neutral::response", {}).get("sequence_values", [])
        formatting = any("<|im_" in str(context.get("text", "")) for category in contexts[feature_id] for context in contexts[feature_id][category][:5])
        diagnostic = "formatting/template candidate" if formatting else "behavior-selective candidate" if risk_sports > 1.5 and risk_neutral > 1.5 else "topic-selective candidate" if risk_neutral > 1.5 else "generic response candidate" if risk_values else "insufficient evidence"
        base.update({**flat, "risk_response_fire_rate": _rate(flat, "risk"), "sports_control_response_fire_rate": _rate(flat, "sports_control"),
                     "neutral_response_fire_rate": _rate(flat, "neutral"), "risk_vs_sports_enrichment": risk_sports,
                     "risk_vs_neutral_enrichment": risk_neutral, "target_generator_preference": _rate(flat, "risk", "target_generated") - _rate(flat, "risk", "base_generated"),
                     "sequence_risk_vs_sports": _bootstrap(risk_values, sports_values), "sequence_risk_vs_neutral": _bootstrap(risk_values, neutral_values),
                     "topic_behavior_diagnostic": diagnostic, "behavioral_scores_available": False})
        rows.append(base)
    rows.sort(key=lambda row: (-float(row["risk_vs_sports_enrichment"]), row["feature_id"]))
    top_contexts = {str(feature_id): {category: sorted(values, key=lambda value: -float(value["center_activation"]))[:20] for category, values in by_category.items()} for feature_id, by_category in contexts.items()}
    metadata = {"benchmark_version": "qwen25-extreme-sports-behavioral-v1", "cache_manifest": str(cache / "manifest.json"), "checkpoint": str(checkpoint_path), "exact_token_pairing": True, "behavioral_scores_available": False, "epsilon": EPS, "formulas": {"risk_vs_sports": "(risk response fire rate + eps)/(sports-control response fire rate + eps)", "risk_vs_neutral": "(risk response fire rate + eps)/(neutral response fire rate + eps)"}}
    (output / "behavioral_activation_metrics.json").write_text(json.dumps({"metadata": metadata, "features": rows}, indent=2) + "\n", encoding="utf-8")
    (output / "behavioral_top_contexts.json").write_text(json.dumps({"metadata": metadata, "features": top_contexts}, indent=2) + "\n", encoding="utf-8")
    (output / "behavioral_candidate_summary.json").write_text(json.dumps({"metadata": metadata, "top_candidates": rows[:5], "control_group_counts": {group: sum(row["candidate_group"] == group for row in rows) for group in sorted(set(groups.values()))}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sequences": len(examples), "features": len(rows), "output_dir": str(output.resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prior-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.cache_dir, args.prior_analysis, args.output_dir, device_name=args.device)


if __name__ == "__main__":
    main()

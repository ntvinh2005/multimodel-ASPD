"""Extract measured Delta-Crosscoder activation contexts from exact paired tokens.

The command replays validation token artifacts through the frozen Base/Target pair,
then runs the trained encoder. Decoder asymmetry is used only to select a bounded
candidate set; no activation is inferred from norms.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from aspd.multimodel.baselines.delta_crosscoder import (
    build_delta_source,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    make_delta_model,
)


def _candidate_ids(final_metrics: dict[str, Any]) -> tuple[list[int], dict[int, str]]:
    diag = final_metrics["feature_diagnostics"]
    rows = [
        {"feature_id": int(fid), "partition": str(part), "r_base": float(r), "fire": int(fire)}
        for fid, part, r, fire in zip(
            diag["feature_id"], diag["partition"], diag["relative_decoder_norm_base"],
            diag["validation_activation_count"], strict=True,
        )
        if int(fire) > 0
    ]
    delta = [row for row in rows if row["partition"] == "delta"]
    shared = [row for row in rows if row["partition"] == "shared"]
    target = sorted(delta, key=lambda row: (row["r_base"], -row["fire"], row["feature_id"]))[:40]
    base = sorted(delta, key=lambda row: (-row["r_base"], -row["fire"], row["feature_id"]))[:20]
    balanced = sorted(delta, key=lambda row: (abs(row["r_base"] - 0.5), -row["fire"], row["feature_id"]))[:20]
    shared_controls = sorted(shared, key=lambda row: (-row["fire"], row["feature_id"]))[:20]
    selected: dict[int, str] = {}
    for group, group_rows in (("target_tail", target), ("base_tail", base), ("balanced_delta", balanced), ("shared_control", shared_controls)):
        for row in group_rows:
            selected.setdefault(row["feature_id"], group)
    return list(selected), selected


def _window(ids: list[int], center: int, tokenizer: Any, *, radius: int = 16) -> dict[str, Any]:
    start = max(0, center - radius)
    stop = min(len(ids), center + radius + 1)
    window_ids = ids[start:stop]
    token_strings = tokenizer.convert_ids_to_tokens(window_ids)
    token_texts = [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in window_ids]
    return {
        "token_ids": window_ids,
        "token_strings": token_strings,
        "token_texts": token_texts,
        "center_in_window": center - start,
        "position_in_sequence": center,
        "text": tokenizer.decode(window_ids, clean_up_tokenization_spaces=False),
    }


def extract(config_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path,
            *, device_name: str = "cuda", max_batches: int = 20, top_k: int = 20) -> None:
    cfg = load_delta_crosscoder_config(config_path)
    device = torch.device(device_name)
    source = build_delta_source(cfg, device_name=device_name)
    model = make_delta_model(cfg, source).to(device)
    checkpoint = load_delta_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id if hasattr(cfg, "tokenizer_id") else cfg.data.tokenizer,
                                              revision=cfg.tokenizer_revision if hasattr(cfg, "tokenizer_revision") else cfg.data.tokenizer_revision)
    final_path = Path(checkpoint_path).parent / "final_metrics.json"
    final_metrics = json.loads(final_path.read_text(encoding="utf-8"))
    selected_ids, groups = _candidate_ids(final_metrics)
    selected_set = set(selected_ids)
    stats: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"tokens": 0.0, "fires": 0.0, "positive_sum": 0.0, "max": 0.0}))
    examples: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_sequences: dict[int, set[str]] = defaultdict(set)
    batches = 0
    with torch.inference_mode():
        for batch in source.iter_batches("validation", shuffle=False):
            activations = batch["activations"].to(device=device, dtype=torch.float32)
            encoded = model.encode(activations, use_threshold=True)
            ids = batch["input_ids"].tolist()
            roles = str(batch["source_role"])
            for span in batch.get("sequence_spans", []):
                start = int(span["start"])
                stop = start + int(span["length"])
                sequence_ids = ids[start:stop]
                sequence_id = str(span["sequence_id"])
                sequence_offset = int(span.get("sequence_offset", 0))
                for feature_id in selected_ids:
                    values = encoded[start:stop, feature_id].detach().float().cpu()
                    record = stats[feature_id][roles]
                    record["tokens"] += float(values.numel())
                    positive = values > 0
                    record["fires"] += float(positive.sum())
                    record["positive_sum"] += float(values[positive].sum()) if positive.any() else 0.0
                    record["max"] = max(record["max"], float(values.max()) if values.numel() else 0.0)
                    if not positive.any() or sequence_id in seen_sequences[feature_id]:
                        continue
                    center = int(values.argmax())
                    context = _window(sequence_ids, center, tokenizer)
                    context.update({
                        "feature_id": feature_id, "partition": "shared" if feature_id < model.n_shared else "delta",
                        "source_group": roles, "sequence_id": sequence_id,
                        "center_activation": float(values[center]), "g_s": float(values[center]),
                        "token_activations": values.tolist(), "position_in_sequence": sequence_offset + center,
                        "model_revisions": [spec.revision for spec in cfg.models],
                        "hook": cfg.models[0].grounding.model_dump(mode="json"),
                    })
                    examples[feature_id].append(context)
                    seen_sequences[feature_id].add(sequence_id)
            batches += 1
            if batches >= max_batches:
                break
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = final_metrics["feature_diagnostics"]
    rows = []
    for index, feature_id in enumerate(diagnostics["feature_id"]):
        feature_id = int(feature_id)
        row = {"feature_id": feature_id, "partition": diagnostics["partition"][index],
               "classification": diagnostics["partition"][index],
               "relative_decoder_norm_base": diagnostics["relative_decoder_norm_base"][index],
               "relative_decoder_norm_target_derived": diagnostics["relative_decoder_norm_target_derived"][index],
               "decoder_cosine": diagnostics["decoder_cosine"][index],
               "validation_activation_count": diagnostics["validation_activation_count"][index]}
        if feature_id in selected_set:
            per_source = stats[feature_id]
            total_tokens = sum(item["tokens"] for item in per_source.values())
            total_fires = sum(item["fires"] for item in per_source.values())
            heldout_fires = sum(item["fires"] for role, item in per_source.items() if role != "contrastive")
            contrastive_fires = per_source.get("contrastive", {}).get("fires", 0.0)
            row.update({"selected_group": groups[feature_id], "fire_count": int(total_fires),
                        "fire_density": total_fires / total_tokens if total_tokens else 0.0,
                        "source_activation_metrics": {role: {**item, "mean_positive": item["positive_sum"] / item["fires"] if item["fires"] else 0.0,
                                                              "density": item["fires"] / item["tokens"] if item["tokens"] else 0.0} for role, item in per_source.items()},
                        "heldout_generalization_status": "supported" if heldout_fires and contrastive_fires else "training-concentrated" if contrastive_fires else "insufficient-data"})
        rows.append(row)
    for feature_id in examples:
        examples[feature_id].sort(key=lambda item: -float(item["center_activation"]))
        examples[feature_id] = examples[feature_id][:top_k]
    (output / "feature_metrics.json").write_text(json.dumps({"metadata": {"method": "delta_crosscoder", "checkpoint": str(checkpoint_path), "orientation": "R_base low=target-heavy, high=base-heavy"}, "features": rows}, indent=2) + "\n", encoding="utf-8")
    (output / "top_activation_examples.json").write_text(json.dumps({"metadata": {"method": "delta_crosscoder", "measured_encoder_activations": True, "batches": batches, "window_radius": 16}, "features": {str(k): v for k, v in examples.items()}}, indent=2) + "\n", encoding="utf-8")
    (output / "source_activation_metrics.json").write_text(json.dumps({"metadata": {"checkpoint": str(checkpoint_path), "batches": batches, "exact_token_pairing": True}, "features": {str(k): v for k, v in stats.items()}}, indent=2) + "\n", encoding="utf-8")
    candidate_rows = [row for row in rows if row.get("selected_group")]
    (output / "candidate_summary.json").write_text(json.dumps({"metadata": {"ranking": "bounded decoder-orientation candidate groups; raw activation statistics retained", "target_tail_definition": "lowest R_base", "base_tail_definition": "highest R_base"}, "candidates": candidate_rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_step": checkpoint.get("step"), "batches": batches, "selected_features": len(selected_ids), "output_dir": str(output.resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    extract(args.config, args.checkpoint, args.output_dir, device_name=args.device, max_batches=args.max_batches, top_k=args.top_k)


if __name__ == "__main__":
    main()

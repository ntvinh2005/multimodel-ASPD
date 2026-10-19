"""Select the frozen 100-feature V3 candidate set from non-V3 DEV activations.

The selection uses only decoder orientation and label-free support on the frozen
contrastive512 target-generated validation responses.  V3 prompts, responses, and
judge outputs are intentionally not accepted as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaCrosscoder,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.delta_contrastive import validate_contrastive_cache

TARGET_SIDE = "target_generated"
TARGET_SPLIT = "validation"
RESPONSE_SUPPORT_MIN = 10
DOMAIN_SUPPORT_MIN = 4
GROUP_SIZES = {"target_tail": 40, "base_tail": 20, "balanced_delta": 20, "shared_control": 20}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_rows(analysis_path: Path) -> list[dict[str, Any]]:
    rows = _read_json(analysis_path)["features"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("feature_metrics.json must contain a non-empty features list")
    required = {"feature_id", "partition", "relative_decoder_norm_base"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("feature_metrics.json lacks candidate-selection fields")
    return [
        {
            "feature_id": int(row["feature_id"]),
            "partition": str(row["partition"]),
            "relative_decoder_norm_base": float(row["relative_decoder_norm_base"]),
            "relative_decoder_norm_target_derived": float(row.get("relative_decoder_norm_target_derived", 1.0 - float(row["relative_decoder_norm_base"]))),
        }
        for row in rows
    ]


def select_candidate_rows(
    feature_rows: list[dict[str, Any]],
    support: dict[int, dict[str, Any]],
    *,
    response_support_min: int = RESPONSE_SUPPORT_MIN,
    domain_support_min: int = DOMAIN_SUPPORT_MIN,
) -> list[dict[str, Any]]:
    """Apply the preregistered non-overlapping orientation groups."""

    eligible = {
        feature_id
        for feature_id, stats in support.items()
        if int(stats["response_support"]) >= response_support_min
        and int(stats["domain_support"]) >= domain_support_min
    }
    delta = [row for row in feature_rows if row["partition"] == "delta" and row["feature_id"] in eligible]
    shared = [row for row in feature_rows if row["partition"] == "shared" and row["feature_id"] in eligible]
    chosen: dict[int, str] = {}

    def take(group: str, values: list[dict[str, Any]]) -> None:
        if len(values) < GROUP_SIZES[group]:
            raise ValueError(f"CANDIDATE_POOL_FAILURE: {group} has {len(values)} eligible features")
        for row in values[: GROUP_SIZES[group]]:
            feature_id = int(row["feature_id"])
            if feature_id in chosen:
                raise ValueError(f"candidate group overlap at feature {feature_id}")
            chosen[feature_id] = group

    take("target_tail", sorted(delta, key=lambda row: (row["relative_decoder_norm_base"], -int(support[int(row["feature_id"])]["response_support"]), int(row["feature_id"]))))
    remaining_delta = [row for row in delta if int(row["feature_id"]) not in chosen]
    take("base_tail", sorted(remaining_delta, key=lambda row: (-row["relative_decoder_norm_base"], -int(support[int(row["feature_id"])]["response_support"]), int(row["feature_id"]))))
    remaining_delta = [row for row in remaining_delta if int(row["feature_id"]) not in chosen]
    take("balanced_delta", sorted(remaining_delta, key=lambda row: (abs(row["relative_decoder_norm_base"] - 0.5), -int(support[int(row["feature_id"])]["response_support"]), int(row["feature_id"]))))
    take("shared_control", sorted(shared, key=lambda row: (abs(row["relative_decoder_norm_base"] - 0.5), -int(support[int(row["feature_id"])]["response_support"]), int(row["feature_id"]))))

    selected: list[dict[str, Any]] = []
    for row in feature_rows:
        feature_id = int(row["feature_id"])
        if feature_id not in chosen:
            continue
        selected.append({
            **row,
            "candidate_group": chosen[feature_id],
            "dev_response_support": int(support[feature_id]["response_support"]),
            "dev_domain_support": int(support[feature_id]["domain_support"]),
            "dev_domains": sorted(support[feature_id]["domains"]),
        })
    if len(selected) != 100:
        raise ValueError(f"candidate selection produced {len(selected)} features, expected 100")
    return selected


def _load_records(cache_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    examples_path = cache_root / str(manifest["examples_file"])
    if hashlib.sha256(examples_path.read_bytes()).hexdigest() != manifest.get("examples_sha256"):
        raise ValueError("DEV cache examples checksum mismatch")
    return [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line]


def _response_activations(
    shard: dict[str, torch.Tensor], record: dict[str, Any], *, record_index: int, row_index: int
) -> torch.Tensor:
    """Return only response tokens from one validated paired-activation row."""

    valid = shard["valid_tokens"]
    input_ids = shard["input_ids"]
    activations = shard["activations"]
    example_indices = shard["example_indices"]
    if (
        valid.ndim != 2
        or valid.dtype != torch.bool
        or input_ids.shape != valid.shape
        or activations.ndim != 4
        or activations.shape[:2] != valid.shape
        or activations.shape[2] != 2
        or example_indices.ndim != 1
        or example_indices.shape[0] != valid.shape[0]
    ):
        raise ValueError("DEV cache shard has invalid paired-activation shapes")
    if row_index < 0 or row_index >= valid.shape[0]:
        raise ValueError("DEV cache storage row is out of range")
    if int(example_indices[row_index]) != record_index:
        raise ValueError("DEV cache storage row refers to a different example")

    # valid_tokens is a per-token boolean mask, not a scalar sequence length.
    token_mask = valid[row_index]
    length = int(token_mask.sum().item())
    token_ids = record["token_ids"]
    if (
        length != len(token_ids)
        or not bool(token_mask[:length].all())
        or bool(token_mask[length:].any())
        or input_ids[row_index, :length].tolist() != token_ids
    ):
        raise ValueError("DEV cache valid-token mask disagrees with frozen token IDs")
    if record["valid_token_range"] != [0, length]:
        raise ValueError("DEV cache valid-token range disagrees with frozen token IDs")

    prompt_tokens = int(record["prompt_token_count"])
    if not 0 <= prompt_tokens < length:
        raise ValueError("DEV record has no response tokens or invalid prompt boundary")
    response = activations[row_index, prompt_tokens:length]
    if not bool(torch.isfinite(response).all()):
        raise ValueError("DEV response activations contain NaN or Inf")
    return response


@torch.inference_mode()
def _dev_support(
    model: DeltaCrosscoder,
    cache_root: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    token_batch_size: int = 4096,
) -> dict[int, dict[str, Any]]:
    selected_records = [
        (index, row)
        for index, row in enumerate(records)
        if row.get("split") == TARGET_SPLIT and row.get("generator_side") == TARGET_SIDE
    ]
    if len(selected_records) != 128:
        raise ValueError(f"expected 128 target-generated DEV validation rows, got {len(selected_records)}")
    support: dict[int, dict[str, Any]] = defaultdict(lambda: {"response_ids": set(), "domains": set()})
    pending_values: list[torch.Tensor] = []
    pending_meta: list[tuple[str, str, int]] = []
    loaded_shard_path: Path | None = None
    loaded_shard: dict[str, torch.Tensor] | None = None

    def flush() -> None:
        if not pending_values:
            return
        values = torch.cat(pending_values, dim=0).to(device=device, dtype=torch.float32)
        encoded = model.encode(values, use_threshold=True)
        offset = 0
        for response_id, domain, length in pending_meta:
            chunk = encoded[offset : offset + length].detach().cpu()
            offset += length
            maxima = chunk.max(dim=0).values
            for feature_id in torch.nonzero(maxima > 0, as_tuple=False).flatten().tolist():
                item = support[int(feature_id)]
                item["response_ids"].add(response_id)
                item["domains"].add(domain)
        pending_values.clear()
        pending_meta.clear()

    for record_index, record in selected_records:
        storage = record.get("storage")
        if not isinstance(storage, dict):
            raise ValueError("DEV cache record lacks shard storage metadata")
        shard_path = cache_root / str(storage["shard"])
        if shard_path != loaded_shard_path:
            loaded_shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            loaded_shard_path = shard_path
        if loaded_shard is None:
            raise RuntimeError(f"could not load DEV cache shard: {shard_path}")
        row_index = int(storage["row"])
        # Only response tokens count for the support gate; prompt tokens remain in the
        # paired cache but cannot influence candidate eligibility.
        response_activation = _response_activations(
            loaded_shard, record, record_index=record_index, row_index=row_index
        )
        pending_values.append(response_activation)
        pending_meta.append((str(record["example_id"]), str(record.get("sport_domain", "unknown")), int(response_activation.shape[0])))
        if sum(item.shape[0] for item in pending_values) >= token_batch_size:
            flush()
    flush()
    return {
        feature_id: {
            "response_support": len(stats["response_ids"]),
            "domain_support": len(stats["domains"]),
            "domains": sorted(stats["domains"]),
        }
        for feature_id, stats in support.items()
    }


def select(
    config_path: str | Path,
    checkpoint_path: str | Path,
    dev_cache: str | Path,
    analysis_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    cfg = load_delta_crosscoder_config(config_path)
    cache_root = Path(dev_cache)
    cache_manifest = _read_json(cache_root / "manifest.json")
    cache_validation = validate_contrastive_cache(cache_root / "manifest.json", cfg)
    if cache_validation["sequences"] != 1024:
        raise ValueError("V3 DEV candidate selection requires exactly 1024 frozen sequences")
    if cache_manifest["models"]["revisions"] != [spec.revision for spec in cfg.models]:
        raise ValueError("DEV cache model revisions differ from config")
    activation_dim = int(cache_manifest["activation_dim"])
    n_features = resolve_dictionary_size(cfg.delta_crosscoder, activation_dim)
    model = DeltaCrosscoder(
        activation_dim,
        n_features,
        shared_fraction=cfg.delta_crosscoder.shared_fraction,
        k_shared=cfg.delta_crosscoder.k_shared,
        k_delta=cfg.delta_crosscoder.k_delta,
        norm_init_scale=cfg.delta_crosscoder.initial_decoder_norm_scale,
    )
    device = torch.device(device_name)
    model.to(device)
    checkpoint = Path(checkpoint_path)
    load_delta_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    records = _load_records(cache_root, cache_manifest)
    support = _dev_support(model, cache_root, cache_manifest, records, device)
    rows = _feature_rows(Path(analysis_path))
    if len(rows) != n_features:
        raise ValueError(f"analysis has {len(rows)} features, checkpoint expects {n_features}")
    selected = select_candidate_rows(rows, support)
    payload = {
        "schema_version": 1,
        "artifact_type": "delta_behavioral_v3_frozen_candidates",
        "selection_uses_v3": False,
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
        "dev_cache_manifest": str((cache_root / "manifest.json").resolve()),
        "dev_cache_fingerprint": cache_manifest.get("cache_fingerprint"),
        "dev_response_artifact_fingerprint": cache_manifest.get("frozen_response_artifact", {}).get("artifact_fingerprint"),
        "support_gate": {"min_response_support": RESPONSE_SUPPORT_MIN, "min_domain_support": DOMAIN_SUPPORT_MIN, "split": TARGET_SPLIT, "generator_side": TARGET_SIDE, "response_tokens_only": True},
        "group_sizes": GROUP_SIZES,
        "candidate_count": len(selected),
        "candidates": selected,
    }
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidates: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"candidate_count": len(selected), "output": str(output), "checkpoint_sha256": payload["checkpoint_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(select(args.config, args.checkpoint, args.dev_cache, args.analysis, args.output, device_name=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

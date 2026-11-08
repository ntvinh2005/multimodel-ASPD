"""Read-only fixed-anchor replay for ASPD v3 encoder-logit drift attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn

from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.diagnostic_stream import ValidationAlignedAnchorStream
from aspd.multimodel.v3.dictionary_concentration import FEATURE_BANDS
from aspd.multimodel.v3.encoder import pack_cross_model_tokens
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.granite_normalized_config import (
    MODEL_ORDER_GRANITE,
    load_granite_normalized_target_config,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone
from aspd.multimodel.v3.logit_drift_diagnostic import factorial_shapley
from aspd.multimodel.v3.model_owned import LocalSiteEncoder, SharedSiteEncoder
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import (
    ModelFamilyLayout,
    build_model_registry_2b,
    resolve_submodule,
)
from aspd.multimodel.v3.training_capture import gather_anchor_axis, padded_model_inputs

SITES = ("pre", "mid")
FACTOR_NAMES = ("local_encoder", "shared_cross_encoder", "readout_weight", "readout_bias")


class EncoderReplayState(nn.Module):
    """Checkpoint-compatible encoder subset without decoder or rank-one banks."""

    def __init__(
        self,
        config: Any,
        layout: ModelFamilyLayout,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.local_encoders = nn.ModuleDict(
            {
                site: LocalSiteEncoder(
                    layout.hidden_size,
                    config,
                    device=device,
                    dtype=torch.float32,
                )
                for site in SITES
            }
        )
        self.shared_encoders = nn.ModuleDict(
            {
                site: SharedSiteEncoder(
                    config,
                    model_order=tuple(config.architecture.model_order),
                    device=device,
                    dtype=torch.float32,
                )
                for site in SITES
            }
        )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-0", type=Path, required=True)
    parser.add_argument("--checkpoint-1", type=Path, required=True)
    parser.add_argument("--feature-activity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic table: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _broadcast(value: Any) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "RESEARCHER_AUTHORIZED_READ_ONLY_LOGIT_DRIFT_REPLAY",
        "authorized_by": "researcher",
        "authorized_on": "2026-11-07",
        "experiment_family": "aspd_v3_granite_normalized_target_unconstrained_decoder_v1",
        "training_job_id": "50138545",
        "heldout_evaluation_job_id": "50148074",
        "checkpoint_steps": [1000, 1221],
        "split": "validation",
        "split_unit": "document",
        "pool_anchors": 2048,
        "pool_count": 1,
        "model_order": list(MODEL_ORDER_GRANITE),
        "factor_order": list(FACTOR_NAMES),
        "counterfactual_count": 16,
        "gpu_count": 3,
        "gpu_type": "b200",
        "read_only": True,
        "training_authorized_by_this_contract": False,
        "checkpoint_creation_authorized": False,
    }
    if payload != expected:
        raise ValueError("encoder-logit replay contract changed from researcher authorization")
    return payload


def _validate_inputs(args: argparse.Namespace, config: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    run_manifest_path = args.run_dir / "run_manifest.json"
    required = (
        args.config,
        args.contract,
        run_manifest_path,
        args.feature_activity,
        args.checkpoint_0 / "manifest.json",
        args.checkpoint_0 / "COMPLETE",
        args.checkpoint_1 / "manifest.json",
        args.checkpoint_1 / "COMPLETE",
        Path(config.runtime.dataset_manifest),
        Path(config.runtime.alignment_manifest),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"encoder-logit replay inputs missing: {missing}")
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.config)
    dataset = json.loads(Path(config.runtime.dataset_manifest).read_text(encoding="utf-8"))
    alignment = json.loads(Path(config.runtime.alignment_manifest).read_text(encoding="utf-8"))
    if run.get("slurm_job_id") != contract["training_job_id"]:
        raise ValueError("encoder-logit replay training job provenance changed")
    expected_run = {
        "config_fingerprint": config.fingerprint(),
        "config_file_sha256": config_sha,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "alignment_manifest_sha256": alignment["manifest_sha256"],
        "model_order": list(MODEL_ORDER_GRANITE),
        "logical_target_count": 168,
    }
    mismatch = {
        key: {"expected": value, "observed": run.get(key)}
        for key, value in expected_run.items()
        if run.get(key) != value
    }
    if mismatch:
        raise ValueError(f"encoder-logit replay run provenance mismatch: {mismatch}")
    manifests = []
    for checkpoint, step in zip(
        (args.checkpoint_0, args.checkpoint_1), contract["checkpoint_steps"], strict=True
    ):
        manifest_path = checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads((checkpoint / "COMPLETE").read_text(encoding="utf-8"))
        observed_sha = sha256_file(manifest_path)
        expected_checkpoint = {
            "status": "COMPLETE",
            "step": step,
            "world_size": 3,
            "config_fingerprint": config.fingerprint(),
            "config_file_sha256": config_sha,
            "dataset_manifest_sha256": dataset["manifest_sha256"],
        }
        checkpoint_mismatch = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected_checkpoint.items()
            if manifest.get(key) != value
        }
        if marker.get("manifest_sha256") != observed_sha:
            checkpoint_mismatch["marker.manifest_sha256"] = {
                "expected": observed_sha,
                "observed": marker.get("manifest_sha256"),
            }
        if checkpoint_mismatch:
            raise ValueError(
                f"encoder-logit replay checkpoint provenance mismatch: {checkpoint_mismatch}"
            )
        manifests.append({"step": step, "manifest_sha256": observed_sha})
    return {
        "training_git_sha": run["git_sha"],
        "config_fingerprint": config.fingerprint(),
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "alignment_manifest_sha256": alignment["manifest_sha256"],
        "checkpoint_manifests": manifests,
        "feature_activity_sha256": sha256_file(args.feature_activity),
    }


def _tensor(value: Any, path: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise TypeError(f"{path} did not expose a tensor")


def _capture_residuals(
    model: nn.Module,
    layout: ModelFamilyLayout,
    pool: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, tuple[Tensor, ...]]:
    """Capture only the 16 residual targets needed by the encoder replay."""

    assert_frozen_backbone(model)
    model_inputs, documents, tokens = padded_model_inputs(
        pool, pad_token_id=pad_token_id, device=device
    )
    residuals: dict[str, dict[int, Tensor]] = {"pre": {}, "mid": {}}

    class _SelectedLayersComplete(Exception):
        pass

    with ExitStack() as stack:
        for site, kind in (("pre", "residual_pre"), ("mid", "residual_mid")):
            for layer in layout.layers:
                path, module = resolve_submodule(model, layout.grounding_paths(kind, layer))

                def capture(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    capture_site: str = site,
                    capture_layer: int = layer,
                    module_path: str = path,
                ) -> None:
                    residuals[capture_site][capture_layer] = gather_anchor_axis(
                        _tensor(args, module_path), documents, tokens
                    )

                stack.callback(module.register_forward_pre_hook(capture).remove)
        deepest_path = f"{layout.layer_prefixes[0]}.{layout.layers[-1]}"
        deepest = model.get_submodule(deepest_path)

        def stop_after_deepest(
            _module: nn.Module, _args: tuple[Any, ...], _output: Any
        ) -> None:
            raise _SelectedLayersComplete

        stack.callback(deepest.register_forward_hook(stop_after_deepest).remove)
        with torch.inference_mode():
            try:
                model(**model_inputs, use_cache=False, return_dict=True)
            except _SelectedLayersComplete:
                pass
    result = {
        site: tuple(residuals[site][layer] for layer in layout.layers) for site in SITES
    }
    expected = (1, len(pool.anchor_token_indices), layout.hidden_size)
    if any(len(values) != 8 or any(tuple(value.shape) != expected for value in values) for values in result.values()):
        raise ValueError("residual-only replay capture has invalid shapes")
    return result


def _pad_id(tokenizer: Any) -> int:
    value = tokenizer.pad_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("tokenizer exposes neither padding nor EOS token")
    return int(value)


def _shared_digest(module: EncoderReplayState) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.shared_encoders.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_encoder_state(
    config: Any,
    layout: ModelFamilyLayout,
    local_key: str,
    checkpoint: Path,
    provenance: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[EncoderReplayState, dict[str, Any]]:
    state = EncoderReplayState(config, layout, device=device)
    manager = ModelOwnedCheckpoint(checkpoint.parent, keep_complete=2)
    runtime, manifest = manager.load_model_only(
        path=checkpoint,
        model=state,
        config_fingerprint=provenance["config_fingerprint"],
        config_file_sha256=provenance["config_sha256"],
        dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
        expected_model_key=local_key,
    )
    state.eval().requires_grad_(False)
    digest = _shared_digest(state)
    digests: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise ValueError("replicated shared encoder differs across replay ranks")
    return state, {"completed_steps": int(runtime["completed_steps"]), "manifest": manifest}


def _gather_tokens(local: Tensor, model_order: Sequence[str]) -> dict[str, Tensor]:
    gathered = [torch.empty_like(local) for _ in model_order]
    dist.all_gather(gathered, local)
    return {key: gathered[index] for index, key in enumerate(model_order)}


def _hidden_representation(shared: SharedSiteEncoder, tokens: Mapping[str, Tensor]) -> Tensor:
    packed = pack_cross_model_tokens(tokens, model_order=shared.model_order)
    encoded = shared.cross(shared.input_adapter(packed.values))
    return encoded.mean(dim=2)


def _logit_summary(logits: Tensor) -> dict[str, Any]:
    values = logits.float()
    positive = (values > 0).sum(dim=-1).float()
    sampled = values.reshape(-1)[::4096]
    quantiles = torch.quantile(
        sampled,
        torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=values.device),
    )
    candidate_quantiles = torch.quantile(
        positive,
        torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=values.device),
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "negative_fraction": float((values < 0).float().mean()),
        "zero_fraction": float((values == 0).float().mean()),
        "positive_fraction": float((values > 0).float().mean()),
        "sampled_quantiles": {
            key: float(value)
            for key, value in zip(("q01", "q10", "median", "q90", "q99"), quantiles, strict=True)
        },
        "positive_candidates_per_anchor": {
            "mean": float(positive.mean()),
            "std": float(positive.std(unbiased=False)),
            **{
                key: float(value)
                for key, value in zip(
                    ("q01", "q10", "median", "q90", "q99"),
                    candidate_quantiles,
                    strict=True,
                )
            },
        },
    }


def _feature_counts(path: Path) -> dict[str, Tensor]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for site in SITES:
        models = []
        for model in MODEL_ORDER_GRANITE:
            selected = sorted(
                (row for row in rows if row["site"] == site and row["model"] == model),
                key=lambda row: int(row["feature"]),
            )
            models.append(torch.tensor([int(row["heldout_fires"]) for row in selected]))
        if any(not torch.equal(models[0], value) for value in models[1:]):
            raise ValueError("held-out shared-gate counts disagree across models")
        result[site] = models[0]
    return result


def _group_masks(counts: Tensor) -> dict[str, Tensor]:
    top = torch.argsort(counts, descending=True)[:82]
    top_mask = torch.zeros_like(counts, dtype=torch.bool)
    top_mask[top] = True
    masks = {
        "all": torch.ones_like(counts, dtype=torch.bool),
        "top_82_heldout": top_mask,
        "heldout_active": counts > 0,
        "heldout_inactive": counts == 0,
    }
    for band in FEATURE_BANDS:
        mask = torch.zeros_like(counts, dtype=torch.bool)
        mask[band.lower : band.upper] = True
        masks[band.name] = mask
    return masks


def _site_replay(
    site: str,
    states: Sequence[EncoderReplayState],
    representations: Mapping[tuple[int, int], Tensor],
    native_endpoints: Mapping[int, Tensor],
    counts: Tensor,
    valid: Tensor,
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    combinations: dict[tuple[int, ...], dict[str, Any]] = {}
    negative_values: dict[tuple[int, ...], Tensor] = {}
    candidate_values: dict[tuple[int, ...], Tensor] = {}
    feature_negative: dict[tuple[int, ...], Tensor] = {}
    feature_mean: dict[tuple[int, ...], Tensor] = {}
    endpoint_logits: dict[int, Tensor] = {}
    endpoint_checks: dict[str, Any] = {}
    endpoint_batch_topk: dict[str, Any] = {}
    for local_index in range(2):
        for shared_index in range(2):
            hidden = representations[(local_index, shared_index)].float()
            for weight_index in range(2):
                weight = states[weight_index].shared_encoders[site].readout.output.weight.float()
                for bias_index in range(2):
                    bias = states[bias_index].shared_encoders[site].readout.output.bias.float()
                    key = (local_index, shared_index, weight_index, bias_index)
                    logits = hidden @ weight.T + bias
                    summary = _logit_summary(logits)
                    combinations[key] = summary
                    negative_values[key] = (logits < 0).float().mean()
                    candidate_values[key] = (logits > 0).sum(dim=-1).float().mean()
                    feature_negative[key] = (logits < 0).float().mean(dim=(0, 1)).cpu()
                    feature_mean[key] = logits.mean(dim=(0, 1)).cpu()
                    if len(set(key)) == 1:
                        endpoint = key[0]
                        endpoint_logits[endpoint] = logits.detach()
                        native = native_endpoints[endpoint].float()
                        difference = (native - logits).abs()
                        endpoint_checks[str(endpoint)] = {
                            "native_vs_fp32_mean_absolute_difference": float(difference.mean()),
                            "native_vs_fp32_max_absolute_difference": float(difference.max()),
                            "native_vs_fp32_sign_disagreement_fraction": float(
                                ((native > 0) != (logits > 0)).float().mean()
                            ),
                        }
                        controller = V3BatchTopK(
                            config.architecture.dictionary_size,
                            config.architecture.batch_topk_k,
                            pool_anchors=config.architecture.pool_tokens,
                            device=logits.device,
                            sync_statistics=False,
                            materialize_n_selected=False,
                            validate_nonnegative=False,
                        )
                        code = controller(torch.relu(native), valid, update_statistics=False)
                        selected = code.gate[valid]
                        endpoint_batch_topk[str(endpoint)] = {
                            "positive_firings_per_anchor": float(
                                selected.sum(dim=-1).float().mean()
                            ),
                            "distinct_features_firing": int(selected.any(dim=0).sum()),
                        }
    shapley_negative = factorial_shapley(negative_values)
    shapley_candidates = factorial_shapley(candidate_values)
    shapley_feature_negative = factorial_shapley(feature_negative)
    shapley_feature_mean = factorial_shapley(feature_mean)
    start = endpoint_logits[0]
    end = endpoint_logits[1]
    deaths = ((start > 0) & (end <= 0)).float().mean(dim=(0, 1)).cpu()
    revivals = ((start <= 0) & (end > 0)).float().mean(dim=(0, 1)).cpu()
    feature_rows = []
    for feature in range(config.architecture.dictionary_size):
        band = next(
            band.name for band in FEATURE_BANDS if band.lower <= feature < band.upper
        )
        row: dict[str, Any] = {
            "site": site,
            "feature": feature,
            "band": band,
            "heldout_fires": int(counts[feature]),
            "negative_fraction_step_1000": float(feature_negative[(0, 0, 0, 0)][feature]),
            "negative_fraction_step_1221": float(feature_negative[(1, 1, 1, 1)][feature]),
            "mean_logit_step_1000": float(feature_mean[(0, 0, 0, 0)][feature]),
            "mean_logit_step_1221": float(feature_mean[(1, 1, 1, 1)][feature]),
            "positive_to_nonpositive_probability": float(deaths[feature]),
            "nonpositive_to_positive_probability": float(revivals[feature]),
        }
        for index, factor in enumerate(FACTOR_NAMES):
            row[f"shapley_negative_fraction_{factor}"] = float(
                shapley_feature_negative[index][feature]
            )
            row[f"shapley_mean_logit_{factor}"] = float(shapley_feature_mean[index][feature])
        feature_rows.append(row)
    group_rows = []
    for group, mask in _group_masks(counts).items():
        selected = [feature_rows[index] for index in mask.nonzero(as_tuple=True)[0].tolist()]
        row = {"site": site, "group": group, "feature_count": len(selected)}
        fields = [
            key
            for key in feature_rows[0]
            if key.startswith("shapley_")
            or key
            in {
                "negative_fraction_step_1000",
                "negative_fraction_step_1221",
                "mean_logit_step_1000",
                "mean_logit_step_1221",
                "positive_to_nonpositive_probability",
                "nonpositive_to_positive_probability",
            }
        ]
        for field in fields:
            row[field] = sum(float(value[field]) for value in selected) / len(selected)
        group_rows.append(row)
    serialized_combinations = {
        "".join(str(value) for value in key): summary
        for key, summary in sorted(combinations.items())
    }
    result = {
        "site": site,
        "factor_order": list(FACTOR_NAMES),
        "combinations": serialized_combinations,
        "shapley_negative_fraction": {
            factor: float(shapley_negative[index])
            for index, factor in enumerate(FACTOR_NAMES)
        },
        "shapley_positive_candidates_per_anchor": {
            factor: float(shapley_candidates[index])
            for index, factor in enumerate(FACTOR_NAMES)
        },
        "endpoint_native_fp32_checks": endpoint_checks,
        "endpoint_batch_topk": endpoint_batch_topk,
        "endpoint_positive_to_nonpositive_fraction": float(
            ((start > 0) & (end <= 0)).float().mean()
        ),
        "endpoint_nonpositive_to_positive_fraction": float(
            ((start <= 0) & (end > 0)).float().mean()
        ),
    }
    return result, feature_rows, group_rows


def main() -> None:
    args = _args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 3:
        raise RuntimeError("encoder-logit replay requires exactly three ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()

    config = load_granite_normalized_target_config(args.config)
    contract = _load_contract(args.contract)
    if tuple(config.architecture.model_order) != MODEL_ORDER_GRANITE:
        raise ValueError("encoder-logit replay model order changed")
    provenance = _validate_inputs(args, config, contract)
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite replay output: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    registry = build_model_registry_2b(config)
    local_key = MODEL_ORDER_GRANITE[rank]
    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=args.local_files_only)
    loaded = load_frozen_v3_model(
        registry[local_key], device, local_files_only=args.local_files_only
    )
    assert_frozen_backbone(loaded.forward_model)

    stream = None
    if rank == 0:
        tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
            for key in MODEL_ORDER_GRANITE
        }
        dataset = load_pinned_pile_dataset(config.data.dataset_id, config.data.dataset_revision)
        stream = ValidationAlignedAnchorStream(
            dataset=dataset,
            dataset_id=config.data.dataset_id,
            dataset_revision=config.data.dataset_revision,
            tokenizers=tokenizers,
            tokenizer_revisions={
                key: registry[key].tokenizer_revision for key in MODEL_ORDER_GRANITE
            },
            split_policy=HashSplitPolicy(
                modulus=config.data.split_modulus,
                validation_buckets=frozenset(config.data.validation_buckets),
                test_buckets=frozenset(config.data.test_buckets),
                seed=config.data.split_seed,
            ),
            dedup_path=args.output_dir / "validation_stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
            model_order=MODEL_ORDER_GRANITE,
        )
        pool_payload = stream.next_pool(pool_anchors=contract["pool_anchors"]).to_payload()
    else:
        pool_payload = None
    pool = AlignedAnchorPool.from_payload(_broadcast(pool_payload))
    pool.validate(
        expected_anchors=contract["pool_anchors"],
        sequence_length=config.data.sequence_length,
        model_order=MODEL_ORDER_GRANITE,
    )
    residuals = _capture_residuals(
        loaded.forward_model,
        registry[local_key],
        pool.models[local_key],
        pad_token_id=_pad_id(tokenizer),
        device=device,
    )
    del loaded
    torch.cuda.empty_cache()

    states = []
    state_reports = []
    for checkpoint in (args.checkpoint_0, args.checkpoint_1):
        state, report = _load_encoder_state(
            config,
            registry[local_key],
            local_key,
            checkpoint,
            provenance,
            device=device,
        )
        states.append(state)
        state_reports.append(report)
    valid = torch.ones(1, pool.anchor_count, dtype=torch.bool, device=device)
    gathered_tokens: dict[tuple[int, str], dict[str, Tensor]] = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for local_index, state in enumerate(states):
            for site in SITES:
                local = state.local_encoders[site](residuals[site])
                gathered_tokens[(local_index, site)] = _gather_tokens(
                    local, MODEL_ORDER_GRANITE
                )
        representations: dict[tuple[int, int, str], Tensor] = {}
        native_endpoints: dict[tuple[int, str], Tensor] = {}
        for local_index in range(2):
            for shared_index, state in enumerate(states):
                for site in SITES:
                    tokens = gathered_tokens[(local_index, site)]
                    representations[(local_index, shared_index, site)] = _hidden_representation(
                        state.shared_encoders[site], tokens
                    )
                    if local_index == shared_index:
                        _scores, logits = state.shared_encoders[site].forward_with_logits(
                            tokens, valid
                        )
                        native_endpoints[(local_index, site)] = logits.detach()

    counts = _feature_counts(args.feature_activity) if rank == 0 else None
    count_payload = {
        site: value.tolist() for site, value in counts.items()
    } if rank == 0 and counts is not None else None
    shared_counts = {
        site: torch.tensor(value, dtype=torch.long)
        for site, value in _broadcast(count_payload).items()
    }
    site_reports = []
    feature_rows = []
    group_rows = []
    for site in SITES:
        report, site_features, site_groups = _site_replay(
            site,
            states,
            {
                (local_index, shared_index): representations[
                    (local_index, shared_index, site)
                ]
                for local_index in range(2)
                for shared_index in range(2)
            },
            {endpoint: native_endpoints[(endpoint, site)] for endpoint in range(2)},
            shared_counts[site],
            valid,
            config,
        )
        site_reports.append(report)
        feature_rows.extend(site_features)
        group_rows.extend(site_groups)

    consistency = torch.tensor(
        [
            site_reports[0]["combinations"]["0000"]["mean"],
            site_reports[0]["combinations"]["1111"]["mean"],
            site_reports[1]["combinations"]["0000"]["mean"],
            site_reports[1]["combinations"]["1111"]["mean"],
        ],
        device=device,
    )
    minimum = consistency.clone()
    maximum = consistency.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if float((maximum - minimum).abs().max()) > 1e-6:
        raise ValueError("replay endpoint summaries differ across ranks")

    if rank == 0:
        anchor_digest = hashlib.sha256(
            json.dumps(pool.anchor_ids, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "status": "READ_ONLY_FIXED_ANCHOR_LOGIT_DRIFT_REPLAY_COMPLETE",
            "diagnostic_git_sha": _git_sha(),
            "training_git_sha": provenance["training_git_sha"],
            "training_job_id": contract["training_job_id"],
            "config_fingerprint": provenance["config_fingerprint"],
            "config_sha256": provenance["config_sha256"],
            "contract_sha256": sha256_file(args.contract),
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "alignment_manifest_sha256": provenance["alignment_manifest_sha256"],
            "feature_activity_sha256": provenance["feature_activity_sha256"],
            "checkpoint_manifests": provenance["checkpoint_manifests"],
            "model_order": list(MODEL_ORDER_GRANITE),
            "factor_order": list(FACTOR_NAMES),
            "counterfactual_count": 16,
            "accepted_anchors": pool.anchor_count,
            "anchor_ids_sha256": anchor_digest,
            "split": contract["split"],
            "read_only": True,
            "optimizer_or_backward_used": False,
            "checkpoint_created": False,
            "state_reports": state_reports,
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(args.output_dir / "experiment_manifest.json", manifest)
        _write_json(
            args.output_dir / "counterfactual_metrics.json",
            {"schema_version": 1, "sites": site_reports},
        )
        _write_csv(args.output_dir / "feature_attribution.csv", feature_rows)
        _write_csv(args.output_dir / "group_attribution.csv", group_rows)
    if stream is not None:
        stream.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

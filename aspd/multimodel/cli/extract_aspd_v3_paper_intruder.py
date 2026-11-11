"""Extract natural v3.1 firings for the paper's Intruder evaluation protocol."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from aspd.eval.autointerp_db import sample_keys
from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _capture,
    _finite_state_report,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.diagnostic_stream import DiagnosticAlignedAnchorStream
from aspd.multimodel.v3.hooks import assert_frozen_backbone
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract, component_key
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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


def _broadcast(value: Any) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _validate_static_inputs(
    args: argparse.Namespace,
    config: Any,
    protocol: PaperIntruderContract,
) -> dict[str, Any]:
    if sha256_file(args.config) != protocol.source_config_sha256:
        raise ValueError("source training config hash differs from the Intruder contract")
    if tuple(config.architecture.model_order) != protocol.model_order:
        raise ValueError("source-model order differs from the Intruder contract")
    if any(
        forbidden in model.model_id.lower()
        for forbidden in protocol.forbidden_model_substrings
        for model in config.models
    ):
        raise ValueError("source-model roster contains a forbidden model family")
    if config.architecture.dictionary_size != 8192 or config.architecture.batch_topk_k != 32:
        raise ValueError("v3.1 dictionary or BatchTopK sparsity changed")
    if config.objective.gate_logit_normalization != "per_anchor_feature_zscore":
        raise ValueError("Intruder extraction requires the canonical z-score gate")
    if config.objective.gate_logit_normalization_epsilon != 1.0e-5:
        raise ValueError("z-score epsilon differs from the canonical v3.1 value")
    manifest_path = args.checkpoint / "manifest.json"
    marker_path = args.checkpoint / "COMPLETE"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError("checkpoint lacks manifest.json or COMPLETE")
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != protocol.checkpoint_manifest_sha256:
        raise ValueError("checkpoint manifest hash differs from the frozen evaluation input")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETE",
        "step": protocol.checkpoint_step,
        "world_size": len(protocol.model_order),
        "config_file_sha256": protocol.source_config_sha256,
        "config_fingerprint": config.fingerprint(),
    }
    mismatch = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if marker.get("manifest_sha256") != observed_manifest_sha:
        mismatch["COMPLETE.manifest_sha256"] = {
            "expected": observed_manifest_sha,
            "observed": marker.get("manifest_sha256"),
        }
    if mismatch:
        raise ValueError(f"checkpoint provenance mismatch: {mismatch}")
    dataset_manifest = Path(config.runtime.dataset_manifest)
    alignment_manifest = Path(config.runtime.alignment_manifest)
    if not dataset_manifest.is_file() or not alignment_manifest.is_file():
        raise FileNotFoundError("pinned dataset/alignment manifest is missing")
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    alignment_payload = json.loads(alignment_manifest.read_text(encoding="utf-8"))
    dataset_sha = dataset_payload.get("manifest_sha256")
    alignment_sha = alignment_payload.get("manifest_sha256")
    if not isinstance(dataset_sha, str) or not isinstance(alignment_sha, str):
        raise ValueError("dataset/alignment manifest lacks its canonical manifest_sha256")
    if manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("checkpoint and current dataset manifest hashes differ")
    return {
        "checkpoint_manifest": manifest,
        "checkpoint_manifest_sha256": observed_manifest_sha,
        "dataset_manifest_sha256": dataset_sha,
        "dataset_manifest_file_sha256": sha256_file(dataset_manifest),
        "alignment_manifest_sha256": alignment_sha,
        "alignment_manifest_file_sha256": sha256_file(alignment_manifest),
    }


def _context_rows_for_feature(
    *,
    pool: AlignedAnchorPool,
    model_key: str,
    feature_positions: np.ndarray,
    feature_values: np.ndarray,
    tokens_per_side: int,
) -> list[dict[str, Any]]:
    """Create one strongest annotated context per source document in this pool."""

    inputs = pool.models[model_key]
    by_document: dict[int, list[int]] = {}
    for event_index, position in enumerate(feature_positions):
        document_index = int(inputs.anchor_document_indices[int(position)])
        by_document.setdefault(document_index, []).append(event_index)
    rows: list[dict[str, Any]] = []
    for document_index, event_indices in by_document.items():
        peak_event = max(event_indices, key=lambda index: float(feature_values[index]))
        peak_position = int(feature_positions[peak_event])
        peak_token = int(inputs.anchor_token_indices[peak_position])
        document_tokens = inputs.input_ids[document_index]
        lower = max(0, peak_token - tokens_per_side)
        upper = min(len(document_tokens), peak_token + tokens_per_side + 1)
        token_activations: dict[int, float] = {}
        for event_index in event_indices:
            position = int(feature_positions[event_index])
            token_index = int(inputs.anchor_token_indices[position])
            token_activations[token_index] = max(
                token_activations.get(token_index, 0.0), float(feature_values[event_index])
            )
        activations = [float(token_activations.get(index, 0.0)) for index in range(lower, upper)]
        firings = [value > 0 for value in activations]
        if not any(firings):
            raise RuntimeError("retained Intruder context lost its natural firing anchor")
        rows.append(
            {
                "activation": float(feature_values[peak_event]),
                "anchor_id": pool.anchor_ids[peak_position],
                "document_id": pool.document_ids[document_index],
                "source_group": pool.source_observations[document_index].get(
                    "source_group", "unknown"
                ),
                "token_ids": [int(value) for value in document_tokens[lower:upper]],
                "firings": firings,
                "activations": activations,
            }
        )
    return rows


def _retain_contexts(
    retained: list[dict[str, dict[str, Any]]],
    *,
    code: torch.Tensor,
    pool: AlignedAnchorPool,
    model_key: str,
    tokens_per_side: int,
    limit: int,
    allowed_features: set[int] | None,
) -> np.ndarray:
    position, feature = (code > 0).nonzero(as_tuple=True)
    values = code[position, feature]
    position_np = position.cpu().numpy().astype(np.int32, copy=False)
    feature_np = feature.cpu().numpy().astype(np.int32, copy=False)
    values_np = values.cpu().numpy().astype(np.float32, copy=False)
    counts = np.bincount(feature_np, minlength=len(retained)).astype(np.int64, copy=False)
    order = np.argsort(feature_np, kind="stable")
    ordered = feature_np[order]
    start = 0
    while start < len(order):
        stop = start + 1
        feature_id = int(ordered[start])
        while stop < len(order) and int(ordered[stop]) == feature_id:
            stop += 1
        if allowed_features is None or feature_id in allowed_features:
            group = order[start:stop]
            rows = _context_rows_for_feature(
                pool=pool,
                model_key=model_key,
                feature_positions=position_np[group],
                feature_values=values_np[group],
                tokens_per_side=tokens_per_side,
            )
            by_document = retained[feature_id]
            for row in rows:
                document_id = str(row["document_id"])
                previous = by_document.get(document_id)
                if previous is not None and float(previous["activation"]) >= float(
                    row["activation"]
                ):
                    continue
                if previous is None and len(by_document) >= limit:
                    weakest_id, weakest = min(
                        by_document.items(), key=lambda item: float(item[1]["activation"])
                    )
                    if float(weakest["activation"]) >= float(row["activation"]):
                        continue
                    del by_document[weakest_id]
                by_document[document_id] = row
        start = stop
    return counts


def _make_stream(
    *,
    config: Any,
    registry: Any,
    local_key: str,
    tokenizer: Any,
    model_order: tuple[str, ...],
    output_dir: Path,
    pass_index: int,
    local_files_only: bool,
) -> DiagnosticAlignedAnchorStream:
    alignment_tokenizers = {
        key: tokenizer
        if key == local_key
        else load_v3_tokenizer(registry[key], local_files_only=local_files_only)
        for key in model_order
    }
    return DiagnosticAlignedAnchorStream(
        dataset=load_pinned_pile_dataset(config.data.dataset_id, config.data.dataset_revision),
        dataset_id=config.data.dataset_id,
        dataset_revision=config.data.dataset_revision,
        tokenizers=alignment_tokenizers,
        tokenizer_revisions={key: registry[key].tokenizer_revision for key in model_order},
        split_policy=HashSplitPolicy(
            modulus=config.data.split_modulus,
            validation_buckets=frozenset(config.data.validation_buckets),
            test_buckets=frozenset(config.data.test_buckets),
            seed=config.data.split_seed,
        ),
        dedup_path=output_dir / f"test_stream_pass{pass_index}_dedup.sqlite",
        selected_split="test",
        sequence_length=config.data.sequence_length,
        model_order=model_order,
    )


def main() -> None:
    args = _args()
    evaluation_git_sha = _git_sha()
    protocol = PaperIntruderContract.load(args.protocol)
    config = load_main500m_config(args.config)
    provenance = _validate_static_inputs(args, config, protocol)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != len(protocol.model_order):
        raise RuntimeError("paper Intruder extraction requires three model-owner ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=90))
    started = time.perf_counter()
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite extraction output: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    registry = build_model_registry_2b(config)
    model_order = tuple(config.architecture.model_order)
    local_key = model_order[rank]
    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=args.local_files_only)
    loaded = load_frozen_v3_model(
        registry[local_key], device, local_files_only=args.local_files_only
    )
    assert_frozen_backbone(loaded.forward_model)
    source_before = _parameter_signature(loaded.forward_model)
    model = ModelOwnedMain2B(
        config,
        registry,
        local_key,
        native_biases={target.key: None for target in registry[local_key].targets()},
        device=device,
        dtype=torch.float32,
        sparse_backend="dense",
    )
    manager = ModelOwnedCheckpoint(args.checkpoint.parent, keep_complete=2)
    runtime_state, checkpoint_manifest = manager.load_model_only(
        path=args.checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=protocol.source_config_sha256,
        dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
        expected_model_key=local_key,
    )
    model.eval().requires_grad_(False)
    finite = _finite_state_report(model)
    digest = _shared_digest(model)
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise ValueError("replicated shared encoders differ across checkpoint owners")
    runtime_states: list[Any] = [None] * world
    dist.all_gather_object(runtime_states, runtime_state)
    exposure_counts = {int(row["accepted_exposures"]) for row in runtime_states}
    if exposure_counts != {protocol.accepted_training_exposures}:
        raise ValueError(f"checkpoint exposure mismatch: {sorted(exposure_counts)}")

    n_features = config.architecture.dictionary_size
    pools = protocol.accepted_evaluation_anchors // protocol.pool_anchors
    first_retained = {
        site: [dict() for _ in range(n_features)] for site in ("pre", "mid")
    }
    counts = {
        site: np.zeros(n_features, dtype=np.int64) for site in ("pre", "mid")
    }

    def run_pass(
        pass_index: int,
        retained: dict[str, list[dict[str, dict[str, Any]]]],
        *,
        limit: int,
        allowed: dict[str, set[int]] | None,
        collect_counts: bool,
    ) -> None:
        stream = None
        if rank == 0:
            if local_key != protocol.display_model_key:
                raise RuntimeError("rank zero must own the configured display tokenizer")
            stream = _make_stream(
                config=config,
                registry=registry,
                local_key=local_key,
                tokenizer=tokenizer,
                model_order=model_order,
                output_dir=args.output_dir,
                pass_index=pass_index,
                local_files_only=args.local_files_only,
            )
        accepted = 0
        for _pool_index in range(pools):
            payload = (
                None
                if rank
                else stream.next_pool(pool_anchors=protocol.pool_anchors).to_payload()
            )
            pool = AlignedAnchorPool.from_payload(_broadcast(payload))
            capture = _capture(
                pool,
                local_key=local_key,
                tokenizer=tokenizer,
                loaded=loaded,
                registry=registry,
                device=device,
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _scores, sparse, _continuous, _binary = model.encode(
                    capture, update_statistics=False
                )
            if rank == 0:
                for site in ("pre", "mid"):
                    code = sparse[site].values[capture.valid_anchors].detach().float()
                    observed = _retain_contexts(
                        retained[site],
                        code=code,
                        pool=pool,
                        model_key=protocol.display_model_key,
                        tokens_per_side=protocol.context_tokens_per_side,
                        limit=limit,
                        allowed_features=None if allowed is None else allowed[site],
                    )
                    if collect_counts:
                        counts[site] += observed
            accepted += pool.anchor_count
            del capture, sparse
        if accepted != protocol.accepted_evaluation_anchors:
            raise ValueError("evaluation accepted-anchor budget changed")
        if stream is not None:
            stream.close()

    run_pass(
        1,
        first_retained,
        limit=protocol.minimum_examples_per_component,
        allowed=None,
        collect_counts=True,
    )
    sampled_keys = None
    allowed = None
    if rank == 0:
        eligible_keys = sorted(
            component_key(site, feature)
            for site in ("pre", "mid")
            for feature in range(n_features)
            if len(first_retained[site][feature]) >= protocol.minimum_examples_per_component
        )
        if len(eligible_keys) < protocol.n_subsample:
            raise RuntimeError(
                f"only {len(eligible_keys)} components are paper-Intruder eligible"
            )
        sampled_keys = sample_keys(eligible_keys, protocol.n_subsample, protocol.seed)
        allowed = {"pre": set(), "mid": set()}
        for key in sampled_keys:
            site, feature = key.split("/", 1)
            allowed[site].add(int(feature))
    sampled_keys = _broadcast(sampled_keys)
    allowed = _broadcast(allowed)
    target_retained = {
        site: [dict() for _ in range(n_features)] for site in ("pre", "mid")
    }
    run_pass(
        2,
        target_retained,
        limit=protocol.activation_examples_per_target,
        allowed=allowed,
        collect_counts=False,
    )

    if _parameter_signature(loaded.forward_model) != source_before:
        raise RuntimeError("frozen source model changed during Intruder extraction")
    local_report = {
        "rank": rank,
        "model": local_key,
        "finite_state": finite,
        "shared_encoder_sha256": digest,
        "source_weights_unchanged": "PASS",
    }
    rank_reports: list[Any] = [None] * world
    dist.all_gather_object(rank_reports, local_report)
    if rank == 0:
        sampled_set = set(sampled_keys)
        contexts_path = args.output_dir / "component_contexts.jsonl"
        eligible_count = 0
        with contexts_path.open("x", encoding="utf-8") as handle:
            for site in ("pre", "mid"):
                for feature in range(n_features):
                    key = component_key(site, feature)
                    contexts = (
                        target_retained[site][feature]
                        if key in sampled_set
                        else first_retained[site][feature]
                    )
                    if len(contexts) < protocol.minimum_examples_per_component:
                        continue
                    eligible_count += 1
                    ordered_contexts = sorted(
                        contexts.values(),
                        key=lambda row: (-float(row["activation"]), row["anchor_id"]),
                    )
                    handle.write(
                        json.dumps(
                            {
                                "component_key": key,
                                "site": site,
                                "feature": feature,
                                "firing_count": int(counts[site][feature]),
                                "firing_density": float(
                                    counts[site][feature]
                                    / protocol.accepted_evaluation_anchors
                                ),
                                "sampled_target": key in sampled_set,
                                "contexts": ordered_contexts,
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        _write_json(
            args.output_dir / "sampled_keys.json",
            {
                "seed": protocol.seed,
                "n_subsample": protocol.n_subsample,
                "n_eligible": eligible_count,
                "keys": sampled_keys,
            },
        )
        manifest = {
            "schema_version": 1,
            "status": "ASPD_V3_1_PAPER_INTRUDER_EXTRACTION_COMPLETE",
            "experiment_family": protocol.experiment_family,
            "selected_split": protocol.selected_split,
            "accepted_evaluation_anchors": protocol.accepted_evaluation_anchors,
            "natural_gate_only": True,
            "gate": "per-anchor feature-axis z-score -> ReLU -> pool-wide BatchTopK",
            "forced_activations": 0,
            "checkpoint_step": int(checkpoint_manifest["step"]),
            "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
            "accepted_training_exposures": protocol.accepted_training_exposures,
            "model_order": list(model_order),
            "dictionary_size_per_site": n_features,
            "eligible_components": eligible_count,
            "sampled_components": len(sampled_keys),
            "display_model_key": protocol.display_model_key,
            "source_model_provenance": {
                key: {
                    "model_id": registry[key].model_id,
                    "revision": registry[key].revision,
                    "tokenizer_revision": registry[key].tokenizer_revision,
                }
                for key in model_order
            },
            "training_git_sha": "6cf7d0761f15e3115d535b413c3ec9675182f318",
            "evaluation_git_sha": evaluation_git_sha,
            "config_sha256": protocol.source_config_sha256,
            "config_fingerprint": config.fingerprint(),
            "protocol_sha256": sha256_file(args.protocol),
            "protocol_fingerprint": protocol.fingerprint(),
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "alignment_manifest_sha256": provenance["alignment_manifest_sha256"],
            "component_contexts_sha256": sha256_file(contexts_path),
            "sampled_keys_sha256": sha256_file(args.output_dir / "sampled_keys.json"),
            "rank_reports": rank_reports,
            "paper_exact": {
                "intruder_trial_shape": True,
                "component_subsample": True,
                "seed": True,
                "density_matching": True,
                "context_window": True,
                "target_example_cap": True,
                "judge": True,
            },
            "v3_required_adaptations": {
                "firing_semantics": "natural z-score ReLU pool-wide BatchTopK",
                "shared_sites": ["pre", "mid"],
                "display_tokenizer": protocol.display_model_key,
                "split": "document-disjoint test rather than paper training harvest",
                "evaluation_anchor_budget": protocol.accepted_evaluation_anchors,
                "donor_example_cap": protocol.minimum_examples_per_component,
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
            },
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(args.output_dir / "experiment_manifest.json", manifest)
        _write_json(
            args.output_dir / "COMPLETE",
            {
                "status": "COMPLETE",
                "manifest_sha256": sha256_file(args.output_dir / "experiment_manifest.json"),
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

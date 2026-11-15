"""Extract document-disjoint firing and non-firing contexts for feature coherence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _capture,
    _finite_state_report,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.cli.extract_aspd_v3_paper_intruder import _validate_static_inputs
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.diagnostic_stream import DiagnosticAlignedAnchorStream
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceContract
from aspd.multimodel.v3.hooks import assert_frozen_backbone
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b

Mode = Literal["discovery-new", "confirmatory"]
_U64_MASK = (1 << 64) - 1
_FEATURE_MIX = 0x9E3779B97F4A7C15
_ORDINARY_MIX = 0x243F6A8885A308D3
_HARD_MIX = 0x13198A2E03707344


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("discovery-new", "confirmatory"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _broadcast(value: Any) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _load_preparation(
    directory: Path, contract: FeatureCoherenceContract
) -> tuple[list[dict[str, Any]], frozenset[str], dict[str, Any]]:
    if (directory / "PREPARED").read_text(encoding="utf-8").strip() != (
        "ASPD_V3_1_FEATURE_COHERENCE_PREPARED"
    ):
        raise ValueError("feature-coherence preparation marker is invalid")
    hashes = json.loads((directory / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"prepared artifact changed: {name}")
    manifest = json.loads((directory / "preparation_manifest.json").read_text())
    if manifest["contract_fingerprint"] != contract.fingerprint():
        raise ValueError("preparation and extraction contracts differ")
    selected = json.loads((directory / "selected_components.json").read_text())
    if len(selected) != 160:
        raise ValueError("prepared component sample no longer contains 160 keys")
    denylist = json.loads((directory / "discovery_document_denylist.json").read_text())
    document_ids = frozenset(str(value) for value in denylist["document_ids"])
    if len(document_ids) != denylist["document_count"] or len(document_ids) != 313:
        raise ValueError("discovery document denylist changed")
    return selected, document_ids, manifest


def _anchor_priority_base(anchor_ids: list[str]) -> np.ndarray:
    return np.fromiter(
        (
            int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
            for value in anchor_ids
        ),
        dtype=np.uint64,
        count=len(anchor_ids),
    )


def _anchor_priorities(
    anchor_ids: list[str], feature: int, kind: str
) -> np.ndarray:
    return _mixed_anchor_priorities(_anchor_priority_base(anchor_ids), feature, kind)


def _mixed_anchor_priorities(
    base: np.ndarray, feature: int, kind: str
) -> np.ndarray:
    kind_mix = _ORDINARY_MIX if kind == "ordinary" else _HARD_MIX
    mixed = (feature * _FEATURE_MIX + kind_mix) & _U64_MASK
    return base ^ np.uint64(mixed)


def _canonical_boundaries(pool: AlignedAnchorPool) -> list[int]:
    boundaries = [
        int(boundary)
        for observation in pool.source_observations
        for boundary in observation["anchor_character_boundaries"]
    ]
    if len(boundaries) != pool.anchor_count:
        raise ValueError("canonical boundary count differs from aligned anchors")
    return boundaries


def _render_context(
    *,
    pool: AlignedAnchorPool,
    position: int,
    model_key: str,
    tokenizer: Any,
    characters_per_side: int,
    boundaries: list[int],
    offset_cache: dict[str, list[tuple[int, int]]],
) -> dict[str, Any]:
    inputs = pool.models[model_key]
    document_index = int(inputs.anchor_document_indices[position])
    token_index = int(inputs.anchor_token_indices[position])
    observation = pool.source_observations[document_index]
    document_id = str(pool.document_ids[document_index])
    canonical_text = str(observation["canonical_text"])
    boundary = boundaries[position]
    if document_id not in offset_cache:
        encoded = tokenizer(
            canonical_text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=len(inputs.input_ids[document_index]),
        )
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        offset_cache[document_id] = [(int(start), int(end)) for start, end in offsets]
    offsets = offset_cache[document_id]
    if not 0 <= token_index < len(offsets):
        raise ValueError("aligned anchor token is outside display-tokenizer offsets")
    anchor_start, anchor_end = offsets[token_index]
    if anchor_start >= anchor_end or anchor_end != boundary:
        raise ValueError("display-tokenizer offset does not reproduce canonical boundary")
    lower = max(0, boundary - characters_per_side)
    upper = min(len(canonical_text), boundary + characters_per_side)
    relative_start = anchor_start - lower
    relative_end = anchor_end - lower
    context = canonical_text[lower:upper]
    marked = (
        context[:relative_start]
        + "⟪ANCHOR⟫"
        + context[relative_start:relative_end]
        + "⟪/ANCHOR⟫"
        + context[relative_end:]
    )
    return {
        "anchor_id": str(pool.anchor_ids[position]),
        "document_id": document_id,
        "source_group": str(observation.get("source_group", "unknown")),
        "model_token_index": token_index,
        "canonical_character_boundary": boundary,
        "canonical_anchor_start": anchor_start,
        "canonical_anchor_end": anchor_end,
        "canonical_anchor_text": canonical_text[anchor_start:anchor_end],
        "canonical_context_start": lower,
        "canonical_context_end": upper,
        "canonical_context": context,
        "marked_canonical_context": marked,
    }


def _retain_positive(
    retained: dict[str, dict[str, Any]],
    *,
    feature: int,
    score: np.ndarray,
    selected: np.ndarray,
    pool: AlignedAnchorPool,
    tokenizer: Any,
    characters_per_side: int,
    boundaries: list[int],
    offset_cache: dict[str, list[tuple[int, int]]],
    limit: int,
) -> int:
    positions = np.flatnonzero(selected[:, feature] > 0)
    by_document: dict[str, int] = {}
    for position in positions:
        document_index = int(pool.models["gemma"].anchor_document_indices[position])
        document_id = str(pool.document_ids[document_index])
        prior = by_document.get(document_id)
        if prior is None or selected[position, feature] > selected[prior, feature]:
            by_document[document_id] = int(position)
    for document_id, position in by_document.items():
        activation = float(selected[position, feature])
        previous = retained.get(document_id)
        if previous is not None and float(previous["activation"]) >= activation:
            continue
        if previous is None and len(retained) >= limit:
            weakest_id, weakest = min(
                retained.items(), key=lambda item: float(item[1]["activation"])
            )
            if float(weakest["activation"]) >= activation:
                continue
            del retained[weakest_id]
        row = _render_context(
            pool=pool,
            position=position,
            model_key="gemma",
            tokenizer=tokenizer,
            characters_per_side=characters_per_side,
            boundaries=boundaries,
            offset_cache=offset_cache,
        )
        row.update(
            {
                "activation": activation,
                "pre_topk_score": float(score[position, feature]),
                "selected_natural_firing": True,
            }
        )
        retained[document_id] = row
    return len(positions)


def _retain_negative_candidates(
    retained: dict[str, dict[str, Any]],
    *,
    feature: int,
    kind: Literal["ordinary", "hard"],
    score: np.ndarray,
    selected: np.ndarray,
    pool: AlignedAnchorPool,
    tokenizer: Any,
    characters_per_side: int,
    boundaries: list[int],
    offset_cache: dict[str, list[tuple[int, int]]],
    anchor_priority_base: np.ndarray,
    limit: int,
) -> int:
    if kind == "ordinary":
        mask = (score[:, feature] <= 0) & (selected[:, feature] == 0)
    else:
        mask = (score[:, feature] > 0) & (selected[:, feature] == 0)
    positions = np.flatnonzero(mask)
    priorities = _mixed_anchor_priorities(anchor_priority_base, feature, kind)
    by_document: dict[str, tuple[int, int]] = {}
    for position in positions:
        document_index = int(pool.models["gemma"].anchor_document_indices[position])
        document_id = str(pool.document_ids[document_index])
        candidate = (int(priorities[position]), int(position))
        if document_id not in by_document or candidate < by_document[document_id]:
            by_document[document_id] = candidate
    for document_id, (priority, position) in by_document.items():
        previous = retained.get(document_id)
        if previous is not None and int(previous["sampling_priority_u64"]) <= priority:
            continue
        if previous is None and len(retained) >= limit:
            worst_id, worst = max(
                retained.items(), key=lambda item: int(item[1]["sampling_priority_u64"])
            )
            if int(worst["sampling_priority_u64"]) <= priority:
                continue
            del retained[worst_id]
        row = _render_context(
            pool=pool,
            position=position,
            model_key="gemma",
            tokenizer=tokenizer,
            characters_per_side=characters_per_side,
            boundaries=boundaries,
            offset_cache=offset_cache,
        )
        row.update(
            {
                "activation": 0.0,
                "pre_topk_score": float(score[position, feature]),
                "selected_natural_firing": False,
                "negative_type": kind,
                "sampling_priority_u64": priority,
            }
        )
        retained[document_id] = row
    return len(positions)


def main() -> None:
    args = _args()
    evaluation_git_sha = _git_sha()
    contract = FeatureCoherenceContract.load(args.protocol)
    config = load_main500m_config(args.config)
    provenance = _validate_static_inputs(args, config, contract)
    selected_rows, discovery_documents, preparation = _load_preparation(
        args.preparation_dir, contract
    )
    selected_rows = [
        row
        for row in selected_rows
        if args.mode == "confirmatory" or row["cohort"] == "new_random"
    ]
    selected_by_site = {
        site: sorted(int(row["feature"]) for row in selected_rows if row["site"] == site)
        for site in ("pre", "mid")
    }
    selected_split = (
        contract.discovery_split if args.mode == "discovery-new" else contract.confirmatory_split
    )
    excluded_documents = (
        frozenset() if args.mode == "discovery-new" else discovery_documents
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != len(contract.model_order):
        raise RuntimeError("feature-coherence extraction requires three model-owner ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=90))
    started = time.perf_counter()
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite extraction: {args.output_dir}")
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
        config_file_sha256=contract.source_config_sha256,
        dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
        expected_model_key=local_key,
    )
    model.eval().requires_grad_(False)
    finite = _finite_state_report(model)
    digest = _shared_digest(model)
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise ValueError("replicated shared encoders differ across model owners")
    runtime_states: list[Any] = [None] * world
    dist.all_gather_object(runtime_states, runtime_state)
    exposures = {int(row["accepted_exposures"]) for row in runtime_states}
    if exposures != {contract.accepted_training_exposures}:
        raise ValueError(f"checkpoint exposure mismatch: {sorted(exposures)}")

    stream = None
    if rank == 0:
        if local_key != "gemma":
            raise RuntimeError("rank zero must own the canonical display tokenizer")
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
            for key in model_order
        }
        stream = DiagnosticAlignedAnchorStream(
            dataset=load_pinned_pile_dataset(
                config.data.dataset_id, config.data.dataset_revision
            ),
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
            dedup_path=args.output_dir / f"{selected_split}_stream_dedup.sqlite",
            selected_split=selected_split,
            excluded_document_ids=excluded_documents,
            sequence_length=config.data.sequence_length,
            model_order=model_order,
        )

    positives = {
        site: {feature: {} for feature in features}
        for site, features in selected_by_site.items()
    }
    ordinary = {
        site: {feature: {} for feature in features}
        for site, features in selected_by_site.items()
    }
    hard = {
        site: {feature: {} for feature in features}
        for site, features in selected_by_site.items()
    }
    firing_counts = {
        site: {feature: 0 for feature in features}
        for site, features in selected_by_site.items()
    }
    ordinary_candidate_counts = {
        site: {feature: 0 for feature in features}
        for site, features in selected_by_site.items()
    }
    hard_candidate_counts = {
        site: {feature: 0 for feature in features}
        for site, features in selected_by_site.items()
    }
    observed_documents: set[str] = set()
    pools = contract.accepted_anchors_per_extraction // contract.pool_anchors
    accepted = 0
    for _pool_index in range(pools):
        payload = (
            None
            if rank
            else stream.next_pool(pool_anchors=contract.pool_anchors).to_payload()
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
            scores, sparse, _continuous, _binary = model.encode(
                capture, update_statistics=False
            )
        if rank == 0:
            observed_documents.update(str(value) for value in pool.document_ids)
            boundaries = _canonical_boundaries(pool)
            priority_base = _anchor_priority_base(pool.anchor_ids)
            offset_cache: dict[str, list[tuple[int, int]]] = {}
            for site in ("pre", "mid"):
                score_np = (
                    scores[site][capture.valid_anchors].detach().float().cpu().numpy()
                )
                selected_np = (
                    sparse[site]
                    .values[capture.valid_anchors]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                for feature in selected_by_site[site]:
                    firing_counts[site][feature] += _retain_positive(
                        positives[site][feature],
                        feature=feature,
                        score=score_np,
                        selected=selected_np,
                        pool=pool,
                        tokenizer=tokenizer,
                        characters_per_side=contract.canonical_context_characters_per_side,
                        boundaries=boundaries,
                        offset_cache=offset_cache,
                        limit=contract.contexts_per_component,
                    )
                    if args.mode == "confirmatory":
                        ordinary_candidate_counts[site][feature] += (
                            _retain_negative_candidates(
                                ordinary[site][feature],
                                feature=feature,
                                kind="ordinary",
                                score=score_np,
                                selected=selected_np,
                                pool=pool,
                                tokenizer=tokenizer,
                                characters_per_side=(
                                    contract.canonical_context_characters_per_side
                                ),
                                boundaries=boundaries,
                                offset_cache=offset_cache,
                                anchor_priority_base=priority_base,
                                limit=contract.negative_candidates_per_component,
                            )
                        )
                        hard_candidate_counts[site][feature] += _retain_negative_candidates(
                            hard[site][feature],
                            feature=feature,
                            kind="hard",
                            score=score_np,
                            selected=selected_np,
                            pool=pool,
                            tokenizer=tokenizer,
                            characters_per_side=contract.canonical_context_characters_per_side,
                            boundaries=boundaries,
                            offset_cache=offset_cache,
                            anchor_priority_base=priority_base,
                            limit=contract.hard_negative_candidates_per_component,
                        )
        accepted += pool.anchor_count
        del capture, scores, sparse

    if accepted != contract.accepted_anchors_per_extraction:
        raise ValueError("accepted anchor budget changed")
    if excluded_documents.intersection(observed_documents):
        raise ValueError("discovery document leaked into confirmatory extraction")
    if _parameter_signature(loaded.forward_model) != source_before:
        raise RuntimeError("frozen source model changed during extraction")
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
        contexts_path = args.output_dir / "component_contexts.jsonl"
        with contexts_path.open("x", encoding="utf-8") as handle:
            for row in selected_rows:
                site = str(row["site"])
                feature = int(row["feature"])
                payload = {
                    **row,
                    "selected_split": selected_split,
                    "natural_firing_event_count": firing_counts[site][feature],
                    "positive_contexts": sorted(
                        positives[site][feature].values(),
                        key=lambda value: (
                            -float(value["activation"]),
                            str(value["anchor_id"]),
                        ),
                    ),
                    "ordinary_negative_candidates": sorted(
                        ordinary[site][feature].values(),
                        key=lambda value: int(value["sampling_priority_u64"]),
                    ),
                    "hard_negative_candidates": sorted(
                        hard[site][feature].values(),
                        key=lambda value: int(value["sampling_priority_u64"]),
                    ),
                    "ordinary_negative_event_count": ordinary_candidate_counts[site][feature],
                    "hard_negative_event_count": hard_candidate_counts[site][feature],
                }
                handle.write(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        manifest = {
            "schema_version": 1,
            "status": "ASPD_V3_1_FEATURE_COHERENCE_EXTRACTION_COMPLETE",
            "mode": args.mode,
            "selected_split": selected_split,
            "selected_components": len(selected_rows),
            "accepted_anchors": accepted,
            "unique_source_documents": len(observed_documents),
            "excluded_discovery_documents": len(excluded_documents),
            "excluded_documents_disjoint": not bool(
                excluded_documents.intersection(observed_documents)
            ),
            "natural_firing_definition": "actual sparse BatchTopK selected value > 0",
            "ordinary_negative_definition": "post-ReLU score <= 0 and not selected",
            "hard_negative_definition": "post-ReLU score > 0 and not selected",
            "forced_activations": 0,
            "checkpoint_step": int(checkpoint_manifest["step"]),
            "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
            "accepted_training_exposures": contract.accepted_training_exposures,
            "model_order": list(model_order),
            "source_model_provenance": {
                key: {
                    "model_id": registry[key].model_id,
                    "revision": registry[key].revision,
                    "tokenizer_revision": registry[key].tokenizer_revision,
                }
                for key in model_order
            },
            "evaluation_git_sha": evaluation_git_sha,
            "config_sha256": contract.source_config_sha256,
            "config_fingerprint": config.fingerprint(),
            "protocol_sha256": sha256_file(args.protocol),
            "protocol_fingerprint": contract.fingerprint(),
            "preparation_manifest_sha256": sha256_file(
                args.preparation_dir / "preparation_manifest.json"
            ),
            "preparation_source_git_sha": preparation["source_git_sha"],
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "alignment_manifest_sha256": provenance["alignment_manifest_sha256"],
            "component_contexts_sha256": sha256_file(contexts_path),
            "rank_reports": rank_reports,
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
                "manifest_sha256": sha256_file(
                    args.output_dir / "experiment_manifest.json"
                ),
            },
        )
    if stream is not None:
        stream.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Extract natural gate firings and blinded semantic contexts from ASPD v3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sqlite3
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _capture,
    _finite_state_report,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import _validate_inputs
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.diagnostic_stream import DiagnosticAlignedAnchorStream
from aspd.multimodel.v3.gate_collapse_recovery_config import (
    load_gate_collapse_recovery_config,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.mechanistic_validation import (
    FinalIntruderValidationContract,
    HumanInterpretabilityContract,
    MechanisticValidationContract,
)
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validation-smoke", "test"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
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


ValidationContract = (
    MechanisticValidationContract
    | FinalIntruderValidationContract
    | HumanInterpretabilityContract
)


def _load_protocol(path: Path) -> ValidationContract:
    import yaml

    family = yaml.safe_load(path.read_text(encoding="utf-8")).get("experiment_family")
    if family == "aspd_v3_zscore_mechanistic_validation_v1":
        return MechanisticValidationContract.load(path)
    if family == "aspd_v3_zscore_intruder_final_v1":
        return FinalIntruderValidationContract.load(path)
    if family == "aspd_v3_human_interpretability_v1":
        return HumanInterpretabilityContract.load(path)
    raise ValueError(f"unsupported mechanistic protocol family: {family!r}")


def _broadcast(value: Any) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _exposure_report(run_dir: Path, protocol: ValidationContract) -> dict[str, int]:
    path = run_dir / "exposure_ledger.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"exposure ledger is missing: {path}")
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT accepted, newly_observed, repeated, optimization_steps "
            "FROM exposure_totals"
        ).fetchone()
    if row is None:
        raise ValueError("exposure ledger has no totals row")
    report = {
        "accepted": int(row[0]),
        "newly_observed": int(row[1]),
        "repeated": int(row[2]),
        "optimization_steps": int(row[3]),
    }
    expected = {
        "accepted": protocol.accepted_training_exposures,
        "newly_observed": protocol.accepted_training_exposures,
        "repeated": 0,
        "optimization_steps": protocol.checkpoint_steps,
    }
    if report != expected:
        raise ValueError(f"training exposure mismatch: expected={expected}, observed={report}")
    return report


def _decode_context(
    tokenizer: Any,
    pool: AlignedAnchorPool,
    anchor_index: int,
    *,
    model_key: str,
    tokens_per_side: int,
) -> str:
    inputs = pool.models[model_key]
    document_index = int(inputs.anchor_document_indices[anchor_index])
    token_index = int(inputs.anchor_token_indices[anchor_index])
    token_ids = inputs.input_ids[document_index]
    lower = max(0, token_index - tokens_per_side)
    upper = min(len(token_ids), token_index + tokens_per_side + 1)
    return tokenizer.decode(
        token_ids[lower:upper],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _update_contexts(
    retained: list[dict[str, dict[str, Any]]],
    *,
    features: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    pool: AlignedAnchorPool,
    pool_anchor_offset: int,
    tokenizer: Any,
    model_key: str,
    tokens_per_side: int,
    canonical_characters_per_side: int | None,
    anchor_span_cache: dict[tuple[str, int, int], tuple[int, int, str]],
    limit: int,
) -> None:
    """Retain the strongest activation per document, bounded to ``limit`` documents."""

    canonical_boundaries = None
    if canonical_characters_per_side is not None:
        canonical_boundaries = [
            int(boundary)
            for row in pool.source_observations
            for boundary in row["anchor_character_boundaries"]
        ]
        if len(canonical_boundaries) != pool.anchor_count:
            raise ValueError("canonical anchor boundaries do not match pool anchors")
    order = np.argsort(features, kind="stable")
    ordered_features = features[order]
    start = 0
    while start < len(order):
        stop = start + 1
        feature = int(ordered_features[start])
        while stop < len(order) and int(ordered_features[stop]) == feature:
            stop += 1
        group = order[start:stop]
        if len(group) > limit:
            group = group[np.argpartition(values[group], -limit)[-limit:]]
        for event in group:
            anchor_index = int(positions[event])
            document_index = int(
                pool.models[model_key].anchor_document_indices[anchor_index]
            )
            document_id = pool.document_ids[document_index]
            value = float(values[event])
            by_document = retained[feature]
            previous = by_document.get(document_id)
            if previous is not None and float(previous["activation"]) >= value:
                continue
            if previous is None and len(by_document) >= limit:
                weakest_document, weakest = min(
                    by_document.items(), key=lambda item: float(item[1]["activation"])
                )
                if float(weakest["activation"]) >= value:
                    continue
                del by_document[weakest_document]
            observation = pool.source_observations[document_index]
            context = {
                "activation": value,
                "anchor_index": pool_anchor_offset + anchor_index,
                "anchor_id": pool.anchor_ids[anchor_index],
                "document_id": document_id,
                "source_group": observation.get("source_group", "unknown"),
                "model_token_index": int(
                    pool.models[model_key].anchor_token_indices[anchor_index]
                ),
                "context": _decode_context(
                    tokenizer,
                    pool,
                    anchor_index,
                    model_key=model_key,
                    tokens_per_side=tokens_per_side,
                ),
            }
            if canonical_characters_per_side is not None:
                assert canonical_boundaries is not None
                canonical_text = str(observation["canonical_text"])
                boundary = canonical_boundaries[anchor_index]
                cache_key = (document_id, int(context["model_token_index"]), boundary)
                if cache_key not in anchor_span_cache:
                    encoded = tokenizer(
                        canonical_text,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_offsets_mapping=True,
                        truncation=True,
                        max_length=len(pool.models[model_key].input_ids[document_index]),
                    )
                    offsets = encoded["offset_mapping"]
                    if hasattr(offsets, "tolist"):
                        offsets = offsets.tolist()
                    if offsets and isinstance(offsets[0], list) and len(offsets) == 1:
                        offsets = offsets[0]
                    token_index = int(context["model_token_index"])
                    if not 0 <= token_index < len(offsets):
                        raise ValueError("anchor token index is outside tokenizer offsets")
                    anchor_start, anchor_end = (int(value) for value in offsets[token_index])
                    if anchor_start >= anchor_end or anchor_end != boundary:
                        raise ValueError(
                            "tokenizer offset does not reproduce the aligned canonical boundary"
                        )
                    anchor_span_cache[cache_key] = (
                        anchor_start,
                        anchor_end,
                        canonical_text[anchor_start:anchor_end],
                    )
                anchor_start, anchor_end, anchor_text = anchor_span_cache[cache_key]
                lower = max(0, boundary - canonical_characters_per_side)
                upper = min(len(canonical_text), boundary + canonical_characters_per_side)
                context.update(
                    {
                        "canonical_character_boundary": boundary,
                        "canonical_anchor_start": anchor_start,
                        "canonical_anchor_end": anchor_end,
                        "canonical_anchor_text": anchor_text,
                        "canonical_context_start": lower,
                        "canonical_context_end": upper,
                        "canonical_context": canonical_text[lower:upper],
                    }
                )
            by_document[document_id] = context
        start = stop


def _save_site(
    output_dir: Path,
    site: str,
    *,
    row_counts: list[np.ndarray],
    indices: list[np.ndarray],
    values: list[np.ndarray],
    retained: list[dict[str, dict[str, Any]]],
    accepted_anchors: int,
) -> dict[str, Any]:
    counts = np.concatenate(row_counts).astype(np.int16, copy=False)
    if len(counts) != accepted_anchors:
        raise ValueError(f"{site} sparse row count differs from accepted anchors")
    indptr = np.empty(accepted_anchors + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=indptr[1:])
    feature_indices = np.concatenate(indices).astype(np.uint16, copy=False)
    coefficients = np.concatenate(values).astype(np.float32, copy=False)
    if int(indptr[-1]) != len(feature_indices) or len(feature_indices) != len(coefficients):
        raise ValueError(f"{site} CSR arrays disagree")
    sparse_path = output_dir / f"{site}_natural_firings.npz"
    np.savez_compressed(
        sparse_path,
        indptr=indptr,
        indices=feature_indices,
        values=coefficients,
        shape=np.asarray([accepted_anchors, len(retained)], dtype=np.int64),
    )
    contexts_path = output_dir / f"{site}_feature_contexts.jsonl"
    with contexts_path.open("x", encoding="utf-8") as handle:
        for feature, by_document in enumerate(retained):
            contexts = sorted(
                by_document.values(), key=lambda row: (-float(row["activation"]), row["anchor_id"])
            )
            handle.write(
                json.dumps(
                    {"site": site, "feature": feature, "contexts": contexts},
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    feature_counts = np.bincount(feature_indices, minlength=len(retained))
    return {
        "site": site,
        "accepted_anchors": accepted_anchors,
        "natural_firing_events": int(len(feature_indices)),
        "features_firing": int((feature_counts > 0).sum()),
        "natural_firings_sha256": sha256_file(sparse_path),
        "feature_contexts_sha256": sha256_file(contexts_path),
    }


def main() -> None:
    args = _args()
    evaluation_git_sha = _git_sha()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    protocol = _load_protocol(args.protocol)
    config = load_gate_collapse_recovery_config(args.config)
    if tuple(config.architecture.model_order) != protocol.model_order or world != 2:
        raise RuntimeError("mechanistic extraction requires Gemma/SmolLM2 on two ranks")
    if config.arm_id not in protocol.arms or any(
        forbidden in model.model_id.lower()
        for forbidden in protocol.forbidden_model_substrings
        for model in config.models
    ):
        raise RuntimeError("model roster or experiment arm violates the frozen protocol")
    selected_split = "validation" if args.mode == "validation-smoke" else "test"
    pools = (
        1
        if args.mode == "validation-smoke"
        else protocol.accepted_evaluation_anchors // protocol.pool_anchors
    )
    expected_anchors = pools * protocol.pool_anchors

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # A document-disjoint confirmation extraction must replay the frozen stream past
    # every excluded development document before its first pool. Rank 1 legitimately
    # waits during that rank-0-only scan, so the default ten-minute watchdog is too short.
    dist.init_process_group(
        "nccl", device_id=device, timeout=timedelta(minutes=90)
    )
    started = time.perf_counter()
    validation_args = argparse.Namespace(
        config=args.config,
        contract=args.protocol,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
    )
    provenance = _validate_inputs(validation_args, config)
    exposures = _exposure_report(args.run_dir, protocol)
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
    native_biases = {
        target.key: logical_weight_bias(loaded.forward_model, target)[1]
        for target in registry[local_key].targets()
    }
    model = ModelOwnedMain2B(
        config,
        registry,
        local_key,
        native_biases=native_biases,
        device=device,
        dtype=torch.float32,
        sparse_backend="dense",
    )
    manager = ModelOwnedCheckpoint(args.checkpoint.parent, keep_complete=2)
    runtime_state, checkpoint_manifest = manager.load_model_only(
        path=args.checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=sha256_file(args.config),
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

    excluded_document_ids: frozenset[str] = frozenset()
    excluded_manifest_sha256 = None
    if isinstance(protocol, (FinalIntruderValidationContract, HumanInterpretabilityContract)):
        excluded_path = Path(protocol.excluded_document_manifest)
        excluded_manifest_sha256 = sha256_file(excluded_path)
        if excluded_manifest_sha256 != protocol.excluded_document_manifest_sha256:
            raise ValueError("excluded document manifest hash changed")
        excluded_rows = json.loads(excluded_path.read_text(encoding="utf-8"))
        excluded_document_ids = frozenset(str(row["document_id"]) for row in excluded_rows)
        if len(excluded_document_ids) != protocol.excluded_document_count:
            raise ValueError("excluded document manifest count changed")

    stream = None
    if rank == 0:
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
            excluded_document_ids=excluded_document_ids,
            sequence_length=config.data.sequence_length,
            model_order=model_order,
        )

    def next_pool() -> AlignedAnchorPool:
        payload = (
            None
            if rank
            else stream.next_pool(pool_anchors=protocol.pool_anchors).to_payload()
        )
        return AlignedAnchorPool.from_payload(_broadcast(payload))

    n_features = config.architecture.dictionary_size
    row_counts: dict[str, list[np.ndarray]] = {site: [] for site in ("pre", "mid")}
    indices: dict[str, list[np.ndarray]] = {site: [] for site in ("pre", "mid")}
    coefficients: dict[str, list[np.ndarray]] = {site: [] for site in ("pre", "mid")}
    retained = {
        site: [dict() for _ in range(n_features)] for site in ("pre", "mid")
    }
    anchor_span_cache: dict[tuple[str, int, int], tuple[int, int, str]] = {}
    document_ids: list[str] = []
    document_to_index: dict[str, int] = {}
    document_rows: list[dict[str, Any]] = []
    anchor_document_indices: list[int] = []
    anchor_ids: list[str] = []
    accepted = 0
    for _pool_index in range(pools):
        pool = next_pool()
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
            for observation in pool.source_observations:
                document_id = str(observation["document_id"])
                if document_id not in document_to_index:
                    document_to_index[document_id] = len(document_ids)
                    document_ids.append(document_id)
                    document_rows.append(
                        {
                            "document_id": document_id,
                            "source_group": observation.get("source_group", "unknown"),
                        }
                    )
            for local_document in pool.models["gemma"].anchor_document_indices:
                anchor_document_indices.append(
                    document_to_index[pool.document_ids[int(local_document)]]
                )
            anchor_ids.extend(pool.anchor_ids)
            for site in ("pre", "mid"):
                code = sparse[site].values[capture.valid_anchors].detach().float()
                position, feature = (code > 0).nonzero(as_tuple=True)
                value = code[position, feature]
                position_np = position.cpu().numpy().astype(np.int32, copy=False)
                feature_np = feature.cpu().numpy().astype(np.uint16, copy=False)
                value_np = value.cpu().numpy().astype(np.float32, copy=False)
                row_counts[site].append(
                    (code > 0).sum(dim=1).cpu().numpy().astype(np.int16, copy=False)
                )
                indices[site].append(feature_np)
                coefficients[site].append(value_np)
                _update_contexts(
                    retained[site],
                    features=feature_np,
                    positions=position_np,
                    values=value_np,
                    pool=pool,
                    pool_anchor_offset=accepted,
                    tokenizer=tokenizer,
                    model_key="gemma",
                    tokens_per_side=protocol.context_tokens_per_side,
                    canonical_characters_per_side=(
                        protocol.canonical_context_characters_per_side
                        if isinstance(
                            protocol,
                            (FinalIntruderValidationContract, HumanInterpretabilityContract),
                        )
                        else None
                    ),
                    anchor_span_cache=anchor_span_cache,
                    limit=protocol.retained_contexts_per_feature,
                )
        accepted += pool.anchor_count
        del capture, sparse

    if accepted != expected_anchors:
        raise ValueError("mechanistic extraction accepted-anchor budget changed")
    if excluded_document_ids.intersection(document_ids):
        raise ValueError("excluded source documents leaked into final extraction")
    if _parameter_signature(loaded.forward_model) != source_before:
        raise RuntimeError("frozen source model changed during mechanistic extraction")
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
        np.savez_compressed(
            args.output_dir / "anchors.npz",
            anchor_ids=np.asarray(anchor_ids, dtype="U64"),
            document_indices=np.asarray(anchor_document_indices, dtype=np.int32),
        )
        _write_json(args.output_dir / "documents.json", document_rows)
        site_reports = [
            _save_site(
                args.output_dir,
                site,
                row_counts=row_counts[site],
                indices=indices[site],
                values=coefficients[site],
                retained=retained[site],
                accepted_anchors=accepted,
            )
            for site in ("pre", "mid")
        ]
        with (args.output_dir / "feature_eligibility.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("site", "feature", "retained_unique_documents", "eligible"),
            )
            writer.writeheader()
            for site in ("pre", "mid"):
                for feature, by_document in enumerate(retained[site]):
                    writer.writerow(
                        {
                            "site": site,
                            "feature": feature,
                            "retained_unique_documents": len(by_document),
                            "eligible": int(
                                len(by_document) >= protocol.minimum_unique_documents
                            ),
                        }
                    )
        manifest = {
            "schema_version": 1,
            "status": (
                "MECHANISTIC_VALIDATION_SMOKE_COMPLETE"
                if args.mode == "validation-smoke"
                else "MECHANISTIC_TEST_EXTRACTION_COMPLETE"
            ),
            "experiment_family": protocol.experiment_family,
            "arm": config.arm_id,
            "mode": args.mode,
            "selected_split": selected_split,
            "prior_validation_documents_are_development_data": True,
            "excluded_document_manifest_sha256": excluded_manifest_sha256,
            "excluded_document_count": len(excluded_document_ids),
            "excluded_documents_disjoint": not bool(
                excluded_document_ids.intersection(document_ids)
            ),
            "accepted_evaluation_anchors": accepted,
            "unique_source_documents": len(document_ids),
            "natural_gate_only": True,
            "forced_activations": 0,
            "checkpoint_step": int(checkpoint_manifest["step"]),
            "runtime_state_step": int(runtime_state["completed_steps"]),
            "training_exposures": exposures,
            "model_order": list(model_order),
            "source_model_provenance": {
                key: {
                    "model_id": registry[key].model_id,
                    "revision": registry[key].revision,
                    "tokenizer_revision": registry[key].tokenizer_revision,
                }
                for key in model_order
            },
            "training_git_sha": provenance["training_git_sha"],
            "evaluation_git_sha": evaluation_git_sha,
            "config_sha256": provenance["config_sha256"],
            "config_fingerprint": config.fingerprint(),
            "protocol_sha256": sha256_file(args.protocol),
            "protocol_fingerprint": protocol.fingerprint(),
            "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "alignment_manifest_sha256": provenance["alignment_manifest_sha256"],
            "anchors_sha256": sha256_file(args.output_dir / "anchors.npz"),
            "documents_sha256": sha256_file(args.output_dir / "documents.json"),
            "site_reports": site_reports,
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
                "manifest_sha256": sha256_file(args.output_dir / "experiment_manifest.json"),
                "status": "COMPLETE",
            },
        )
    if stream is not None:
        stream.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

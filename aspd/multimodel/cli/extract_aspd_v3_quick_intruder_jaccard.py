"""Extract held-out natural firings for the completed v3.1 quick evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aspd.multimodel.cli.analyze_aspd_v3_mechanistic import (
    _document_counts,
    _feature_anchor_lists,
    _stratified_features,
)
from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _finite_state_report,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.cli.extract_aspd_v3_mechanistic import _save_site
from aspd.multimodel.cli.extract_aspd_v3_paper_intruder import (
    _make_stream,
    _retain_contexts,
    _validate_static_inputs,
)
from aspd.multimodel.cli.extract_aspd_v3_paper_intruder_colocated import (
    _encode_colocated,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.hooks import assert_frozen_backbone
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.paper_intruder import QuickIntruderJaccardContract
from aspd.multimodel.v3.registry import build_model_registry_2b


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
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


def _sample_features(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    n_features: int,
    document_indices: np.ndarray,
    contract: QuickIntruderJaccardContract,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    firing, counts = _feature_anchor_lists(indptr, indices, n_features)
    document_counts = _document_counts(firing, document_indices)
    sampled, _strata, report = _stratified_features(
        counts,
        document_counts,
        minimum_fires=contract.minimum_feature_fires,
        minimum_documents=contract.minimum_unique_documents,
        sample_size=sample_size,
        seed=contract.feature_sampling_seed,
    )
    return sampled, report


def main() -> None:
    args = _args()
    started = time.perf_counter()
    protocol = QuickIntruderJaccardContract.load(args.protocol)
    config = load_main500m_config(args.config)
    provenance = _validate_static_inputs(args, config, protocol)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite extraction output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    registry = build_model_registry_2b(config)
    model_order = tuple(config.architecture.model_order)
    tokenizers: dict[str, Any] = {}
    loaded_sources: dict[str, Any] = {}
    source_before: dict[str, Any] = {}
    owner_models: dict[str, ModelOwnedMain2B] = {}
    rank_reports: list[dict[str, Any]] = []
    runtime_states: list[dict[str, Any]] = []
    checkpoint_manifest: dict[str, Any] | None = None
    manager = ModelOwnedCheckpoint(args.checkpoint.parent, keep_complete=2)
    for checkpoint_rank, key in enumerate(model_order):
        tokenizers[key] = load_v3_tokenizer(
            registry[key], local_files_only=args.local_files_only
        )
        loaded_sources[key] = load_frozen_v3_model(
            registry[key], device, local_files_only=args.local_files_only
        )
        assert_frozen_backbone(loaded_sources[key].forward_model)
        source_before[key] = _parameter_signature(loaded_sources[key].forward_model)
        owner = ModelOwnedMain2B(
            config,
            registry,
            key,
            native_biases={target.key: None for target in registry[key].targets()},
            device=device,
            dtype=torch.float32,
            sparse_backend="dense",
        )
        runtime_state, observed_manifest = manager.load_model_only_owner(
            path=args.checkpoint,
            model=owner,
            config_fingerprint=config.fingerprint(),
            config_file_sha256=protocol.source_config_sha256,
            dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
            checkpoint_rank=checkpoint_rank,
            expected_checkpoint_world_size=len(model_order),
            expected_model_key=key,
        )
        owner.eval().requires_grad_(False)
        owner_models[key] = owner
        runtime_states.append(runtime_state)
        checkpoint_manifest = observed_manifest
        rank_reports.append(
            {
                "checkpoint_rank": checkpoint_rank,
                "model": key,
                "finite_state": _finite_state_report(owner),
                "shared_encoder_sha256": _shared_digest(owner),
                "source_weights_unchanged": "PENDING_FINAL_CHECK",
            }
        )
    if len({row["shared_encoder_sha256"] for row in rank_reports}) != 1:
        raise ValueError("replicated shared encoders differ across checkpoint owners")
    exposure_counts = {int(row["accepted_exposures"]) for row in runtime_states}
    if exposure_counts != {protocol.accepted_training_exposures}:
        raise ValueError(f"checkpoint exposure mismatch: {sorted(exposure_counts)}")

    n_features = config.architecture.dictionary_size
    pools = 1 if args.mode == "smoke" else protocol.accepted_evaluation_anchors // protocol.pool_anchors
    accepted_anchors = pools * protocol.pool_anchors
    first_retained = {
        site: [dict() for _ in range(n_features)] for site in ("pre", "mid")
    }
    row_counts: dict[str, list[np.ndarray]] = {"pre": [], "mid": []}
    sparse_indices: dict[str, list[np.ndarray]] = {"pre": [], "mid": []}
    sparse_values: dict[str, list[np.ndarray]] = {"pre": [], "mid": []}
    documents: list[dict[str, Any]] = []
    document_to_index: dict[str, int] = {}
    anchor_document_indices: list[int] = []
    anchor_ids: list[str] = []

    def run_pass(
        pass_index: int,
        retained: dict[str, list[dict[str, dict[str, Any]]]],
        *,
        limit: int,
        allowed: dict[str, set[int]] | None,
        collect_sparse: bool,
    ) -> None:
        stream = _make_stream(
            config=config,
            registry=registry,
            local_key=protocol.display_model_key,
            tokenizer=tokenizers[protocol.display_model_key],
            model_order=model_order,
            output_dir=args.output_dir,
            pass_index=pass_index,
            local_files_only=args.local_files_only,
        )
        accepted = 0
        for _pool_index in range(pools):
            pool = stream.next_pool(pool_anchors=protocol.pool_anchors)
            sparse = _encode_colocated(
                pool,
                model_order=model_order,
                tokenizers=tokenizers,
                loaded_sources=loaded_sources,
                owner_models=owner_models,
                registry=registry,
                device=device,
            )
            if collect_sparse:
                for observation in pool.source_observations:
                    document_id = str(observation["document_id"])
                    if document_id not in document_to_index:
                        document_to_index[document_id] = len(documents)
                        documents.append(
                            {
                                "document_id": document_id,
                                "source_group": observation.get("source_group", "unknown"),
                            }
                        )
                display_inputs = pool.models[protocol.display_model_key]
                for local_document in display_inputs.anchor_document_indices:
                    anchor_document_indices.append(
                        document_to_index[pool.document_ids[int(local_document)]]
                    )
                anchor_ids.extend(str(value) for value in pool.anchor_ids)
            for site in ("pre", "mid"):
                code = sparse[site]
                if collect_sparse:
                    positions, features = (code > 0).nonzero(as_tuple=True)
                    values = code[positions, features]
                    row_counts[site].append(
                        torch.bincount(positions, minlength=pool.anchor_count)
                        .cpu()
                        .numpy()
                    )
                    sparse_indices[site].append(features.cpu().numpy())
                    sparse_values[site].append(values.cpu().numpy())
                _retain_contexts(
                    retained[site],
                    code=code,
                    pool=pool,
                    model_key=protocol.display_model_key,
                    tokens_per_side=protocol.context_tokens_per_side,
                    limit=limit,
                    allowed_features=None if allowed is None else allowed[site],
                    pool_anchor_offset=accepted,
                    tokenizer=tokenizers[protocol.display_model_key],
                )
            accepted += pool.anchor_count
            del sparse
        stream.close()
        if accepted != accepted_anchors:
            raise ValueError("evaluation accepted-anchor budget changed")

    run_pass(
        1,
        first_retained,
        limit=protocol.contexts_per_feature,
        allowed=None,
        collect_sparse=True,
    )
    document_indices_array = np.asarray(anchor_document_indices, dtype=np.int32)
    site_sampling: dict[str, dict[str, Any]] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    allowed: dict[str, set[int]] = {"pre": set(), "mid": set()}
    sample_size = protocol.features_per_site if args.mode == "full" else 5
    for site in ("pre", "mid"):
        counts = np.concatenate(row_counts[site]).astype(np.int16, copy=False)
        indptr = np.empty(accepted_anchors + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, dtype=np.int64, out=indptr[1:])
        indices = np.concatenate(sparse_indices[site]).astype(np.uint16, copy=False)
        selected[site], site_sampling[site] = _sample_features(
            indptr=indptr,
            indices=indices,
            n_features=n_features,
            document_indices=document_indices_array,
            contract=protocol,
            sample_size=sample_size,
        )
        allowed[site] = {int(row["feature"]) for row in selected[site]}

    target_retained = {
        site: [dict() for _ in range(n_features)] for site in ("pre", "mid")
    }
    run_pass(
        2,
        target_retained,
        limit=protocol.retained_contexts_per_feature,
        allowed=allowed,
        collect_sparse=False,
    )
    for site in ("pre", "mid"):
        for feature in allowed[site]:
            first_retained[site][feature] = target_retained[site][feature]
    for site in ("pre", "mid"):
        for feature in allowed[site]:
            first_retained[site][feature] = target_retained[site][feature]

    _write_json(args.output_dir / "documents.json", documents)
    np.savez_compressed(
        args.output_dir / "anchors.npz",
        anchor_ids=np.asarray(anchor_ids),
        document_indices=document_indices_array,
    )
    site_reports = {
        site: _save_site(
            args.output_dir,
            site,
            row_counts=row_counts[site],
            indices=sparse_indices[site],
            values=sparse_values[site],
            retained=first_retained[site],
            accepted_anchors=accepted_anchors,
        )
        for site in ("pre", "mid")
    }
    _write_json(
        args.output_dir / "sampled_features.json",
        {
            "mode": args.mode,
            "feature_sampling_seed": protocol.feature_sampling_seed,
            "sites": selected,
            "sampling": site_sampling,
        },
    )

    for report in rank_reports:
        key = str(report["model"])
        if _parameter_signature(loaded_sources[key].forward_model) != source_before[key]:
            raise RuntimeError(f"frozen source model changed during extraction: {key}")
        report["source_weights_unchanged"] = "PASS"
    assert checkpoint_manifest is not None
    manifest = {
        "schema_version": 1,
        "status": (
            "ASPD_V3_1_QUICK_EVALUATION_EXTRACTION_COMPLETE"
            if args.mode == "full"
            else "ASPD_V3_1_QUICK_EVALUATION_SMOKE_COMPLETE"
        ),
        "mode": args.mode,
        "experiment_family": protocol.experiment_family,
        "execution_mode": "single_b200_colocated_three_checkpoint_owners",
        "selected_split": protocol.selected_split,
        "split_unit": protocol.split_unit,
        "accepted_evaluation_anchors": accepted_anchors,
        "natural_gate_only": True,
        "gate": "per-anchor feature-axis z-score -> ReLU -> pool-wide BatchTopK",
        "forced_activations": 0,
        "checkpoint_step": int(checkpoint_manifest["step"]),
        "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
        "accepted_training_exposures": protocol.accepted_training_exposures,
        "model_order": list(model_order),
        "dictionary_size_per_site": n_features,
        "sampled_features_per_site": sample_size,
        "source_model_provenance": {
            key: {
                "model_id": registry[key].model_id,
                "revision": registry[key].revision,
                "tokenizer_revision": registry[key].tokenizer_revision,
            }
            for key in model_order
        },
        "evaluation_git_sha": _git_sha(),
        "config_sha256": protocol.source_config_sha256,
        "config_fingerprint": config.fingerprint(),
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_fingerprint": protocol.fingerprint(),
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "alignment_manifest_sha256": provenance["alignment_manifest_sha256"],
        "anchors_sha256": sha256_file(args.output_dir / "anchors.npz"),
        "documents_sha256": sha256_file(args.output_dir / "documents.json"),
        "sampled_features_sha256": sha256_file(args.output_dir / "sampled_features.json"),
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
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(args.output_dir / "experiment_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()

"""Single-B200, three-owner natural-firing extraction for paper Intruder."""

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

from aspd.eval.autointerp_db import sample_keys
from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _capture,
    _finite_state_report,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.cli.extract_aspd_v3_paper_intruder import (
    _make_stream,
    _retain_contexts,
    _validate_static_inputs,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.hooks import assert_frozen_backbone
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract, component_key
from aspd.multimodel.v3.production_stream import AlignedAnchorPool
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


def _encode_colocated(
    pool: AlignedAnchorPool,
    *,
    model_order: tuple[str, ...],
    tokenizers: dict[str, Any],
    loaded_sources: dict[str, Any],
    owner_models: dict[str, ModelOwnedMain2B],
    registry: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Reproduce the three-rank feature exchange on one device, without approximation."""

    valid = torch.ones(1, pool.anchor_count, dtype=torch.bool, device=device)
    tokens: dict[str, dict[str, torch.Tensor]] = {"pre": {}, "mid": {}}
    for key in model_order:
        capture = _capture(
            pool,
            local_key=key,
            tokenizer=tokenizers[key],
            loaded=loaded_sources[key],
            registry=registry,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for site in ("pre", "mid"):
                tokens[site][key] = owner_models[key].local_encoders[site](
                    capture.activations[site]
                )
        del capture
    shared_owner = owner_models[model_order[0]]
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for site in ("pre", "mid"):
            scores, _logits = shared_owner.shared_encoders[site].forward_with_logits(
                tokens[site], valid
            )
            sparse = shared_owner.gates[site](scores, valid, update_statistics=False)
            result[site] = sparse.values[valid].detach().float()
    return result


def main() -> None:
    args = _args()
    started = time.perf_counter()
    evaluation_git_sha = _git_sha()
    protocol = PaperIntruderContract.load(args.protocol)
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
    checkpoint_manifest = None
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
    shared_digests = {row["shared_encoder_sha256"] for row in rank_reports}
    if len(shared_digests) != 1:
        raise ValueError("replicated shared encoders differ across checkpoint owners")
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
            for site in ("pre", "mid"):
                observed = _retain_contexts(
                    retained[site],
                    code=sparse[site],
                    pool=pool,
                    model_key=protocol.display_model_key,
                    tokens_per_side=protocol.context_tokens_per_side,
                    limit=limit,
                    allowed_features=None if allowed is None else allowed[site],
                )
                if collect_counts:
                    counts[site] += observed
            accepted += pool.anchor_count
            del sparse
        stream.close()
        if accepted != protocol.accepted_evaluation_anchors:
            raise ValueError("evaluation accepted-anchor budget changed")

    run_pass(
        1,
        first_retained,
        limit=protocol.minimum_examples_per_component,
        allowed=None,
        collect_counts=True,
    )
    eligible_keys = sorted(
        component_key(site, feature)
        for site in ("pre", "mid")
        for feature in range(n_features)
        if len(first_retained[site][feature]) >= protocol.minimum_examples_per_component
    )
    if len(eligible_keys) < protocol.n_subsample:
        raise RuntimeError(f"only {len(eligible_keys)} components are paper-Intruder eligible")
    sampled_keys = sample_keys(eligible_keys, protocol.n_subsample, protocol.seed)
    allowed: dict[str, set[int]] = {"pre": set(), "mid": set()}
    for key in sampled_keys:
        site, feature = key.split("/", 1)
        allowed[site].add(int(feature))
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

    for report in rank_reports:
        key = str(report["model"])
        if _parameter_signature(loaded_sources[key].forward_model) != source_before[key]:
            raise RuntimeError(f"frozen source model changed during extraction: {key}")
        report["source_weights_unchanged"] = "PASS"
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
                                counts[site][feature] / protocol.accepted_evaluation_anchors
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
    assert checkpoint_manifest is not None
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_PAPER_INTRUDER_EXTRACTION_COMPLETE",
        "experiment_family": protocol.experiment_family,
        "execution_mode": "single_b200_colocated_three_checkpoint_owners",
        "mathematical_parity": (
            "same local packers, shared hierarchical encoder, z-score, ReLU and BatchTopK; "
            "differentiable all-gather replaced by an ordered in-process mapping for read-only inference"
        ),
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


if __name__ == "__main__":
    main()

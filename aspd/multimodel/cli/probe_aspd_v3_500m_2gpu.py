"""Two-GPU colocation runner for the bounded ASPD v3 hypothesis probe."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _broadcast,
    _capture,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.cli.probe_aspd_v3_500m import (
    _full_activation_fvu,
    cosine_matrix,
    support_overlap,
)
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import (
    DiagnosticContract,
    sha256_file,
    validate_frozen_provenance,
)
from aspd.multimodel.v3.diagnostic_stream import ValidationAlignedAnchorStream
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B, load_main_2b_v1_config
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b
from aspd.multimodel.v3.sparse_support import prepare_sparse_support

OWNERS_BY_RANK = {0: ("qwen", "gemma"), 1: ("smol",)}
SOURCE_RANK = {key: rank for rank, keys in OWNERS_BY_RANK.items() for key in keys}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-smoke-report", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _main() -> None:
    args = _args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2 or rank not in OWNERS_BY_RANK:
        raise RuntimeError("two-GPU probe requires exactly two ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()

    contract = DiagnosticContract.load(args.contract)
    validate_frozen_provenance(contract, source_root=args.source_root)
    smoke = json.loads(args.checkpoint_smoke_report.read_text(encoding="utf-8"))
    if smoke.get("status") != "E0_CHECKPOINT_RESTORE_PASS":
        raise ValueError("two-GPU probe requires a passing E0 report")
    if smoke.get("checkpoint_manifest_sha256") != contract.checkpoint_manifest_sha256:
        raise ValueError("E0 report checkpoint differs from probe checkpoint")

    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite probe output: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    config_path = args.source_root / contract.source_config
    config = load_main_2b_v1_config(config_path)
    registry = build_model_registry_2b(config)
    checkpoint = args.source_root / contract.checkpoint
    manager = ModelOwnedCheckpoint(checkpoint.parent, keep_complete=2)
    tokenizers: dict[str, Any] = {}
    loaded_models: dict[str, Any] = {}
    trainable: dict[str, ModelOwnedMain2B] = {}
    source_signatures: dict[str, dict[str, float | int]] = {}
    runtime_steps = set()
    manifests = []

    for key in OWNERS_BY_RANK[rank]:
        tokenizer = load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
        loaded = load_frozen_v3_model(
            registry[key], device, local_files_only=args.local_files_only
        )
        assert_frozen_backbone(loaded.forward_model)
        tokenizers[key] = tokenizer
        loaded_models[key] = loaded
        source_signatures[key] = _parameter_signature(loaded.forward_model)
        native_biases = {
            target.key: logical_weight_bias(loaded.forward_model, target)[1]
            for target in registry[key].targets()
        }
        model = ModelOwnedMain2B(
            config,
            registry,
            key,
            native_biases=native_biases,
            device=device,
            dtype=torch.float32,
            sparse_backend="dense",
        )
        runtime_state, manifest = manager.load_model_only_owner(
            path=checkpoint,
            model=model,
            config_fingerprint=config.fingerprint(),
            config_file_sha256=sha256_file(config_path),
            dataset_manifest_sha256=contract.dataset_manifest_sha256,
            checkpoint_rank=MODEL_ORDER_2B.index(key),
            expected_checkpoint_world_size=3,
            expected_model_key=key,
        )
        runtime_steps.add(int(runtime_state["completed_steps"]))
        manifests.append(manifest)
        model.eval().requires_grad_(True)
        trainable[key] = model

    local_digests = {key: _shared_digest(model) for key, model in trainable.items()}
    gathered_digest_maps: list[dict[str, str] | None] = [None] * world
    dist.all_gather_object(gathered_digest_maps, local_digests)
    all_digests = {
        key: digest for value in gathered_digest_maps if value is not None for key, digest in value.items()
    }
    if set(all_digests) != set(MODEL_ORDER_2B) or len(set(all_digests.values())) != 1:
        raise ValueError("three historical shared-encoder replicas are not identical")
    if runtime_steps != {61036} or any(int(value["step"]) != 61036 for value in manifests):
        raise ValueError("two-GPU probe restored the wrong checkpoint step")

    stream = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizers.get(key)
            or load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
            for key in MODEL_ORDER_2B
        }
        stream = ValidationAlignedAnchorStream(
            dataset=load_pinned_pile_dataset(
                config.data.dataset_id, config.data.dataset_revision
            ),
            dataset_id=config.data.dataset_id,
            dataset_revision=config.data.dataset_revision,
            tokenizers=alignment_tokenizers,
            tokenizer_revisions={
                key: registry[key].tokenizer_revision for key in MODEL_ORDER_2B
            },
            split_policy=HashSplitPolicy(
                modulus=config.data.split_modulus,
                validation_buckets=frozenset(config.data.validation_buckets),
                test_buckets=frozenset(config.data.test_buckets),
                seed=config.data.split_seed,
            ),
            dedup_path=args.output_dir / "validation_stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
        )
    payload = None if rank else stream.next_pool(pool_anchors=contract.pool_anchors).to_payload()
    pool = AlignedAnchorPool.from_payload(_broadcast(payload))
    captures = {
        key: _capture(
            pool,
            local_key=key,
            tokenizer=tokenizers[key],
            loaded=loaded_models[key],
            registry=registry,
            device=device,
        )
        for key in OWNERS_BY_RANK[rank]
    }

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        tokens_by_site: dict[str, dict[str, torch.Tensor]] = {"pre": {}, "mid": {}}
        token_shape = (
            1,
            pool.anchor_count,
            config.architecture.latent_tokens_per_model,
            config.architecture.packed_token_width,
        )
        for site in ("pre", "mid"):
            for key in MODEL_ORDER_2B:
                if key in trainable:
                    tokens = trainable[key].local_encoders[site](
                        captures[key].activations[site]
                    ).detach()
                else:
                    tokens = torch.empty(token_shape, device=device, dtype=torch.bfloat16)
                dist.broadcast(tokens, src=SOURCE_RANK[key])
                tokens_by_site[site][key] = tokens

        local_sparse: dict[str, dict[str, Any]] = {}
        local_scores: dict[str, dict[str, torch.Tensor]] = {}
        local_support: dict[str, dict[str, Any]] = {}
        local_reports: dict[str, dict[str, Any]] = {}
        gradient_vectors: dict[str, dict[str, torch.Tensor]] = {}
        for key, model in trainable.items():
            capture = captures[key]
            full_scores = {}
            full_sparse = {}
            full_support = {}
            for site in ("pre", "mid"):
                scores = model.shared_encoders[site](
                    tokens_by_site[site], capture.valid_anchors
                )
                sparse = model.gates[site](
                    scores, capture.valid_anchors, update_statistics=False
                )
                full_scores[site] = scores
                full_sparse[site] = sparse
                full_support[site] = prepare_sparse_support(
                    sparse.values, capture.valid_anchors, binary=False
                )
            local_scores[key] = full_scores
            local_sparse[key] = full_sparse
            local_support[key] = full_support
            full_fvu = _full_activation_fvu(
                model,
                capture,
                {site: full_sparse[site].values for site in ("pre", "mid")},
            )
            ablations = {}
            for ablated_key in MODEL_ORDER_2B:
                ablated_codes = {}
                gate_report = {}
                for site in ("pre", "mid"):
                    tokens = dict(tokens_by_site[site])
                    tokens[ablated_key] = torch.zeros_like(tokens[ablated_key])
                    scores = model.shared_encoders[site](tokens, capture.valid_anchors)
                    sparse = model.gates[site](
                        scores, capture.valid_anchors, update_statistics=False
                    )
                    ablated_codes[site] = sparse.values
                    gate_report[site] = {
                        "score_relative_l2_change": float(
                            (scores.float() - full_scores[site].float()).norm()
                            / full_scores[site].float().norm()
                        ),
                        "support": support_overlap(
                            full_sparse[site].values, sparse.values
                        ),
                    }
                ablations[ablated_key] = {
                    "gate": gate_report,
                    "local_model_full_activation_fvu": _full_activation_fvu(
                        model, capture, ablated_codes
                    ),
                }
            terms = list(
                model.activation_loss_terms(
                    capture,
                    full_scores,
                    full_sparse,
                    full_support,
                    auxk_may_be_active=False,
                )
            )
            gradient_vectors[key] = {}
            gradient_summary = {}
            for site_index, site in enumerate(("pre", "mid")):
                site_loss = sum(
                    coefficient * activation
                    for name, activation, _auxk, coefficient in terms
                    if name.startswith(f"{site}/")
                )
                parameters = tuple(model.shared_encoders[site].parameters())
                gradients = torch.autograd.grad(
                    site_loss,
                    parameters,
                    retain_graph=site_index == 0,
                    allow_unused=False,
                )
                flat = torch.cat(
                    [value.detach().float().reshape(-1) for value in gradients]
                )
                gradient_vectors[key][site] = flat
                gradient_summary[site] = {
                    "local_weighted_activation_loss": float(site_loss.detach()),
                    "local_shared_gradient_l2": float(flat.norm()),
                }
            local_reports[key] = {
                "model": key,
                "full_activation_fvu": full_fvu,
                "ablations": ablations,
                "gradients": gradient_summary,
            }

    gate_parity: dict[str, dict[str, Any]] = {}
    for site in ("pre", "mid"):
        reference = (
            local_sparse["qwen"][site].values
            if rank == 0
            else torch.empty(
                (1, pool.anchor_count, config.architecture.dictionary_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        dist.broadcast(reference, src=0)
        for key in MODEL_ORDER_2B:
            source = SOURCE_RANK[key]
            value = (
                local_sparse[key][site].values
                if key in local_sparse
                else torch.empty_like(reference)
            )
            dist.broadcast(value, src=source)
            maximum = float((value.float() - reference.float()).abs().max())
            support_equal = bool(torch.equal(value > 0, reference > 0))
            gate_parity.setdefault(key, {})[site] = {
                "max_abs_value_difference_from_qwen_replica": maximum,
                "support_exactly_equal": support_equal,
            }
            if maximum != 0.0 or not support_equal:
                raise ValueError("colocated shared-gate replicas are not exactly equal")

    gradient_cosines = {}
    for site in ("pre", "mid"):
        template = next(iter(gradient_vectors.values()))[site]
        vectors = []
        for key in MODEL_ORDER_2B:
            value = (
                gradient_vectors[key][site]
                if key in gradient_vectors
                else torch.empty_like(template)
            )
            dist.broadcast(value, src=SOURCE_RANK[key])
            vectors.append(value)
        gradient_cosines[site] = {
            "model_order": list(MODEL_ORDER_2B),
            "pairwise_cosine": cosine_matrix(vectors),
        }

    for key, loaded in loaded_models.items():
        if _parameter_signature(loaded.forward_model) != source_signatures[key]:
            raise RuntimeError(f"frozen source model changed during probe: {key}")
    gathered_reports: list[dict[str, dict[str, Any]] | None] = [None] * world
    dist.all_gather_object(gathered_reports, local_reports)
    if rank == 0:
        reports = {
            key: report
            for value in gathered_reports
            if value is not None
            for key, report in value.items()
        }
        if set(reports) != set(MODEL_ORDER_2B):
            raise ValueError("two-GPU probe did not produce all three model reports")
        output = {
            "schema_version": 1,
            "status": "HYPOTHESIS_PROBE_COMPLETE",
            "execution_layout": {
                "gpu_0": ["qwen", "gemma"],
                "gpu_1": ["smol"],
            },
            "checkpoint_step": 61036,
            "accepted_anchors": pool.anchor_count,
            "documents": len(set(pool.document_ids)),
            "model_order": list(MODEL_ORDER_2B),
            "shared_checkpoint_digest": next(iter(all_digests.values())),
            "gate_replica_parity": gate_parity,
            "gradient_cosines": gradient_cosines,
            "source_weights_unchanged": "PASS",
            "optimizer_updates": 0,
            "reports": reports,
            "elapsed_seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        (args.output_dir / "hypothesis_probe.json").write_text(
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("ASPD_V3_HYPOTHESIS_PROBE_COMPLETE")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()

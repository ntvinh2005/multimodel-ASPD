"""Bounded no-update probe for ASPD v3 shared-gate interference and gradient balance."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _broadcast,
    _capture,
    _parameter_signature,
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
from aspd.multimodel.v3.objective import matryoshka_boundaries
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b
from aspd.multimodel.v3.sparse_support import prepare_sparse_support


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-smoke-report", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def support_overlap(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    left = reference.detach() > 0
    right = candidate.detach() > 0
    if left.shape != right.shape:
        raise ValueError("gate support shapes differ")
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    reference_count = int(left.sum())
    return {
        "reference_active": reference_count,
        "candidate_active": int(right.sum()),
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
        "reference_retention": intersection / reference_count if reference_count else 1.0,
    }


def cosine_matrix(vectors: list[torch.Tensor]) -> list[list[float]]:
    if not vectors or any(value.ndim != 1 for value in vectors):
        raise ValueError("gradient cosine requires non-empty flat vectors")
    if len({value.numel() for value in vectors}) != 1:
        raise ValueError("gradient vectors differ in length")
    result = []
    for left in vectors:
        row = []
        for right in vectors:
            denominator = float(left.float().norm() * right.float().norm())
            if denominator <= 0:
                raise FloatingPointError("zero shared-encoder gradient norm")
            row.append(float(torch.dot(left.float(), right.float()) / denominator))
        result.append(row)
    return result


def _full_activation_fvu(
    model: ModelOwnedMain2B,
    capture: Any,
    codes: dict[str, torch.Tensor],
) -> dict[str, float]:
    boundaries = matryoshka_boundaries(8192, (0.0625, 0.0625, 0.125, 0.25, 0.5))
    result = {}
    for site in ("pre", "mid"):
        sse = 0.0
        denominator = 0.0
        for decoder, target in zip(
            model.activation_banks[site].decoders[model.local_model_key],
            capture.activations[site],
            strict=True,
        ):
            predictions = decoder.forward_dense_grouped(
                codes[site], boundaries=boundaries, include_bias=True
            )
            assert isinstance(predictions, tuple)
            prediction = predictions[-1].float()[capture.valid_anchors]
            truth = target.float()[capture.valid_anchors]
            sse += float((prediction - truth).square().sum())
            denominator += float((truth - truth.mean(dim=0, keepdim=True)).square().sum())
        if denominator <= 0 or not math.isfinite(sse / denominator):
            raise FloatingPointError("invalid activation FVU in hypothesis probe")
        result[site] = sse / denominator
    return result


def _main() -> None:
    args = _args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 3:
        raise RuntimeError("hypothesis probe requires three model-owner ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()

    contract = DiagnosticContract.load(args.contract)
    validate_frozen_provenance(contract, source_root=args.source_root)
    smoke = json.loads(args.checkpoint_smoke_report.read_text(encoding="utf-8"))
    if smoke.get("status") != "E0_CHECKPOINT_RESTORE_PASS":
        raise ValueError("hypothesis probe requires a passing E0 report")
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
    local_key = MODEL_ORDER_2B[rank]
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
    checkpoint = args.source_root / contract.checkpoint
    runtime_state, manifest = ModelOwnedCheckpoint(
        checkpoint.parent, keep_complete=2
    ).load_model_only(
        path=checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=sha256_file(config_path),
        dataset_manifest_sha256=contract.dataset_manifest_sha256,
        expected_model_key=local_key,
    )
    model.eval().requires_grad_(True)

    stream = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
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
    capture = _capture(
        pool,
        local_key=local_key,
        tokenizer=tokenizer,
        loaded=loaded,
        registry=registry,
        device=device,
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        gathered_by_site = {}
        full_scores = {}
        full_sparse = {}
        full_support = {}
        for site in ("pre", "mid"):
            local_tokens = model.local_encoders[site](capture.activations[site]).detach()
            gathered = tuple(value.detach() for value in all_gather(local_tokens))
            gathered_by_site[site] = gathered
            tokens = {key: gathered[index] for index, key in enumerate(MODEL_ORDER_2B)}
            scores = model.shared_encoders[site](tokens, capture.valid_anchors)
            sparse = model.gates[site](scores, capture.valid_anchors, update_statistics=False)
            full_scores[site] = scores
            full_sparse[site] = sparse
            full_support[site] = prepare_sparse_support(
                sparse.values, capture.valid_anchors, binary=False
            )

        full_fvu = _full_activation_fvu(
            model, capture, {site: full_sparse[site].values for site in ("pre", "mid")}
        )
        ablations = {}
        for ablated_index, ablated_key in enumerate(MODEL_ORDER_2B):
            ablated_codes = {}
            site_report = {}
            for site in ("pre", "mid"):
                gathered = gathered_by_site[site]
                tokens = {
                    key: (
                        torch.zeros_like(gathered[index])
                        if index == ablated_index
                        else gathered[index]
                    )
                    for index, key in enumerate(MODEL_ORDER_2B)
                }
                scores = model.shared_encoders[site](tokens, capture.valid_anchors)
                sparse = model.gates[site](scores, capture.valid_anchors, update_statistics=False)
                ablated_codes[site] = sparse.values
                score_delta = (scores.float() - full_scores[site].float()).norm()
                score_norm = full_scores[site].float().norm()
                site_report[site] = {
                    "score_relative_l2_change": float(score_delta / score_norm),
                    "support": support_overlap(full_sparse[site].values, sparse.values),
                }
            ablations[ablated_key] = {
                "gate": site_report,
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

    gradient_report = {}
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
        flat = torch.cat([value.detach().float().reshape(-1) for value in gradients])
        gathered_gradients = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(gathered_gradients, flat)
        gradient_report[site] = {
            "local_weighted_activation_loss": float(site_loss.detach()),
            "local_shared_gradient_l2": float(flat.norm()),
            "model_order": list(MODEL_ORDER_2B),
            "pairwise_cosine": cosine_matrix(gathered_gradients),
        }

    if _parameter_signature(loaded.forward_model) != source_before:
        raise RuntimeError("frozen source model changed during hypothesis probe")
    local_report = {
        "rank": rank,
        "model": local_key,
        "full_activation_fvu": full_fvu,
        "ablations": ablations,
        "gradients": gradient_report,
    }
    reports: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(reports, local_report)
    if rank == 0:
        output = {
            "schema_version": 1,
            "status": "HYPOTHESIS_PROBE_COMPLETE",
            "checkpoint_step": manifest["step"],
            "runtime_state_step": runtime_state["completed_steps"],
            "accepted_anchors": pool.anchor_count,
            "documents": len(set(pool.document_ids)),
            "model_order": list(MODEL_ORDER_2B),
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

"""Freeze three-model contexts for ASPD v3.1 signed causal correspondence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.alignment import align_canonical_text
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.functional_correspondence import (
    FunctionalCorrespondenceContract,
    extracted_context_boundary,
)
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.models import load_v3_tokenizer
from aspd.multimodel.v3.registry import build_model_registry_2b


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--sampled-keys", type=Path, required=True)
    parser.add_argument("--heldout-analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _jsonl_selected(path: Path, selected: set[int]) -> dict[int, list[dict[str, Any]]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            feature = int(row["feature"])
            if feature in selected:
                output[feature] = list(row["contexts"])
    if set(output) != selected:
        raise ValueError(f"context artifact lacks sampled features: {sorted(selected - set(output))}")
    return output


def _load_support(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as artifact:
        return artifact["indptr"].astype(np.int64), artifact["indices"].astype(np.int64)


def _fires(indptr: np.ndarray, indices: np.ndarray, anchor: int, feature: int) -> bool:
    row = indices[int(indptr[anchor]) : int(indptr[anchor + 1])]
    return bool(np.any(row == feature))


def _strata(
    diagnostics: list[dict[str, str]], sampled: Mapping[str, set[int]]
) -> dict[tuple[str, str], list[int]]:
    output = {}
    for site in ("pre", "mid"):
        rows = sorted(
            (
                row
                for row in diagnostics
                if row["site"] == site and int(row["feature"]) in sampled[site]
            ),
            key=lambda row: (float(row["internal_contribution_l2_total"]), int(row["feature"])),
        )
        if len(rows) != len(sampled[site]):
            raise ValueError(f"held-out diagnostics lack sampled {site} features")
        for name, part in zip(("low", "medium", "high"), np.array_split(rows, 3), strict=True):
            output[(site, name)] = [int(row["feature"]) for row in part]
    return output


def _prepare_context(
    row: dict[str, Any],
    *,
    gemma_tokenizer: Any,
    tokenizers: Mapping[str, Any],
    model_order: tuple[str, ...],
    contract: FunctionalCorrespondenceContract,
) -> dict[str, Any]:
    context = str(row["context"])
    boundary = extracted_context_boundary(
        token_ids=[int(value) for value in row["token_ids"]],
        activations=[float(value) for value in row["activations"]],
        context=context,
        tokenizer=gemma_tokenizer,
    )
    continuation = context[boundary:]
    continuation_bytes = len(continuation.encode("utf-8"))
    if continuation_bytes < contract.minimum_continuation_utf8_bytes:
        raise ValueError("continuation is shorter than the frozen byte minimum")
    identity = hashlib.sha256(
        f"{row['anchor_id']}\0{context}\0{boundary}".encode()
    ).hexdigest()
    alignment = align_canonical_text(
        context,
        document_id=identity,
        source_group=str(row.get("source_group", "unknown")),
        tokenizers=tokenizers,
        model_order=model_order,
        max_length=512,
    )
    anchors = [anchor for anchor in alignment.anchors if anchor.character_boundary == boundary]
    if len(anchors) != 1:
        raise ValueError("natural peak is not a shared exact tokenizer boundary")
    anchor = anchors[0]
    tokenizations = {}
    for model_key in model_order:
        tokenized = alignment.tokenized[model_key]
        end = tokenized.boundary_to_token.get(len(context))
        if end is None:
            raise ValueError(f"{model_key} lacks an exact full-context boundary")
        prefix_count = anchor.positions[model_key].prefix_token_count
        end_count = end.prefix_token_count
        if end_count - prefix_count < contract.minimum_continuation_tokens_per_model:
            raise ValueError(f"{model_key} continuation is shorter than the token minimum")
        input_ids = list(tokenized.input_ids[:end_count])
        if tokenizers[model_key].decode(
            input_ids[:prefix_count],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ) != context[:boundary]:
            raise ValueError(f"{model_key} prefix roundtrip changed")
        if tokenizers[model_key].decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ) != context:
            raise ValueError(f"{model_key} full-context roundtrip changed")
        tokenizations[model_key] = {
            "input_ids": input_ids,
            "prefix_token_count": prefix_count,
            "continuation_token_count": end_count - prefix_count,
        }
    return {
        "context_id": identity,
        "anchor_id": str(row["anchor_id"]),
        "anchor_index": int(row["anchor_index"]),
        "document_id": str(row["document_id"]),
        "source_group": str(row.get("source_group", "unknown")),
        "activation": float(row["activation"]),
        "canonical_text": context,
        "canonical_prefix": context[:boundary],
        "canonical_continuation": continuation,
        "canonical_character_boundary": boundary,
        "continuation_utf8_bytes": continuation_bytes,
        "tokenizations": tokenizations,
    }


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = FunctionalCorrespondenceContract.load(args.contract)
    config = load_main500m_config(args.config)
    if tuple(config.architecture.model_order) != contract.model_order:
        raise ValueError("main config and functional correspondence model order differ")
    if any("qwen" in str(model.model_id).lower() for model in config.models):
        raise RuntimeError("Qwen is forbidden in the functional correspondence experiment")
    registry = build_model_registry_2b(config)
    tokenizers = {
        key: load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
        for key in contract.model_order
    }
    sampled_payload = json.loads(args.sampled_keys.read_text(encoding="utf-8"))
    sampled: dict[str, set[int]] = {"pre": set(), "mid": set()}
    for key in sampled_payload["keys"]:
        site, feature = str(key).split("/")
        sampled[site].add(int(feature))
    if {site: len(values) for site, values in sampled.items()} != {"pre": 200, "mid": 200}:
        raise ValueError("Stage-1 sampled feature frame changed")
    diagnostics_path = args.heldout_analysis_dir / "aggregated_feature_diagnostics.csv"
    with diagnostics_path.open(encoding="utf-8") as handle:
        diagnostics = list(csv.DictReader(handle))
    diagnostic_index = {
        (row["site"], int(row["feature"])): row for row in diagnostics
    }
    strata = _strata(diagnostics, sampled)
    contexts = {
        site: _jsonl_selected(
            args.context_dir / f"{site}_feature_contexts.jsonl", sampled[site]
        )
        for site in ("pre", "mid")
    }
    supports = {
        site: _load_support(args.context_dir / f"{site}_natural_firings.npz")
        for site in ("pre", "mid")
    }
    rng = np.random.default_rng(contract.selection_seed)
    plans = []
    eligibility = []
    rejection_counts: dict[str, int] = {}

    def prepared(row: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return _prepare_context(
                row,
                gemma_tokenizer=tokenizers["gemma"],
                tokenizers=tokenizers,
                model_order=contract.model_order,
                contract=contract,
            )
        except ValueError as error:
            reason = str(error)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            return None

    for site in ("pre", "mid"):
        indptr, indices = supports[site]
        for stratum, quota in contract.contribution_stratum_quota.items():
            candidate_features = list(strata[(site, stratum)])
            candidate_features = [candidate_features[int(index)] for index in rng.permutation(len(candidate_features))]
            selected_count = 0
            feasible_count = 0
            for feature in candidate_features:
                active = []
                used_documents = set()
                ordered_contexts = sorted(
                    contexts[site][feature],
                    key=lambda row: (-float(row["activation"]), str(row["anchor_id"])),
                )
                for row in ordered_contexts:
                    if row["document_id"] in used_documents or not _fires(
                        indptr, indices, int(row["anchor_index"]), feature
                    ):
                        continue
                    value = prepared(row)
                    if value is None:
                        continue
                    active.append({**value, "context_activity": "active", "donor_feature": None})
                    used_documents.add(row["document_id"])
                    if len(active) == contract.active_contexts_per_component:
                        break
                if len(active) != contract.active_contexts_per_component:
                    continue
                inactive = []
                donor_features = [value for value in candidate_features if value != feature]
                for donor in donor_features:
                    for row in sorted(
                        contexts[site][donor], key=lambda value: str(value["anchor_id"])
                    ):
                        if row["document_id"] in used_documents or _fires(
                            indptr, indices, int(row["anchor_index"]), feature
                        ):
                            continue
                        value = prepared(row)
                        if value is None:
                            continue
                        inactive.append(
                            {
                                **value,
                                "context_activity": "inactive",
                                "donor_feature": donor,
                            }
                        )
                        used_documents.add(row["document_id"])
                        break
                    if len(inactive) == contract.inactive_contexts_per_component:
                        break
                if len(inactive) != contract.inactive_contexts_per_component:
                    continue
                feasible_count += 1
                if selected_count < quota:
                    diagnostic = diagnostic_index[(site, feature)]
                    plans.append(
                        {
                            "site": site,
                            "feature": feature,
                            "contribution_stratum": stratum,
                            "heldout_fires": int(diagnostic["heldout_fires"]),
                            "internal_contribution_l2_total": float(
                                diagnostic["internal_contribution_l2_total"]
                            ),
                            "active_contexts": active,
                            "inactive_contexts": inactive,
                        }
                    )
                    selected_count += 1
                if selected_count == quota:
                    break
            if selected_count != quota:
                raise ValueError(f"insufficient feasible {site}/{stratum} components")
            eligibility.append(
                {
                    "site": site,
                    "contribution_stratum": stratum,
                    "candidate_features": len(candidate_features),
                    "feasible_features_before_quota_reached": feasible_count,
                    "selected_features": selected_count,
                }
            )
    if len(plans) != 2 * contract.components_per_site:
        raise ValueError("functional correspondence component budget changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "FUNCTIONAL_CORRESPONDENCE_PLAN_FROZEN",
        "git_sha": _git_sha(),
        "contract_sha256": sha256_file(args.contract),
        "contract_fingerprint": contract.fingerprint(),
        "config_sha256": sha256_file(args.config),
        "sampled_keys_sha256": sha256_file(args.sampled_keys),
        "heldout_diagnostics_sha256": sha256_file(diagnostics_path),
        "source_artifact_sha256": {
            str(path): sha256_file(path)
            for path in (
                args.context_dir / "pre_feature_contexts.jsonl",
                args.context_dir / "mid_feature_contexts.jsonl",
                args.context_dir / "pre_natural_firings.npz",
                args.context_dir / "mid_natural_firings.npz",
            )
        },
        "eligibility": eligibility,
        "rejection_counts": rejection_counts,
        "plans": plans,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

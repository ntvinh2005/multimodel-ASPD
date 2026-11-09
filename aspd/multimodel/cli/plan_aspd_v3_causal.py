"""Build a frozen active/inactive context plan for ASPD v3 C1 parameter interventions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import CausalValidationContract


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--zscore-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _contexts(path: Path) -> dict[int, list[dict[str, Any]]]:
    return {int(row["feature"]): list(row["contexts"]) for row in _jsonl(path)}


def _load_support(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as artifact:
        return artifact["indptr"].astype(np.int64), artifact["indices"].astype(np.int64)


def _fires(indptr: np.ndarray, indices: np.ndarray, anchor: int, feature: int) -> bool:
    row = indices[int(indptr[anchor]) : int(indptr[anchor + 1])]
    return bool(np.any(row == feature))


def _select_components(
    answers: list[dict[str, Any]],
    contract: CausalValidationContract,
    *,
    arm: str,
    site: str | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(
        contract.selection_seed
        + (0 if arm == "full_only_control" else 1000)
        + (0 if site in (None, "pre") else 100)
    )
    selected = []
    selected_sites = (site,) if site is not None else ("pre", "mid")
    for selected_site in selected_sites:
        for stratum, quota in contract.site_stratum_quota.items():
            candidates = sorted(
                (
                    row
                    for row in answers
                    if row["arm"] == arm
                    and row["site"] == selected_site
                    and row["frequency_stratum"] == stratum
                ),
                key=lambda row: int(row["feature"]),
            )
            if len(candidates) < quota:
                raise ValueError(
                    f"insufficient {arm}/{selected_site}/{stratum} causal candidates: "
                    f"eligible={len(candidates)} required={quota}"
                )
            indices = np.sort(rng.choice(len(candidates), size=quota, replace=False))
            selected.extend(candidates[int(index)] for index in indices)
    return selected


def _context_plan(
    selected: dict[str, Any],
    all_answers: list[dict[str, Any]],
    contexts: dict[int, list[dict[str, Any]]],
    indptr: np.ndarray,
    indices: np.ndarray,
    contract: CausalValidationContract,
    *,
    donor_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    feature = int(selected["feature"])
    excluded_documents = set(selected["target_document_ids"])
    active = [
        row
        for row in contexts[feature][contract.active_context_rank_start :]
        if row["document_id"] not in excluded_documents
        and _fires(indptr, indices, int(row["anchor_index"]), feature)
    ][: contract.active_contexts_per_component]
    if len(active) != contract.active_contexts_per_component:
        raise ValueError(f"feature {feature} lacks held-back naturally active contexts")
    used_documents = excluded_documents | {row["document_id"] for row in active}
    inactive = []
    donors = donor_rows
    if donors is None:
        donors = sorted(
            (
                row
                for row in all_answers
                if row["arm"] == selected["arm"]
                and row["site"] == selected["site"]
                and int(row["feature"]) != feature
                and row["frequency_stratum"] == selected["frequency_stratum"]
            ),
            key=lambda row: int(row["feature"]),
        )
    for donor in donors:
        if int(donor["feature"]) == feature:
            continue
        for row in contexts[int(donor["feature"])]:
            if row["document_id"] in used_documents or not str(row.get("context", "")).strip():
                continue
            if _fires(indptr, indices, int(row["anchor_index"]), feature):
                continue
            inactive.append({**row, "donor_feature": int(donor["feature"])})
            used_documents.add(row["document_id"])
            if len(inactive) == contract.inactive_contexts_per_component:
                break
        if len(inactive) == contract.inactive_contexts_per_component:
            break
    if len(inactive) != contract.inactive_contexts_per_component:
        raise ValueError(f"feature {feature} lacks document-disjoint inactive controls")
    return {
        "arm": selected["arm"],
        "site": selected["site"],
        "feature": feature,
        "frequency_stratum": selected["frequency_stratum"],
        "feature_fires": int(selected["feature_fires"]),
        "feature_unique_documents": int(selected["feature_unique_documents"]),
        "active_contexts": active,
        "inactive_contexts": inactive,
        "discovery_context_document_ids": sorted(excluded_documents),
    }


def _causal_candidate_census(
    *,
    arm: str,
    site: str,
    answers: list[dict[str, Any]],
    contexts: dict[int, list[dict[str, Any]]],
    indices: np.ndarray,
    contract: CausalValidationContract,
) -> list[dict[str, Any]]:
    discovery_features = {
        int(row["feature"])
        for row in answers
        if row["arm"] == arm and row["site"] == site
    }
    counts = np.bincount(indices, minlength=8192)
    eligible = np.asarray(
        sorted(
            feature
            for feature, rows in contexts.items()
            if feature not in discovery_features
            and len(rows) >= contract.active_contexts_per_component
        ),
        dtype=np.int64,
    )
    ordered = eligible[np.lexsort((eligible, counts[eligible]))]
    parts = np.array_split(ordered, 3)
    census = []
    for stratum, features in zip(("low", "medium", "high"), parts, strict=True):
        for feature in features:
            census.append(
                {
                    "arm": arm,
                    "site": site,
                    "feature": int(feature),
                    "frequency_stratum": stratum,
                    "feature_fires": int(counts[feature]),
                    "feature_unique_documents": len(contexts[int(feature)]),
                    "target_document_ids": [],
                }
            )
    return census


def main() -> None:
    args = _args()
    contract = CausalValidationContract.load(args.contract)
    answers = _jsonl(args.analysis_dir / "intruder_answer_key.jsonl")
    directories = {
        "full_only_control": args.control_dir,
        "per_anchor_zscore": args.zscore_dir,
    }
    plans = []
    source_hashes = {}
    eligibility = []
    for arm, directory in directories.items():
        for site in ("pre", "mid"):
            context_path = directory / f"{site}_feature_contexts.jsonl"
            support_path = directory / f"{site}_natural_firings.npz"
            source_hashes[f"{arm}/{site}/contexts"] = sha256_file(context_path)
            source_hashes[f"{arm}/{site}/support"] = sha256_file(support_path)
            context_rows = _contexts(context_path)
            indptr, indices = _load_support(support_path)
            census = _causal_candidate_census(
                arm=arm,
                site=site,
                answers=answers,
                contexts=context_rows,
                indices=indices,
                contract=contract,
            )
            feasible_answers = []
            feasible_plans = {}
            excluded_reasons: dict[str, int] = {}
            donors_by_stratum = {
                stratum: sorted(
                    (row for row in census if row["frequency_stratum"] == stratum),
                    key=lambda row: int(row["feature"]),
                )
                for stratum in contract.site_stratum_quota
            }
            for row in census:
                try:
                    plan = _context_plan(
                        row,
                        census,
                        context_rows,
                        indptr,
                        indices,
                        contract,
                        donor_rows=donors_by_stratum[row["frequency_stratum"]],
                    )
                except ValueError as error:
                    reason = str(error).split(" lacks ", maxsplit=1)[-1]
                    excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
                    continue
                feasible_answers.append(row)
                feasible_plans[int(row["feature"])] = plan
            selected = _select_components(
                feasible_answers, contract, arm=arm, site=site
            )
            plans.extend(feasible_plans[int(row["feature"])] for row in selected)
            eligibility.append(
                {
                    "arm": arm,
                    "site": site,
                    "dictionary_features": 8192,
                    "intruder_discovery_features_excluded": sum(
                        row["arm"] == arm and row["site"] == site for row in answers
                    ),
                    "active_context_candidate_count": len(census),
                    "eligible_count": len(feasible_answers),
                    "selected_count": len(selected),
                    "eligible_by_stratum": {
                        stratum: sum(
                            row["frequency_stratum"] == stratum
                            for row in feasible_answers
                        )
                        for stratum in contract.site_stratum_quota
                    },
                    "excluded_reasons": excluded_reasons,
                }
            )
    if len(plans) != len(contract.arms) * contract.components_per_arm:
        raise ValueError("causal plan component budget changed")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite causal plan: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "CAUSAL_SCREEN_PLAN_FROZEN",
        "contract_sha256": sha256_file(args.contract),
        "contract_fingerprint": contract.fingerprint(),
        "git_sha": _git_sha(),
        "analysis_manifest_sha256": sha256_file(args.analysis_dir / "experiment_manifest.json"),
        "source_artifact_sha256": source_hashes,
        "eligibility": eligibility,
        "plans": plans,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

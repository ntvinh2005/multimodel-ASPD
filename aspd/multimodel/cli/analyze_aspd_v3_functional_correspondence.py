"""Analyze signed three-model ASPD v3.1 functional correspondence effects."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.functional_correspondence import FunctionalCorrespondenceContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load(
    directories: list[Path],
    contract: FunctionalCorrespondenceContract,
    plan_sha: str,
    contract_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    manifests = []
    targets = []
    seen = set()
    for directory in directories:
        manifest_path = directory / "experiment_manifest.json"
        result_path = directory / "intervention_results.jsonl"
        target_path = directory / "target_selection.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = str(manifest["model"])
        if model in seen:
            raise ValueError(f"duplicate functional correspondence model {model}")
        seen.add(model)
        if manifest.get("status") != "FUNCTIONAL_CORRESPONDENCE_CAUSAL_RUN_COMPLETE":
            raise ValueError(f"incomplete functional correspondence input {directory}")
        if manifest.get("source_weights_unchanged") != "PASS" or manifest.get(
            "zero_strength_invariance"
        ) != "PASS":
            raise ValueError(f"causal safety gate failed for {model}")
        if manifest.get("checkpoint_manifest_sha256") != contract.checkpoint_manifest_sha256:
            raise ValueError(f"checkpoint changed for {model}")
        if manifest.get("contract_sha256") != contract_sha:
            raise ValueError(f"contract changed for {model}")
        if manifest.get("plan_sha256") != plan_sha:
            raise ValueError(f"plan changed for {model}")
        if manifest.get("result_sha256") != sha256_file(result_path):
            raise ValueError(f"result hash changed for {model}")
        with result_path.open(encoding="utf-8") as handle:
            model_rows = [json.loads(line) for line in handle]
        if any(row["model"] != model for row in model_rows):
            raise ValueError(f"row model changed for {model}")
        model_targets = json.loads(target_path.read_text(encoding="utf-8"))
        if manifest.get("target_selection_sha256") != sha256_file(target_path):
            raise ValueError(f"target selection hash changed for {model}")
        rows.extend(model_rows)
        targets.extend(model_targets)
        manifests.append({"directory": str(directory), **manifest})
    if seen != set(contract.model_order):
        raise ValueError(f"model coverage changed: {seen}")
    return rows, manifests, targets


PRIMARY = "edited_minus_clean_continuation_logprob_per_utf8_byte"


def _effect_rows(
    rows: list[dict[str, Any]], *, kind: str, seed: int | None
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["intervention_kind"] != kind or float(row["strength"]) != 1.0:
            continue
        if seed is not None and int(row["control_seed"]) != seed:
            continue
        grouped[(row["model"], row["site"], int(row["feature"]))].append(row)
    output = []
    for (model, site, feature), values in sorted(grouped.items()):
        activities = {
            activity: [float(row[PRIMARY]) for row in values if row["context_activity"] == activity]
            for activity in ("active", "inactive")
        }
        if any(len(selected) != 4 for selected in activities.values()):
            raise ValueError(f"context count changed for {model}/{site}/{feature}/{kind}/{seed}")
        active = float(np.mean(activities["active"]))
        inactive = float(np.mean(activities["inactive"]))
        output.append(
            {
                "model": model,
                "site": site,
                "feature": feature,
                "intervention_kind": kind,
                "control_seed": seed,
                "active_effect": active,
                "inactive_effect": inactive,
                "specificity_effect": active - inactive,
                "active_absolute_effect": float(np.mean(np.abs(activities["active"]))),
                "inactive_absolute_effect": float(np.mean(np.abs(activities["inactive"]))),
            }
        )
    return output


def _interval(values: np.ndarray, *, seed: int, replicates: int) -> dict[str, Any]:
    if not len(values):
        raise ValueError("cannot bootstrap an empty statistic")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(lower), float(upper)],
        "bootstrap_unit": "component",
        "bootstrap_replicates": replicates,
    }


def _rho_interval(
    left: np.ndarray, right: np.ndarray, *, seed: int, replicates: int
) -> dict[str, Any]:
    point = float(spearmanr(left, right).statistic)
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(replicates):
        indices = rng.integers(0, len(left), size=len(left))
        value = float(spearmanr(left[indices], right[indices]).statistic)
        if np.isfinite(value):
            sampled.append(value)
    if not sampled:
        raise ValueError("all bootstrap rank correlations were undefined")
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "n": int(len(left)),
        "rho": point,
        "ci95": [float(lower), float(upper)],
        "bootstrap_unit": "component",
        "valid_bootstrap_replicates": len(sampled),
    }


def _signed_alignment(
    effects: list[dict[str, Any]],
    contract: FunctionalCorrespondenceContract,
    *,
    field: str,
    seed_offset: int,
) -> dict[str, Any]:
    index = {
        (row["model"], row["site"], row["feature"]): float(row[field]) for row in effects
    }
    components = sorted({(row["site"], row["feature"]) for row in effects})
    if len(components) != 2 * contract.components_per_site:
        raise ValueError("signed-alignment component count changed")
    matrix = np.asarray(
        [[index[(model, site, feature)] for model in contract.model_order] for site, feature in components]
    )
    same_sign = np.asarray(
        [bool(np.all(row > 0) or np.all(row < 0)) for row in matrix], dtype=np.float64
    )
    pairwise = {}
    sequence = 0
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = contract.model_order[left_index]
        right = contract.model_order[right_index]
        pairwise[f"{left}_vs_{right}"] = {
            **_rho_interval(
                matrix[:, left_index],
                matrix[:, right_index],
                seed=contract.bootstrap_seed + seed_offset + sequence,
                replicates=contract.bootstrap_replicates,
            ),
            "sign_concordance": _interval(
                (np.sign(matrix[:, left_index]) == np.sign(matrix[:, right_index])).astype(
                    np.float64
                ),
                seed=contract.bootstrap_seed + seed_offset + 100 + sequence,
                replicates=contract.bootstrap_replicates,
            ),
        }
        sequence += 1
    return {
        "all_three_sign_agreement": _interval(
            same_sign,
            seed=contract.bootstrap_seed + seed_offset + 200,
            replicates=contract.bootstrap_replicates,
        ),
        "pairwise": pairwise,
    }


def _context_alignment(
    rows: list[dict[str, Any]], contract: FunctionalCorrespondenceContract
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["intervention_kind"] == "learned" and float(row["strength"]) == 1.0
    ]
    index = {(row["model"], row["context_id"]): float(row[PRIMARY]) for row in selected}
    output = {}
    for activity in ("active", "inactive"):
        context_ids = sorted(
            {row["context_id"] for row in selected if row["context_activity"] == activity}
        )
        output[activity] = {}
        for left, right in (("gemma", "smol"), ("gemma", "granite"), ("smol", "granite")):
            left_values = np.asarray([index[(left, context_id)] for context_id in context_ids])
            right_values = np.asarray([index[(right, context_id)] for context_id in context_ids])
            output[activity][f"{left}_vs_{right}"] = _rho_interval(
                left_values,
                right_values,
                seed=contract.bootstrap_seed + len(output[activity]) + (0 if activity == "active" else 50),
                replicates=contract.bootstrap_replicates,
            )
    return output


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FunctionalCorrespondenceContract.load(args.contract)
    plan_sha = sha256_file(args.plan)
    rows, manifests, targets = _load(
        args.input_dir, contract, plan_sha, sha256_file(args.contract)
    )
    expected_rows = len(contract.model_order) * 2 * contract.components_per_site * 8 * 6
    if len(rows) != expected_rows:
        raise ValueError(f"functional correspondence row count changed: {len(rows)}")
    zeros = [
        row
        for row in rows
        if row["intervention_kind"] == "learned" and float(row["strength"]) == 0.0
    ]
    if len(zeros) != len(contract.model_order) * 2 * contract.components_per_site * 8 or any(
        float(row[PRIMARY]) != 0.0 for row in zeros
    ):
        raise ValueError("zero-strength correspondence invariant failed")
    learned = _effect_rows(rows, kind="learned", seed=None)
    random_seeds = {
        seed: _effect_rows(rows, kind="random_rank_one", seed=seed)
        for seed in contract.random_rank_one_seeds
    }
    random_component = _effect_rows(
        rows, kind="random_component", seed=contract.random_component_seed
    )
    learned_alignment = {
        field: _signed_alignment(learned, contract, field=field, seed_offset=index * 500)
        for index, field in enumerate(("active_effect", "specificity_effect"))
    }
    random_alignment = {
        str(seed): _signed_alignment(
            values, contract, field="active_effect", seed_offset=2000 + index * 500
        )
        for index, (seed, values) in enumerate(random_seeds.items())
    }
    random_sign_point = float(
        np.mean(
            [
                value["all_three_sign_agreement"]["mean"]
                for value in random_alignment.values()
            ]
        )
    )
    pairwise_active = learned_alignment["active_effect"]["pairwise"]
    upper_below_zero = sum(value["ci95"][1] < 0 for value in pairwise_active.values())
    all_lower_above_zero = all(value["ci95"][0] > 0 for value in pairwise_active.values())
    learned_sign = learned_alignment["active_effect"]["all_three_sign_agreement"]
    if upper_below_zero >= 2 or learned_sign["ci95"][1] < random_sign_point:
        verdict = "SUPPORTED"
        rationale = "predeclared incompatibility rule passed"
    elif all_lower_above_zero and learned_sign["mean"] > random_sign_point:
        verdict = "NOT_SUPPORTED"
        rationale = "predeclared correspondence-weakened rule passed"
    else:
        verdict = "INCONCLUSIVE"
        rationale = "neither predeclared directional rule passed"
    target_by_component: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in targets:
        target_by_component[(row["site"], int(row["feature"]))].append(
            str(row["target"]).split(".")[-1]
        )
    logical_target_agreement = float(
        np.mean([len(set(values)) == 1 for values in target_by_component.values()])
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "component_effects.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(learned[0]))
        writer.writeheader()
        writer.writerows(learned)
    summary = {
        "schema_version": 1,
        "status": "FUNCTIONAL_CORRESPONDENCE_ANALYSIS_COMPLETE",
        "h7_verdict": verdict,
        "h7_verdict_rationale": rationale,
        "primary_metric": contract.primary_metric,
        "components": 2 * contract.components_per_site,
        "active_contexts_per_component": contract.active_contexts_per_component,
        "inactive_contexts_per_component": contract.inactive_contexts_per_component,
        "learned_signed_alignment": learned_alignment,
        "random_rank_one_active_alignment": random_alignment,
        "random_rank_one_all_three_sign_agreement_point_mean": random_sign_point,
        "random_component_active_alignment": _signed_alignment(
            random_component, contract, field="active_effect", seed_offset=4000
        ),
        "same_canonical_context_alignment": _context_alignment(rows, contract),
        "logical_target_name_agreement_fraction": logical_target_agreement,
        "measurement_boundaries": {
            "effect_sign": "edited minus clean log-probability; negative means ablation reduced likelihood",
            "per_byte": "normalizes different tokenizer segmentations by canonical UTF-8 continuation bytes",
            "target_selection": contract.target_selection_policy,
            "universal_mechanism": "signed correspondence is necessary evidence but does not alone prove identical algorithms",
        },
    }
    _write_json(args.output_dir / "functional_correspondence_summary.json", summary)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "FUNCTIONAL_CORRESPONDENCE_ANALYSIS_COMPLETE",
            "git_sha": _git_sha(),
            "contract_sha256": sha256_file(args.contract),
            "plan_sha256": plan_sha,
            "input_manifests": manifests,
            "summary_sha256": sha256_file(
                args.output_dir / "functional_correspondence_summary.json"
            ),
        },
    )
    (args.output_dir / "COMPLETE").write_text(
        "FUNCTIONAL_CORRESPONDENCE_ANALYSIS_COMPLETE\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Analyze matched ASPD v3 direct parameter causal screens."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import CausalValidationContract

PRIMARY_METRICS = ("kl_clean_to_edited", "logit_delta_rms")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_inputs(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    seen = set()
    for directory in paths:
        manifest_path = directory / "experiment_manifest.json"
        result_path = directory / "intervention_results.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (manifest["arm"], manifest["model"])
        if key in seen:
            raise ValueError(f"duplicate causal input for {key}")
        seen.add(key)
        if manifest.get("status") != "C1_DIRECT_PARAMETER_CAUSAL_SCREEN_COMPLETE":
            raise ValueError(f"incomplete causal input: {directory}")
        if manifest.get("source_weights_unchanged") != "PASS":
            raise ValueError(f"source restoration failed: {directory}")
        if manifest.get("zero_strength_invariance") != "PASS":
            raise ValueError(f"zero-strength invariance failed: {directory}")
        if manifest.get("result_sha256") != sha256_file(result_path):
            raise ValueError(f"causal result hash mismatch: {directory}")
        with result_path.open(encoding="utf-8") as handle:
            loaded = [json.loads(line) for line in handle]
        if any((row["arm"], row["model"]) != key for row in loaded):
            raise ValueError(f"result identity differs from manifest: {directory}")
        manifests.append({"directory": str(directory), **manifest})
        rows.extend(loaded)
    return rows, manifests


def _validate(rows: list[dict[str, Any]], contract: CausalValidationContract) -> None:
    expected = {(arm, model) for arm in contract.arms for model in contract.model_order}
    observed = {(row["arm"], row["model"]) for row in rows}
    if observed != expected:
        raise ValueError(f"causal arm/model coverage differs: {observed} != {expected}")
    for arm, model in sorted(expected):
        subset = [row for row in rows if (row["arm"], row["model"]) == (arm, model)]
        features = {(row["site"], int(row["feature"])) for row in subset}
        if len(features) != contract.components_per_arm:
            raise ValueError(f"component count differs for {arm}/{model}")
        zeros = [
            row
            for row in subset
            if row["intervention_kind"] == "learned" and float(row["strength"]) == 0.0
        ]
        if not zeros or any(float(row[metric]) != 0.0 for row in zeros for metric in PRIMARY_METRICS):
            raise ValueError(f"zero-strength output changed for {arm}/{model}")
        if any("realized_weight_delta_frobenius_norm" not in row["intervention_metadata"] for row in subset):
            raise ValueError(f"missing realized BF16 perturbation audit for {arm}/{model}")


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("empty or non-finite analysis group")
    return float(array.mean())


def _component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["arm"],
                row["model"],
                row["site"],
                int(row["feature"]),
                row["frequency_stratum"],
                row["target"],
                row["intervention_kind"],
                float(row["strength"]),
                row["control_seed"],
            )
        ].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        arm, model, site, feature, stratum, target, kind, strength, seed = key
        by_activity = {
            activity: [row for row in values if row["context_activity"] == activity]
            for activity in ("active", "inactive")
        }
        if any(len(activity_rows) != 4 for activity_rows in by_activity.values()):
            raise ValueError(f"causal context count differs for {key}")
        metadata = values[0]["intervention_metadata"]
        if any(row["intervention_metadata"] != metadata for row in values):
            raise ValueError(f"intervention metadata changed across contexts for {key}")
        summary = {
            "arm": arm,
            "model": model,
            "site": site,
            "feature": feature,
            "frequency_stratum": stratum,
            "target": target,
            "intervention_kind": kind,
            "strength": strength,
            "control_seed": seed,
            "realized_weight_delta_frobenius_norm": float(
                metadata["realized_weight_delta_frobenius_norm"]
            ),
            "realized_to_intended_norm_ratio": float(
                metadata["realized_to_intended_norm_ratio"]
            ),
            "realized_to_intended_cosine": float(metadata["realized_to_intended_cosine"]),
            "realized_nonzero_fraction": float(metadata["realized_nonzero_fraction"]),
        }
        for metric in PRIMARY_METRICS:
            for activity, activity_rows in by_activity.items():
                summary[f"{activity}_{metric}"] = _mean(row[metric] for row in activity_rows)
            summary[f"specificity_{metric}"] = (
                summary[f"active_{metric}"] - summary[f"inactive_{metric}"]
            )
            realized_norm = summary["realized_weight_delta_frobenius_norm"]
            summary[f"active_{metric}_per_realized_norm"] = (
                summary[f"active_{metric}"] / realized_norm if realized_norm else 0.0
            )
            summary[f"specificity_{metric}_per_realized_norm"] = (
                summary[f"specificity_{metric}"] / realized_norm if realized_norm else 0.0
            )
        result.append(summary)
    return result


def _average_random_rank_one(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        if row["intervention_kind"] == "random_rank_one":
            grouped[(row["arm"], row["model"], row["site"], row["feature"])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        row = {key_name: value for key_name, value in zip(("arm", "model", "site", "feature"), key, strict=True)}
        row["intervention_kind"] = "random_rank_one_seed_mean"
        numeric = [
            name
            for name, value in values[0].items()
            if name not in row and name not in {"control_seed", "target", "frequency_stratum", "intervention_kind"}
            and isinstance(value, (int, float))
        ]
        for name in numeric:
            row[name] = _mean(value[name] for value in values)
        row["target"] = values[0]["target"]
        row["frequency_stratum"] = values[0]["frequency_stratum"]
        row["strength"] = 1.0
        row["control_seed"] = None
        result.append(row)
    return result


def _bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("cannot bootstrap empty or non-finite values")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "n_components": int(len(values)),
        "mean": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_unit": "component",
        "bootstrap_replicates": replicates,
    }


def _context_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["strength"]) == 1.0:
            grouped[
                (
                    row["arm"],
                    row["model"],
                    row["site"],
                    int(row["feature"]),
                    row["context_activity"],
                    row["anchor_id"],
                    row["document_id"],
                )
            ].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        learned = [row for row in values if row["intervention_kind"] == "learned"]
        rank_one = [row for row in values if row["intervention_kind"] == "random_rank_one"]
        random_component = [
            row for row in values if row["intervention_kind"] == "random_component"
        ]
        if len(learned) != 1 or len(rank_one) != 3 or len(random_component) != 1:
            raise ValueError(f"causal control coverage differs for context {key}")
        identity = {
            name: value
            for name, value in zip(
                ("arm", "model", "site", "feature", "context_activity", "anchor_id", "document_id"),
                key,
                strict=True,
            )
        }
        for control_name, controls in (
            ("random_rank_one_seed_mean", rank_one),
            ("random_component", random_component),
        ):
            result = {**identity, "control": control_name}
            for metric in PRIMARY_METRICS:
                result[f"learned_minus_control_{metric}"] = float(learned[0][metric]) - _mean(
                    row[metric] for row in controls
                )
            output.append(result)
    return output


def _hierarchical_interval(
    component_values: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    quantity: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    components = sorted(component_values)
    if not components:
        raise ValueError("cannot bootstrap an empty component group")

    def sampled_component(rng: np.random.Generator, component: tuple[str, int]) -> float:
        active = component_values[component]["active"]
        active_mean = float(active[rng.integers(0, len(active), size=len(active))].mean())
        if quantity == "active":
            return active_mean
        inactive = component_values[component]["inactive"]
        inactive_mean = float(inactive[rng.integers(0, len(inactive), size=len(inactive))].mean())
        return active_mean - inactive_mean

    points = []
    for component in components:
        active_mean = float(component_values[component]["active"].mean())
        if quantity == "active":
            points.append(active_mean)
        else:
            points.append(active_mean - float(component_values[component]["inactive"].mean()))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.integers(0, len(components), size=len(components))
        bootstrap[replicate] = _mean(
            sampled_component(rng, components[index]) for index in chosen
        )
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "n_components": len(components),
        "documents_per_component_activity": 4,
        "mean": _mean(points),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_unit": "component_then_document_within_activity",
        "bootstrap_replicates": replicates,
    }


def _hierarchical_estimands(
    rows: list[dict[str, Any]], contract: CausalValidationContract
) -> list[dict[str, Any]]:
    contrasts = _context_contrasts(rows)
    output = []
    sequence = 0
    for arm in contract.arms:
        for model in contract.model_order:
            for site in ("all", "pre", "mid"):
                for metric in PRIMARY_METRICS:
                    for control in ("random_rank_one_seed_mean", "random_component"):
                        selected = [
                            row
                            for row in contrasts
                            if row["arm"] == arm
                            and row["model"] == model
                            and row["control"] == control
                            and (site == "all" or row["site"] == site)
                        ]
                        component_values: dict[tuple[str, int], dict[str, np.ndarray]] = {}
                        for component in sorted({(row["site"], row["feature"]) for row in selected}):
                            component_values[component] = {
                                activity: np.asarray(
                                    [
                                        row[f"learned_minus_control_{metric}"]
                                        for row in selected
                                        if (row["site"], row["feature"]) == component
                                        and row["context_activity"] == activity
                                    ],
                                    dtype=np.float64,
                                )
                                for activity in ("active", "inactive")
                            }
                            if any(len(values) != 4 for values in component_values[component].values()):
                                raise ValueError(f"document count differs for component {component}")
                        for quantity in ("active", "specificity"):
                            interval = _hierarchical_interval(
                                component_values,
                                quantity=quantity,
                                seed=contract.bootstrap_seed + sequence,
                                replicates=contract.bootstrap_replicates,
                            )
                            sequence += 1
                            output.append(
                                {
                                    "arm": arm,
                                    "model": model,
                                    "site": site,
                                    "metric": metric,
                                    "quantity": quantity,
                                    "contrast": f"learned_minus_{control}",
                                    **interval,
                                }
                            )
    return output


def _paired_estimands(
    component_rows: list[dict[str, Any]], contract: CausalValidationContract
) -> list[dict[str, Any]]:
    random_means = _average_random_rank_one(component_rows)
    index = {
        (row["arm"], row["model"], row["site"], row["feature"], row["intervention_kind"], float(row["strength"])): row
        for row in component_rows + random_means
    }
    output = []
    sequence = 0
    for arm in contract.arms:
        for model in contract.model_order:
            for site in ("all", "pre", "mid"):
                components = sorted(
                    {
                        (row["site"], int(row["feature"]))
                        for row in component_rows
                        if row["arm"] == arm
                        and row["model"] == model
                        and (site == "all" or row["site"] == site)
                    }
                )
                for metric in PRIMARY_METRICS:
                    for control in ("random_rank_one_seed_mean", "random_component"):
                        for quantity in ("active", "specificity"):
                            field = f"{quantity}_{metric}"
                            differences = []
                            normalized_differences = []
                            for component_site, feature in components:
                                learned = index[(arm, model, component_site, feature, "learned", 1.0)]
                                random_control = index[(arm, model, component_site, feature, control, 1.0)]
                                differences.append(learned[field] - random_control[field])
                                normalized_field = f"{field}_per_realized_norm"
                                normalized_differences.append(
                                    learned[normalized_field] - random_control[normalized_field]
                                )
                            for normalization, values in (
                                ("raw", differences),
                                ("per_realized_weight_delta_norm", normalized_differences),
                            ):
                                record = _bootstrap_mean(
                                    np.asarray(values),
                                    seed=contract.bootstrap_seed + sequence,
                                    replicates=contract.bootstrap_replicates,
                                )
                                sequence += 1
                                output.append(
                                    {
                                        "arm": arm,
                                        "model": model,
                                        "site": site,
                                        "metric": metric,
                                        "quantity": quantity,
                                        "contrast": f"learned_minus_{control}",
                                        "normalization": normalization,
                                        **record,
                                    }
                                )
    return output


def _dose_response(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["arm"], row["model"], row["site"], row["feature"], float(row["strength"])): row
        for row in component_rows
        if row["intervention_kind"] == "learned"
    }
    output = []
    groups = sorted({key[:3] for key in index})
    for arm, model, site in groups:
        components = sorted({key[3] for key in index if key[:3] == (arm, model, site)})
        for metric in PRIMARY_METRICS:
            field = f"active_{metric}"
            half = np.asarray([index[(arm, model, site, feature, 0.5)][field] for feature in components])
            full = np.asarray([index[(arm, model, site, feature, 1.0)][field] for feature in components])
            output.append(
                {
                    "arm": arm,
                    "model": model,
                    "site": site,
                    "metric": metric,
                    "n_components": len(components),
                    "mean_strength_0_5": float(half.mean()),
                    "mean_strength_1_0": float(full.mean()),
                    "fraction_monotonic": float(np.mean(full >= half)),
                }
            )
    return output


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _cross_model(component_rows: list[dict[str, Any]], contract: CausalValidationContract) -> list[dict[str, Any]]:
    learned = {
        (row["arm"], row["model"], row["site"], row["feature"]): row
        for row in component_rows
        if row["intervention_kind"] == "learned" and float(row["strength"]) == 1.0
    }
    output = []
    for arm in contract.arms:
        for site in ("all", "pre", "mid"):
            features = sorted(
                {
                    (key[2], key[3])
                    for key in learned
                    if key[0] == arm and (site == "all" or key[2] == site)
                }
            )
            for metric in PRIMARY_METRICS:
                for quantity in ("active", "specificity"):
                    field = f"{quantity}_{metric}"
                    left = np.asarray([learned[(arm, "gemma", component_site, feature)][field] for component_site, feature in features])
                    right = np.asarray([learned[(arm, "smol", component_site, feature)][field] for component_site, feature in features])
                    output.append(
                        {
                            "arm": arm,
                            "site": site,
                            "metric": metric,
                            "quantity": quantity,
                            "n_components": len(features),
                            "pearson": _correlation(left, right),
                            "spearman": _correlation(_rank(left), _rank(right)),
                            "scope_warning": "magnitude agreement only; not functional-direction correspondence",
                        }
                    )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite causal analysis: {args.output_dir}")
    contract = CausalValidationContract.load(args.contract)
    rows, manifests = _load_inputs(args.input_dir)
    _validate(rows, contract)
    components = _component_rows(rows)
    estimands = _paired_estimands(components, contract)
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "component_effects.csv", components)
    _write_json(
        args.output_dir / "primary_hierarchical_estimands.json",
        _hierarchical_estimands(rows, contract),
    )
    _write_json(args.output_dir / "paired_estimands.json", estimands)
    _write_json(args.output_dir / "dose_response.json", _dose_response(components))
    _write_json(args.output_dir / "cross_model_correspondence.json", _cross_model(components, contract))
    _write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "schema_version": 1,
            "status": "CAUSAL_SCREEN_ANALYSIS_COMPLETE",
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "contract_sha256": sha256_file(args.contract),
            "input_manifests": manifests,
            "input_rows": len(rows),
            "component_intervention_rows": len(components),
            "primary_metrics": list(PRIMARY_METRICS),
            "primary_bootstrap_unit": "component_then_document_within_activity",
            "sensitivity_bootstrap_unit": "component",
            "limitations": [
                "single training seed per arm",
                "C1 single-matrix interventions only",
                "cross-model correlations compare effect magnitude, not semantic direction",
            ],
        },
    )


if __name__ == "__main__":
    main()

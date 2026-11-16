"""Analyze frozen independent rule validation for ASPD v3.1 feature coherence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from aspd.multimodel.cli.analyze_aspd_v3_interpretability_clusters import _fit_feature
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--rules-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--selected-components", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--confirmatory-input", type=Path, required=True)
    parser.add_argument("--confirmatory-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _binary_stats(rows: list[dict[str, Any]], prediction: str) -> dict[str, float | int]:
    positive = [row for row in rows if row["actual_natural_firing"]]
    negative = [row for row in rows if not row["actual_natural_firing"]]
    if not positive or not negative:
        raise ValueError("balanced accuracy requires both natural firing and non-firing rows")
    sensitivity = float(np.mean([bool(row[prediction]) for row in positive]))
    specificity = float(np.mean([not bool(row[prediction]) for row in negative]))
    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_n": len(positive),
        "negative_n": len(negative),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample matched positive/negative pair IDs; preserve general/multi pairing."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    pairs = []
    for pair_id, members in sorted(grouped.items()):
        positive = [row for row in members if row["actual_natural_firing"]]
        negative = [row for row in members if not row["actual_natural_firing"]]
        if len(positive) == 1 and len(negative) == 1:
            pairs.append((pair_id, positive[0], negative[0]))
    if len(pairs) < 2:
        raise ValueError("too few complete matched pairs for bootstrap")
    general = np.asarray(
        [[float(pos["general_yes"]), float(not neg["general_yes"])] for _, pos, neg in pairs]
    )
    multi = np.asarray(
        [[float(pos["multi_yes"]), float(not neg["multi_yes"])] for _, pos, neg in pairs]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(replicates, len(pairs)))
    general_ba = 0.5 * general[indices].mean(axis=1).sum(axis=1)
    multi_ba = 0.5 * multi[indices].mean(axis=1).sum(axis=1)
    return general_ba, multi_ba, multi_ba - general_ba


def _aggregate_component_mean(
    rows: list[dict[str, Any]], field: str, *, replicates: int, seed: int
) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": _quantile_interval(bootstrap), "components": len(values)}


def _adjusted_site_difference(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    """OLS PRE coefficient adjusted for log discovery frequency and cohort."""

    y = np.asarray([float(row["general_balanced_accuracy"]) for row in rows])
    x = np.asarray(
        [
            [
                1.0,
                float(row["site"] == "pre"),
                math.log1p(float(row["discovery_natural_firing_events"])),
                float(row["cohort"] == "new_random"),
            ]
            for row in rows
        ]
    )
    estimate = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        indices = rng.integers(0, len(rows), len(rows))
        if len(np.unique(x[indices, 1])) < 2:
            continue
        bootstrap.append(float(np.linalg.lstsq(x[indices], y[indices], rcond=None)[0][1]))
    return {
        "pre_minus_mid_adjusted_general_balanced_accuracy": estimate,
        "ci95": _quantile_interval(np.asarray(bootstrap)),
        "components": len(rows),
        "covariates": ["log1p_discovery_natural_firing_events", "new_random_cohort"],
    }


def _analyze_controls(
    *,
    contract: FeatureCoherenceAnalysisContract,
    heldout_dir: Path,
    confirmatory_input: Path,
    confirmatory_embeddings: Path,
) -> list[dict[str, Any]]:
    input_rows = _jsonl(confirmatory_input)
    embeddings = np.load(confirmatory_embeddings, mmap_mode="r", allow_pickle=False)
    if len(input_rows) != len(embeddings):
        raise ValueError("confirmatory embedding rows differ from input")
    row_to_index = {str(row["row_id"]): index for index, row in enumerate(input_rows)}
    examples = _jsonl(heldout_dir / "heldout_validation_examples.jsonl")
    firing_ids: dict[str, list[str]] = defaultdict(list)
    for row in examples:
        if row["actual_natural_firing"]:
            firing_ids[str(row["component_key"])].append(str(row["row_id"]))
    controls: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for row in _jsonl(heldout_dir / "cluster_control_sets.jsonl"):
        controls[str(row["component_key"])][str(row["control"])] = [str(value) for value in row["row_ids"]]
    output = []
    conditions = ("natural_firing", "random_source_length_matched_nonfiring", "topic_matched_nonfiring", "frequency_matched_other_component")
    for component_index, key in enumerate(sorted(firing_ids)):
        sets = {"natural_firing": firing_ids[key], **controls[key]}
        for condition_index, condition in enumerate(conditions):
            row_ids = sets[condition]
            values = np.asarray(embeddings[[row_to_index[row_id] for row_id in row_ids]], dtype=np.float32)
            metrics, _labels, _centers = _fit_feature(
                values,
                np.zeros(len(values), dtype=np.float32),
                candidate_k=[2, 3, 4, 5],
                restarts=10,
                seed=contract.bootstrap_seed + component_index * 100_000 + condition_index * 10_000,
                minimum_silhouette=0.10,
                minimum_cluster_fraction=0.10,
                minimum_stability=0.60,
            )
            if not math.isfinite(float(metrics["activation_kruskal_pvalue_exploratory"])):
                metrics["activation_kruskal_pvalue_exploratory"] = None
            output.append({"component_key": key, "condition": condition, "contexts": len(row_ids), **metrics})
    return output


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    calibration_manifest = _json(args.calibration_dir / "experiment_manifest.json")
    validation_manifest = _json(args.validation_dir / "experiment_manifest.json")
    tasks_manifest = _json(args.tasks_dir / "preparation_manifest.json")
    rules_manifest = _json(args.rules_dir / "freeze_manifest.json")
    heldout_manifest = _json(args.heldout_dir / "preparation_manifest.json")
    if calibration_manifest["status"] != "ASPD_V3_1_FEATURE_RULE_CALIBRATION_COMPLETE":
        raise ValueError("independent validator calibration did not pass")
    if validation_manifest["status"] != "ASPD_V3_1_FEATURE_RULE_VALIDATION_COMPLETE":
        raise ValueError("held-out validator run is incomplete")
    if any(
        manifest["analysis_contract_fingerprint"] != contract.fingerprint()
        for manifest in (calibration_manifest, validation_manifest, tasks_manifest, rules_manifest, heldout_manifest)
    ):
        raise ValueError("analysis contract fingerprint mismatch")
    tasks_path = args.tasks_dir / "validation_tasks.jsonl"
    if sha256_file(tasks_path) != tasks_manifest["validation_tasks_sha256"]:
        raise ValueError("validation tasks changed")
    tasks = _jsonl(tasks_path)
    results = _jsonl(args.validation_dir / "validator_results.jsonl")
    if sha256_file(args.validation_dir / "validator_results.jsonl") != validation_manifest["results_sha256"]:
        raise ValueError("validator results changed")
    result_index = {str(row["task_id"]): row for row in results}
    if len(tasks) != len(results) or len(result_index) != len(tasks):
        raise ValueError("validator result coverage changed")
    joined = []
    for task in tasks:
        result = result_index[str(task["task_id"])]
        joined.append({**task, "prediction": result["prediction"], "valid": result["valid"], "yes": result["prediction"] == "YES"})

    selected = {str(row["component_key"]): row for row in _json(args.selected_components)}
    clusters = {str(row["component_key"]): row for row in _jsonl(args.discovery_dir / "component_clusters.jsonl")}
    rules = {str(row["component_key"]): row for row in _jsonl(args.rules_dir / "candidate_rules.jsonl")}
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_component[str(row["component_key"])].append(row)
    component_rows = []
    bin_rows = []
    for component_index, key in enumerate(sorted(by_component)):
        component_tasks = by_component[key]
        ordinary = [row for row in component_tasks if row["condition"] == "ordinary"]
        by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in ordinary:
            by_example[str(row["example_id"])].append(row)
        examples = []
        for example_id, rows in by_example.items():
            general = [row for row in rows if row["rule_type"] == "general"]
            groups = [row for row in rows if row["rule_type"] == "group"]
            if len(general) != 1 or not groups:
                raise ValueError(f"incomplete ordinary rules for {key}/{example_id}")
            base = general[0]
            examples.append(
                {
                    "example_id": example_id,
                    "pair_id": base["pair_id"],
                    "actual_natural_firing": base["actual_natural_firing"],
                    "target_discovery_cluster": base["target_discovery_cluster"],
                    "activation_bin": base["activation_bin"],
                    "document_id": base["document_id"],
                    "general_yes": base["yes"],
                    "multi_yes": any(row["yes"] for row in groups),
                }
            )
        general_stats = _binary_stats(examples, "general_yes")
        multi_stats = _binary_stats(examples, "multi_yes")
        general_boot, multi_boot, advantage_boot = _paired_bootstrap(
            examples, replicates=contract.bootstrap_replicates, seed=contract.bootstrap_seed + component_index
        )
        positive_general = [row for row in ordinary if row["rule_type"] == "general" and row["actual_natural_firing"]]
        group_counts = Counter(int(row["target_discovery_cluster"]) for row in positive_general)
        discovery_group_count = len(clusters[key]["groups"])
        covered_groups = [group for group, count in group_counts.items() if count >= contract.minimum_heldout_contexts_per_group]
        group_coverage = len(covered_groups) / discovery_group_count
        general_group_recalls = {
            str(group): float(np.mean([row["yes"] for row in positive_general if int(row["target_discovery_cluster"]) == group]))
            for group in covered_groups
        }
        minimum_general_group_recall = min(general_group_recalls.values(), default=0.0)
        qualifying_group_rules = 0
        group_rule_diagnostics = []
        group_tasks = [row for row in ordinary if row["rule_type"] == "group"]
        for source_cluster in sorted({int(row["source_cluster"]) for row in group_tasks}):
            rule_rows = [row for row in group_tasks if int(row["source_cluster"]) == source_cluster]
            within = [row for row in rule_rows if row["actual_natural_firing"] and int(row["target_discovery_cluster"]) == source_cluster]
            across = [row for row in rule_rows if row["actual_natural_firing"] and int(row["target_discovery_cluster"]) != source_cluster]
            negative = [row for row in rule_rows if not row["actual_natural_firing"]]
            within_rate = float(np.mean([row["yes"] for row in within])) if within else 0.0
            across_rate = float(np.mean([row["yes"] for row in across])) if across else 0.0
            specificity = float(np.mean([not row["yes"] for row in negative]))
            balanced_accuracy = 0.5 * (within_rate + specificity)
            within_across = within_rate - across_rate
            qualified = bool(
                len(within) >= contract.minimum_heldout_contexts_per_group
                and balanced_accuracy >= contract.polysemantic_group_rule_balanced_accuracy
                and within_across >= contract.polysemantic_within_across_advantage
            )
            qualifying_group_rules += int(qualified)
            group_rule_diagnostics.append(
                {
                    "source_cluster": source_cluster,
                    "within_n": len(within),
                    "across_n": len(across),
                    "negative_n": len(negative),
                    "within_recall": within_rate,
                    "across_positive_rate": across_rate,
                    "specificity": specificity,
                    "balanced_accuracy_own_group_vs_nonfiring": balanced_accuracy,
                    "within_minus_across": within_across,
                    "qualified_distinct_rule": qualified,
                }
            )
        general_ci = _quantile_interval(general_boot)
        multi_ci = _quantile_interval(multi_boot)
        advantage = float(multi_stats["balanced_accuracy"] - general_stats["balanced_accuracy"])
        advantage_ci = _quantile_interval(advantage_boot)
        coherent = bool(
            general_stats["balanced_accuracy"] >= contract.coherent_balanced_accuracy
            and general_ci[0] >= contract.coherent_lower_ci
            and general_stats["specificity"] >= contract.coherent_specificity
            and group_coverage >= contract.coherent_group_coverage
            and minimum_general_group_recall >= contract.coherent_minimum_group_recall
            and advantage_ci[1] < contract.coherent_multi_rule_upper_advantage
        )
        polysemantic = bool(
            multi_stats["balanced_accuracy"] >= contract.polysemantic_multi_balanced_accuracy
            and multi_ci[0] >= contract.polysemantic_multi_lower_ci
            and advantage >= contract.polysemantic_point_advantage
            and advantage_ci[0] > contract.polysemantic_lower_advantage
            and qualifying_group_rules >= 2
            and group_coverage >= contract.coherent_group_coverage
            and (
                general_stats["balanced_accuracy"] < contract.polysemantic_general_failure_balanced_accuracy
                or minimum_general_group_recall < contract.polysemantic_general_failure_group_recall
            )
        )
        no_rule = bool(
            general_stats["balanced_accuracy"] <= contract.no_rule_balanced_accuracy
            and multi_stats["balanced_accuracy"] <= contract.no_rule_balanced_accuracy
            and max(general_ci[1], multi_ci[1]) < contract.no_rule_upper_ci
        )
        if coherent:
            classification = "single_coherent_rule"
        elif polysemantic:
            classification = "multiple_unrelated_rules"
        elif no_rule:
            classification = "no_reliable_linguistic_rule"
        else:
            classification = "inconclusive"
        hard_general = [row for row in component_tasks if row["condition"] == "hard" and row["rule_type"] == "general"]
        hard_false_positive_rate = float(np.mean([row["yes"] for row in hard_general]))
        for activation_bin in contract.activation_bins:
            bin_values = [row["yes"] for row in positive_general if row["activation_bin"] == activation_bin]
            bin_rows.append(
                {
                    "component_key": key,
                    "site": selected[key]["site"],
                    "cohort": selected[key]["cohort"],
                    "activation_bin": activation_bin,
                    "positive_n": len(bin_values),
                    "general_rule_presence_rate": float(np.mean(bin_values)),
                }
            )
        component_rows.append(
            {
                "component_key": key,
                "site": selected[key]["site"],
                "cohort": selected[key]["cohort"],
                "historical_credible_multi_cluster": bool(selected[key]["historical_credible_multi_cluster"]),
                "discovery_natural_firing_events": int(selected[key]["discovery_natural_firing_events"]),
                "discovery_unique_firing_documents": int(selected[key]["discovery_unique_firing_documents"]),
                "discovery_groups": discovery_group_count,
                "covered_groups": len(covered_groups),
                "group_coverage": group_coverage,
                "minimum_general_group_recall": minimum_general_group_recall,
                "general_group_recalls": general_group_recalls,
                "general_balanced_accuracy": general_stats["balanced_accuracy"],
                "general_balanced_accuracy_ci95": general_ci,
                "general_sensitivity": general_stats["sensitivity"],
                "general_specificity": general_stats["specificity"],
                "multi_balanced_accuracy": multi_stats["balanced_accuracy"],
                "multi_balanced_accuracy_ci95": multi_ci,
                "multi_sensitivity": multi_stats["sensitivity"],
                "multi_specificity": multi_stats["specificity"],
                "multi_minus_general": advantage,
                "multi_minus_general_ci95": advantage_ci,
                "qualifying_distinct_group_rules": qualifying_group_rules,
                "group_rule_diagnostics": group_rule_diagnostics,
                "hard_negative_general_false_positive_rate": hard_false_positive_rate,
                "ordinary_positive_n": general_stats["positive_n"],
                "ordinary_negative_n": general_stats["negative_n"],
                "hard_negative_n": len(hard_general),
                "classification": classification,
                "general_rule": rules[key]["general_rule"],
                "group_rules": rules[key]["group_rules"],
            }
        )

    controls = _analyze_controls(
        contract=contract,
        heldout_dir=args.heldout_dir,
        confirmatory_input=args.confirmatory_input,
        confirmatory_embeddings=args.confirmatory_embeddings,
    )
    control_summary = {}
    for condition in sorted({row["condition"] for row in controls}):
        subset = [row for row in controls if row["condition"] == condition]
        count = sum(bool(row["credible_multi_cluster"]) for row in subset)
        control_summary[condition] = {
            "credible_multi_cluster": count,
            "components": len(subset),
            "fraction": count / len(subset),
            "ci95": _wilson(count, len(subset)),
            "mean_selected_k": float(np.mean([row["selected_k"] for row in subset])),
            "mean_silhouette": float(np.mean([row["silhouette"] for row in subset])),
        }

    historical = {}
    with args.historical_metrics.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            historical[str(row["component_key"])] = row
    intruder_rows = [row for row in component_rows if row["component_key"] in historical]
    intruder_correlation = spearmanr(
        [float(historical[row["component_key"]]["failed_judge_accuracy"]) for row in intruder_rows],
        [float(row["general_balanced_accuracy"]) for row in intruder_rows],
    )
    summaries = {}
    for label, subset in {
        "overall": component_rows,
        "pre": [row for row in component_rows if row["site"] == "pre"],
        "mid": [row for row in component_rows if row["site"] == "mid"],
        "original_h5": [row for row in component_rows if row["cohort"] == "original_h5"],
        "new_random": [row for row in component_rows if row["cohort"] == "new_random"],
        "historical_78": [row for row in component_rows if row["cohort"] == "original_h5" and row["historical_credible_multi_cluster"]],
    }.items():
        counts = Counter(str(row["classification"]) for row in subset)
        summaries[label] = {
            "components": len(subset),
            "classification_counts": dict(sorted(counts.items())),
            "classification_fraction_ci95": {
                key: {"fraction": value / len(subset), "ci95": _wilson(value, len(subset))}
                for key, value in sorted(counts.items())
            },
            "general_balanced_accuracy": _aggregate_component_mean(
                subset, "general_balanced_accuracy", replicates=contract.bootstrap_replicates, seed=contract.bootstrap_seed + len(summaries)
            ),
            "multi_balanced_accuracy": _aggregate_component_mean(
                subset, "multi_balanced_accuracy", replicates=contract.bootstrap_replicates, seed=contract.bootstrap_seed + 100 + len(summaries)
            ),
        }
    strength_summary = {}
    for activation_bin in contract.activation_bins:
        subset = [row for row in bin_rows if row["activation_bin"] == activation_bin]
        strength_summary[activation_bin] = _aggregate_component_mean(
            subset, "general_rule_presence_rate", replicates=contract.bootstrap_replicates, seed=contract.bootstrap_seed + 200 + len(strength_summary)
        )
        strength_summary[activation_bin]["positive_contexts"] = sum(int(row["positive_n"]) for row in subset)

    case_priority = {
        "single_coherent_rule": 0,
        "multiple_unrelated_rules": 1,
        "no_reliable_linguistic_rule": 2,
        "inconclusive": 3,
    }
    cases = []
    per_class_limit = {"single_coherent_rule": 3, "multiple_unrelated_rules": 3, "no_reliable_linguistic_rule": 2, "inconclusive": 2}
    for classification, limit in per_class_limit.items():
        subset = [row for row in component_rows if row["classification"] == classification]
        subset.sort(key=lambda row: (-abs(float(row["multi_minus_general"])), -float(row["general_balanced_accuracy"]), row["component_key"]))
        for row in subset[:limit]:
            key = str(row["component_key"])
            positive_tasks = [task for task in joined if task["component_key"] == key and task["rule_type"] == "general" and task["condition"] == "ordinary" and task["actual_natural_firing"]]
            representative = []
            for task in positive_tasks[:2]:
                context = str(task["messages"][1]["content"]).split("\n\nContext:\n", 1)[1].rsplit("\n\nLabel:", 1)[0]
                representative.append({"example_id": task["example_id"], "prediction": task["prediction"], "activation_bin": task["activation_bin"], "target_discovery_cluster": task["target_discovery_cluster"], "context": context})
            cases.append({"component_key": key, "classification": classification, "general_rule": row["general_rule"], "general_balanced_accuracy": row["general_balanced_accuracy"], "multi_balanced_accuracy": row["multi_balanced_accuracy"], "multi_minus_general": row["multi_minus_general"], "representative_heldout_firings": representative})
    cases.sort(key=lambda row: (case_priority[row["classification"]], row["component_key"]))

    summary = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_COHERENCE_RULE_ANALYSIS_COMPLETE",
        "human_evidence_claimed": False,
        "interpretation_scope": "Nonhuman independent held-out rule-presence validation; not human evidence and not mechanistic causation.",
        "components": summaries,
        "activation_strength": strength_summary,
        "cluster_controls": control_summary,
        "pre_mid_adjusted": _adjusted_site_difference(component_rows, replicates=contract.bootstrap_replicates, seed=contract.bootstrap_seed + 300),
        "historical_intruder_association": {
            "components": len(intruder_rows),
            "spearman_rho": float(intruder_correlation.statistic),
            "pvalue_exploratory": float(intruder_correlation.pvalue),
            "warning": "Historical Intruder values came from the failed position-biased judge and are exploratory only.",
        },
        "validator": {
            "calibration": calibration_manifest["summary"],
            "validation": validation_manifest["summary"],
            "uncertain_or_invalid_is_counted_as_rule_absent": True,
        },
    }
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "per_component_results.jsonl", component_rows)
    flat_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in component_rows
    ]
    _write_csv(args.output_dir / "per_component_results.csv", flat_rows)
    _write_jsonl(args.output_dir / "activation_strength_results.jsonl", bin_rows)
    _write_jsonl(args.output_dir / "cluster_control_results.jsonl", controls)
    _write_jsonl(args.output_dir / "case_studies.jsonl", cases)
    _write_json(args.output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "git_sha": _git_sha(),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "analysis_contract_sha256": sha256_file(args.contract),
        "tasks_manifest_sha256": sha256_file(args.tasks_dir / "preparation_manifest.json"),
        "calibration_manifest_sha256": sha256_file(args.calibration_dir / "experiment_manifest.json"),
        "validation_manifest_sha256": sha256_file(args.validation_dir / "experiment_manifest.json"),
        "rules_manifest_sha256": sha256_file(args.rules_dir / "freeze_manifest.json"),
        "heldout_manifest_sha256": sha256_file(args.heldout_dir / "preparation_manifest.json"),
        "confirmatory_input_sha256": sha256_file(args.confirmatory_input),
        "confirmatory_embeddings_sha256": sha256_file(args.confirmatory_embeddings),
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("per_component_results.jsonl", "per_component_results.csv", "activation_strength_results.jsonl", "cluster_control_results.jsonl", "case_studies.jsonl", "summary.json")
        },
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)
    (args.output_dir / "COMPLETE").write_text(summary["status"] + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

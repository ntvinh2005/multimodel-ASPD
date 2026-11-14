"""Freeze CPU diagnostics, clustering inputs and blinded human tasks for ASPD v3.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--sparse-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any, *, protected: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    if protected:
        path.chmod(0o600)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _component_key(site: str, feature: int) -> str:
    return f"{site}/{feature:04d}"


def _document_is_calibration(document_id: str, *, modulus: int, bucket: int) -> bool:
    return int(hashlib.sha256(document_id.encode()).hexdigest(), 16) % modulus == bucket


def threshold_coverage(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    anchor_documents: np.ndarray,
    fixed_thresholds: list[float],
    quantiles: list[float],
    modulus: int,
    bucket: int,
    minimum_events: int,
    n_features: int,
) -> dict[str, Any]:
    calibration_anchor = np.asarray(
        [_document_is_calibration(str(value), modulus=modulus, bucket=bucket) for value in anchor_documents]
    )
    event_anchor = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    calibration_event = calibration_anchor[event_anchor]
    test_event = ~calibration_event
    test_indices = indices[test_event].astype(np.int64, copy=False)
    test_values = values[test_event]
    test_documents = anchor_documents[event_anchor[test_event]]

    def summarize(mask: np.ndarray, label: str) -> dict[str, Any]:
        retained_indices = test_indices[mask]
        counts = np.bincount(retained_indices, minlength=n_features)
        return {
            "label": label,
            "retained_events": int(mask.sum()),
            "retained_event_fraction": float(mask.mean()),
            "features_with_any_event": int(np.count_nonzero(counts)),
            "eligible_features": int(np.count_nonzero(counts >= minimum_events)),
            "features_lost_from_baseline": int(n_features - np.count_nonzero(counts)),
            "retained_documents": int(np.unique(test_documents[mask]).size),
            "firing_count_quantiles": [
                float(value) for value in np.quantile(counts, [0, 0.5, 0.9, 0.99, 1])
            ],
        }

    rows = [summarize(np.ones(len(test_values), dtype=bool), "natural_gs_gt_0")]
    for threshold in fixed_thresholds:
        rows.append(summarize(test_values >= threshold, f"fixed_gs_ge_{threshold:g}"))

    calibration_indices = indices[calibration_event].astype(np.int64, copy=False)
    calibration_values = values[calibration_event]
    thresholds = {quantile: np.full(n_features, np.inf, dtype=np.float32) for quantile in quantiles}
    order = np.argsort(calibration_indices, kind="stable")
    ordered_features = calibration_indices[order]
    starts = np.flatnonzero(np.r_[True, ordered_features[1:] != ordered_features[:-1]])
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops, strict=True):
        feature = int(ordered_features[start])
        feature_values = calibration_values[order[start:stop]]
        for quantile in quantiles:
            thresholds[quantile][feature] = np.quantile(feature_values, quantile)
    for quantile in quantiles:
        cutoff = thresholds[quantile][test_indices]
        rows.append(summarize(test_values >= cutoff, f"calibration_within_feature_q{quantile:g}"))
    return {
        "calibration_anchors": int(calibration_anchor.sum()),
        "test_anchors": int((~calibration_anchor).sum()),
        "calibration_events": int(calibration_event.sum()),
        "test_events": int(test_event.sum()),
        "conditions": rows,
    }


def _judge_component_scores(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["component_key"])].append(row)
    return {
        key: {
            "correct": int(sum(bool(row["correct"]) for row in values)),
            "valid": int(sum(bool(row["valid"]) for row in values)),
            "accuracy": float(np.mean([bool(row["correct"]) for row in values])),
        }
        for key, values in grouped.items()
    }


def activation_judge_relationship(
    answers: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    activation_by_component: dict[str, list[float]] = defaultdict(list)
    donor_density_gap: dict[str, list[float]] = defaultdict(list)
    source_match: dict[str, list[float]] = defaultdict(list)
    for row in answers:
        key = str(row["component_key"])
        activation_by_component[key].extend(float(value["activation"]) for value in row["target_contexts"])
        donor_density_gap[key].append(
            abs(float(row["target_firing_density"]) - float(row["donor_firing_density"]))
        )
        target_groups = {str(value["source_group"]) for value in row["target_contexts"]}
        source_match[key].append(float(str(row["donor_context"]["source_group"]) in target_groups))
    judge = _judge_component_scores(results)
    rows = []
    for key in sorted(judge):
        rows.append(
            {
                "component_key": key,
                **judge[key],
                "mean_target_activation": float(np.mean(activation_by_component[key])),
                "median_target_activation": float(np.median(activation_by_component[key])),
                "mean_donor_density_gap": float(np.mean(donor_density_gap[key])),
                "donor_source_group_match_fraction": float(np.mean(source_match[key])),
            }
        )
    report = {"components": len(rows), "trials": len(results)}
    for field in (
        "mean_target_activation",
        "median_target_activation",
        "mean_donor_density_gap",
        "donor_source_group_match_fraction",
    ):
        statistic = spearmanr(
            [row[field] for row in rows], [row["accuracy"] for row in rows]
        )
        report[f"spearman_{field}_vs_failed_judge_accuracy"] = {
            "rho": float(statistic.statistic),
            "pvalue_exploratory": float(statistic.pvalue),
        }
    report["measurement_warning"] = (
        "The Llama judge failed position reliability; associations are exploratory and are not "
        "semantic-coherence estimates."
    )
    return {"summary": report, "rows": rows}


def _frequency_strata(counts: np.ndarray, eligible: set[int]) -> dict[int, str]:
    values = np.asarray([counts[index] for index in sorted(eligible)], dtype=float)
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    return {
        index: "low" if counts[index] <= low else "medium" if counts[index] <= high else "high"
        for index in eligible
    }


def choose_clustering_keys(
    *,
    context_counts: dict[str, int],
    firing_counts: dict[str, np.ndarray],
    per_stratum: int,
    minimum_contexts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    strata_bounds: dict[str, Any] = {}
    for site in ("pre", "mid"):
        candidates = {
            int(key.split("/")[1])
            for key, count in context_counts.items()
            if key.startswith(f"{site}/") and count >= minimum_contexts
        }
        strata = _frequency_strata(firing_counts[site], candidates)
        for stratum in ("low", "medium", "high"):
            group = sorted(index for index, value in strata.items() if value == stratum)
            if len(group) < per_stratum:
                raise ValueError(f"insufficient {site}/{stratum} clustering candidates")
            chosen = sorted(rng.choice(group, size=per_stratum, replace=False).astype(int).tolist())
            selected.extend(
                {
                    "component_key": _component_key(site, feature),
                    "site": site,
                    "feature": feature,
                    "frequency_stratum": stratum,
                    "firing_count": int(firing_counts[site][feature]),
                    "available_contexts": int(context_counts[_component_key(site, feature)]),
                }
                for feature in chosen
            )
            strata_bounds[f"{site}.{stratum}"] = {
                "candidate_count": len(group),
                "selected_count": len(chosen),
            }
    return selected, strata_bounds


def _human_package(
    *,
    tasks: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    judge_scores: dict[str, dict[str, Any]],
    count: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    task_by_id = {str(row["task_id"]): row for row in tasks}
    answers_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answers:
        answers_by_component[str(row["component_key"])].append(row)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for component, score in judge_scores.items():
        band = "zero" if score["correct"] == 0 else "high" if score["correct"] >= 6 else "intermediate"
        groups[(component.split("/")[0], band)].append(component)
    if count % 2:
        raise ValueError("human-package count must be even for equal-site sampling")
    per_site = count // 2
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for site in ("pre", "mid"):
        zero = sorted(groups[(site, "zero")])
        if len(zero) >= per_site:
            site_chosen = rng.choice(zero, size=per_site, replace=False).tolist()
        else:
            remaining = per_site - len(zero)
            intermediate_count = remaining // 2
            high_count = remaining - intermediate_count
            intermediate = sorted(groups[(site, "intermediate")])
            high = sorted(groups[(site, "high")])
            if len(intermediate) < intermediate_count or len(high) < high_count:
                raise ValueError(f"insufficient human-package candidates for {site}")
            site_chosen = [
                *zero,
                *rng.choice(intermediate, size=intermediate_count, replace=False).tolist(),
                *rng.choice(high, size=high_count, replace=False).tolist(),
            ]
        chosen.extend(site_chosen)
    rng.shuffle(chosen)
    public_rows = []
    protected_rows = []
    for index, component in enumerate(chosen, start=1):
        candidates = sorted(answers_by_component[component], key=lambda row: int(row["trial_index"]))
        desired = (index - 1) % 5 + 1
        answer = min(candidates, key=lambda row: (abs(int(row["correct_position"]) - desired), int(row["trial_index"])))
        task = task_by_id[str(answer["task_id"])]
        item_id = f"Feature Set {index:03d}"
        public_rows.append({"item_id": item_id, "question": task["question"]})
        protected_rows.append(
            {
                "item_id": item_id,
                "component_key": component,
                "task_id": answer["task_id"],
                "correct_position": int(answer["correct_position"]),
                "failed_judge_score_band": (
                    "zero" if judge_scores[component]["correct"] == 0
                    else "high" if judge_scores[component]["correct"] >= 6 else "intermediate"
                ),
            }
        )
    dataset_hash = hashlib.sha256(
        json.dumps(public_rows, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    response = {
        "schema_version": 1,
        "dataset_sha256": dataset_hash,
        "evaluator_id": "",
        "responses": {
            row["item_id"]: {
                "intruder_choice": None,
                "unique_intruder": None,
                "coherence_0_to_4": None,
                "pattern_category": None,
                "confidence_0_to_3": None,
                "explanation": "",
            }
            for row in public_rows
        },
    }
    _write_json(output / "human_tasks.json", public_rows)
    _write_json(output / "human_answer_key.json", protected_rows, protected=True)
    _write_json(output / "human_response_template.json", response)
    cards = "\n".join(
        f"<section><h2>{html.escape(row['item_id'])}</h2><pre>{html.escape(row['question'])}</pre></section>"
        for row in public_rows
    )
    (output / "human_review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>ASPD v3.1 blinded review</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:auto;padding:24px}"
        "section{border:1px solid #bbb;padding:16px;margin:20px 0}pre{white-space:pre-wrap}</style>"
        "<h1>ASPD v3.1 blinded human review</h1><p>For each item record intruder choice "
        "or abstain, whether the answer is unique, coherence 0-4, category "
        "(semantic/syntactic/formatting/token-boundary/mixed/none/insufficient), confidence "
        "0-3, and a short explanation in the response template.</p>" + cards,
        encoding="utf-8",
    )
    return {"items": len(public_rows), "dataset_sha256": dataset_hash, "required_raters": 2}


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("training_authorized_by_this_contract") is not False:
        raise ValueError("Stage 2 preparation must not authorize training")
    if protocol.get("forbidden_model_substrings") != ["qwen"]:
        raise ValueError("Stage 2 model exclusion changed")
    extraction_manifest = json.loads(
        (args.extraction_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if extraction_manifest["checkpoint_manifest_sha256"] != protocol["checkpoint_manifest_sha256"]:
        raise ValueError("Stage 1 extraction and Stage 2 checkpoint differ")
    args.output_dir.mkdir(parents=True)

    documents = json.loads((args.sparse_dir / "documents.json").read_text(encoding="utf-8"))
    document_ids = np.asarray([str(row["document_id"]) for row in documents], dtype=object)
    with np.load(args.sparse_dir / "anchors.npz") as anchors:
        anchor_documents = document_ids[anchors["document_indices"].astype(np.int64, copy=False)]

    firing_counts: dict[str, np.ndarray] = {}
    threshold_reports = {}
    for site in ("pre", "mid"):
        with np.load(args.sparse_dir / f"{site}_natural_firings.npz") as artifact:
            indptr = artifact["indptr"]
            indices = artifact["indices"]
            values = artifact["values"]
            shape = artifact["shape"]
            firing_counts[site] = np.bincount(indices.astype(np.int64), minlength=int(shape[1]))
            threshold_reports[site] = threshold_coverage(
                indptr=indptr,
                indices=indices,
                values=values,
                anchor_documents=anchor_documents,
                fixed_thresholds=[float(value) for value in protocol["activation_thresholds"]],
                quantiles=[float(value) for value in protocol["within_feature_quantiles"]],
                modulus=int(protocol["calibration_document_hash_modulus"]),
                bucket=int(protocol["calibration_document_hash_bucket"]),
                minimum_events=int(protocol["minimum_retained_events"]),
                n_features=int(shape[1]),
            )
    _write_json(args.output_dir / "activation_threshold_coverage.json", threshold_reports)

    tasks_dir = args.extraction_dir / "frozen_tasks"
    paper_tasks = _jsonl(tasks_dir / "paper_tasks.jsonl")
    paper_answers = _jsonl(tasks_dir / "paper_answer_key.jsonl")
    judge_results = _jsonl(args.judge_dir / "results.jsonl")
    relationship = activation_judge_relationship(paper_answers, judge_results)
    _write_json(args.output_dir / "activation_failed_judge_relationship.json", relationship["summary"])
    with (args.output_dir / "component_observations.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(relationship["rows"][0]))
        writer.writeheader()
        writer.writerows(relationship["rows"])

    context_counts: dict[str, int] = {}
    with (args.extraction_dir / "component_contexts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            context_counts[str(row["component_key"])] = len(row["contexts"])
    selected, strata = choose_clustering_keys(
        context_counts=context_counts,
        firing_counts=firing_counts,
        per_stratum=int(protocol["clustering_features_per_site_frequency_stratum"]),
        minimum_contexts=int(protocol["clustering_minimum_contexts"]),
        seed=int(protocol["clustering_seed"]),
    )
    selected_by_key = {row["component_key"]: row for row in selected}
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    embedding_rows: list[dict[str, Any]] = []
    with (args.extraction_dir / "component_contexts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["component_key"])
            if key not in selected_by_key:
                continue
            contexts = sorted(
                row["contexts"], key=lambda value: (-float(value["activation"]), str(value["anchor_id"]))
            )[: int(protocol["clustering_contexts_per_feature"])]
            for context_index, context in enumerate(contexts):
                embedding_rows.append(
                    {
                        **selected_by_key[key],
                        "context_index": context_index,
                        "activation": float(context["activation"]),
                        "anchor_id": str(context["anchor_id"]),
                        "document_id": str(context["document_id"]),
                        "source_group": str(context["source_group"]),
                        "text": tokenizer.decode(
                            context["token_ids"],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).strip(),
                    }
                )
    _write_jsonl(args.output_dir / "clustering_embedding_input.jsonl", embedding_rows)
    _write_json(
        args.output_dir / "clustering_sample_manifest.json",
        {"selected": selected, "strata": strata, "contexts": len(embedding_rows)},
    )
    human = _human_package(
        tasks=paper_tasks,
        answers=paper_answers,
        judge_scores=_judge_component_scores(judge_results),
        count=int(protocol["human_components"]),
        seed=int(protocol["human_sampling_seed"]),
        output=args.output_dir,
    )
    inputs = [
        args.protocol,
        args.extraction_dir / "experiment_manifest.json",
        args.extraction_dir / "component_contexts.jsonl",
        args.sparse_dir / "pre_natural_firings.npz",
        args.sparse_dir / "mid_natural_firings.npz",
        args.judge_dir / "results.jsonl",
    ]
    _write_json(
        args.output_dir / "preparation_manifest.json",
        {
            "schema_version": 1,
            "status": "ASPD_V3_1_INTERPRETABILITY_STAGE2_PREPARED",
            "git_sha": _git_sha(),
            "protocol_sha256": sha256_file(args.protocol),
            "checkpoint_manifest_sha256": protocol["checkpoint_manifest_sha256"],
            "input_sha256": {str(path): sha256_file(path) for path in inputs},
            "natural_gate_only": True,
            "forced_activations": 0,
            "clustering_features": len(selected),
            "clustering_contexts": len(embedding_rows),
            "human_package": human,
            "human_results_status": "PENDING_TWO_INDEPENDENT_RATERS",
            "judge_semantic_status": "INVALID_FOR_CONFIRMATORY_COHERENCE",
        },
    )
    (args.output_dir / "PREPARED").write_text("ASPD_V3_1_INTERPRETABILITY_STAGE2_PREPARED\n")


if __name__ == "__main__":
    main()

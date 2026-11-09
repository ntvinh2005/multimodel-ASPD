"""Grade completed blinded human ASPD evaluations after response submission."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.human_interpretability import (
    bootstrap_group,
    canonical_json_bytes,
    pairwise_rater_agreement,
    sha256_file,
    summarize_rows,
    validate_responses,
)
from aspd.multimodel.v3.mechanistic_validation import HumanInterpretabilityContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--public-dataset", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--answer-key-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


def _scored_rows(
    evaluator_id: str,
    responses: dict[str, dict[str, Any]],
    answers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for answer in answers:
        item_id = str(answer["item_id"])
        response = responses[item_id]
        choice = response["intruder_choice"]
        selected_context = None
        if choice is not None:
            selected_context = answer["displayed_contexts"][int(choice) - 1]
        rows.append(
            {
                "evaluator_id": evaluator_id,
                "item_id": item_id,
                "parent_uid": answer["parent_uid"],
                "repeat_of_parent_uid": answer.get("repeat_of_parent_uid"),
                "arm": answer["arm"],
                "site": answer["site"],
                "feature": int(answer["feature"]),
                "frequency_stratum": answer["frequency_stratum"],
                "feature_fires": int(answer["feature_fires"]),
                "abstained": bool(response["cannot_determine"]),
                "selected_canonical_index": (
                    None if selected_context is None else selected_context["anchor_id"]
                ),
                "correct": bool(
                    not response["cannot_determine"]
                    and int(choice) == int(answer["intruder_position"])
                ),
                "coherence": int(response["coherence"]),
                "pattern_category": response["pattern_category"],
                "confidence": int(response["confidence"]),
                "explanation": response["explanation"],
            }
        )
    return rows


def _within_rater(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = {
        str(row["parent_uid"]): row
        for row in rows
        if not row.get("repeat_of_parent_uid")
    }
    repeats = [row for row in rows if row.get("repeat_of_parent_uid")]
    comparisons = []
    for repeat in repeats:
        original = primary[str(repeat["repeat_of_parent_uid"])]
        comparisons.append(
            {
                "parent_uid": repeat["parent_uid"],
                "intruder_choice_consistent": (
                    original["abstained"] == repeat["abstained"]
                    and original["selected_canonical_index"] == repeat["selected_canonical_index"]
                ),
                "coherence_difference": abs(original["coherence"] - repeat["coherence"]),
                "category_consistent": original["pattern_category"] == repeat["pattern_category"],
                "abstention_consistent": original["abstained"] == repeat["abstained"],
            }
        )
    return {
        "repeated_items": len(comparisons),
        "intruder_choice_consistency": (
            float(np.mean([row["intruder_choice_consistent"] for row in comparisons]))
            if comparisons else None
        ),
        "mean_absolute_coherence_difference": (
            float(np.mean([row["coherence_difference"] for row in comparisons]))
            if comparisons else None
        ),
        "category_consistency": (
            float(np.mean([row["category_consistent"] for row in comparisons]))
            if comparisons else None
        ),
        "abstention_consistency": (
            float(np.mean([row["abstention_consistent"] for row in comparisons]))
            if comparisons else None
        ),
        "comparisons": comparisons,
    }


def _group_results(
    rows: list[dict[str, Any]], contract: HumanInterpretabilityContract, seed_offset: int
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['arm']}.{row['site']}"].append(row)
    result: dict[str, Any] = {}
    draws: dict[str, dict[str, list[float]]] = {}
    for index, key in enumerate(sorted(grouped)):
        summary = summarize_rows(grouped[key])
        boot = bootstrap_group(
            grouped[key],
            replicates=contract.bootstrap_replicates,
            seed=contract.bootstrap_seed + seed_offset + index,
        )
        draws[key] = {
            "accuracy": boot.pop("intruder_accuracy_draws"),
            "coherence": boot.pop("coherence_mean_draws"),
        }
        summary.update(boot)
        summary["frequency_strata"] = {
            stratum: summarize_rows(
                [row for row in grouped[key] if row["frequency_stratum"] == stratum]
            )
            for stratum in contract.frequency_strata
        }
        result[key] = summary
    return result, draws


def _treatment_effects(draws: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for site in ("pre", "mid"):
        control = draws[f"full_only_control.{site}"]
        treatment = draws[f"per_anchor_zscore.{site}"]
        accuracy = np.asarray(treatment["accuracy"]) - np.asarray(control["accuracy"])
        coherence = np.asarray(treatment["coherence"]) - np.asarray(control["coherence"])
        effects[site] = {
            "intruder_accuracy_difference_mean": float(accuracy.mean()),
            "intruder_accuracy_difference_ci95": np.quantile(accuracy, [0.025, 0.975]).tolist(),
            "coherence_difference_mean": float(coherence.mean()),
            "coherence_difference_ci95": np.quantile(coherence, [0.025, 0.975]).tolist(),
        }
    return effects


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASPD v3.1 human interpretability results",
        "",
        f"Evaluators: {len(result['evaluators'])}",
        "",
        "These results are exploratory when only one human evaluator participated.",
        "Abstentions count as incorrect in intention-to-evaluate Intruder accuracy.",
        "",
    ]
    for evaluator, payload in result["evaluators"].items():
        lines.extend([f"## Evaluator `{evaluator}`", ""])
        for key, group in payload["groups"].items():
            lines.append(
                f"- `{key}`: n={group['features']}, accuracy={group['intruder_accuracy_intention_to_evaluate']:.3f}, "
                f"abstention={group['abstention_rate']:.3f}, coherence={group['coherence_mean']:.3f}."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = HumanInterpretabilityContract.load(args.protocol)
    dataset = _json(args.public_dataset)
    dataset_sha256 = sha256_file(args.public_dataset)
    public_ids = {str(item["item_id"]) for item in dataset["items"]}
    answer_manifest = _json(args.answer_key_manifest)
    if answer_manifest["dataset_sha256"] != dataset_sha256:
        raise ValueError("answer key belongs to a different public dataset")
    if answer_manifest["answer_key_sha256"] != sha256_file(args.answer_key):
        raise ValueError("answer key hash changed")
    answers = _jsonl(args.answer_key)
    answer_ids = {str(row["item_id"]) for row in answers}
    scientific_ids = {
        str(item["item_id"]) for item in dataset["items"] if item["phase"] == "scientific"
    }
    if answer_ids != scientific_ids:
        raise ValueError("answer key and public scientific task IDs disagree")

    args.output_dir.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    evaluators: dict[str, Any] = {}
    rater_rows: dict[str, list[dict[str, Any]]] = {}
    for index, response_path in enumerate(args.responses):
        payload = _json(response_path)
        evaluator_id = str(payload.get("evaluator_id", "")).strip()
        if evaluator_id in evaluators:
            raise ValueError(f"duplicate evaluator_id: {evaluator_id}")
        responses = validate_responses(
            payload, dataset_sha256=dataset_sha256, public_item_ids=public_ids
        )
        rows = _scored_rows(evaluator_id, responses, answers)
        groups, draws = _group_results(rows, contract, seed_offset=index * 10_000)
        rater_rows[evaluator_id] = rows
        all_rows.extend(rows)
        evaluators[evaluator_id] = {
            "response_file": str(response_path),
            "response_sha256": sha256_file(response_path),
            "groups": groups,
            "treatment_effects": _treatment_effects(draws),
            "within_rater_repeat_consistency": _within_rater(rows),
        }

    between_raters = []
    for left, right in itertools.combinations(sorted(rater_rows), 2):
        between_raters.append(
            {
                "left": left,
                "right": right,
                **pairwise_rater_agreement(rater_rows[left], rater_rows[right]),
            }
        )
    result = {
        "schema_version": 1,
        "status": "HUMAN_INTERPRETABILITY_ANALYSIS_COMPLETE",
        "dataset_sha256": dataset_sha256,
        "answer_key_sha256": sha256_file(args.answer_key),
        "single_rater_limit": len(evaluators) == 1,
        "evaluators": evaluators,
        "between_rater_agreement": between_raters,
    }
    _write_json(args.output_dir / "analysis.json", result)
    with (args.output_dir / "scored_responses.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

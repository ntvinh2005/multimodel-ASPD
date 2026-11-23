"""Analyze the explicitly nonhuman blinded H4 qualitative audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row["attempted"]]
    unique = [row for row in rows if row["agent_unique_intruder"]]
    coherent = [row for row in rows if row["agent_coherent"]]
    non_topical = [row for row in rows if row["agent_non_topical_coherent"]]
    unique_non_topical = [
        row for row in non_topical if row["agent_unique_intruder"] and row["attempted"]
    ]
    return {
        "items": len(rows),
        "attempted": len(attempted),
        "agent_unique_intruder_items": len(unique),
        "agent_unique_intruder_fraction": _mean(
            [float(row["agent_unique_intruder"]) for row in rows]
        ),
        "agent_coherent_items": len(coherent),
        "agent_coherent_fraction": _mean([float(row["agent_coherent"]) for row in rows]),
        "agent_non_topical_coherent_items": len(non_topical),
        "agent_non_topical_coherent_fraction": _mean(
            [float(row["agent_non_topical_coherent"]) for row in rows]
        ),
        "mean_coherence": _mean([float(row["coherence_0_to_4"]) for row in rows]),
        "intruder_accuracy_intention_to_evaluate": _mean(
            [float(row["correct"]) for row in rows]
        ),
        "intruder_accuracy_attempted": _mean([float(row["correct"]) for row in attempted]),
        "intruder_accuracy_agent_unique": _mean([float(row["correct"]) for row in unique]),
        "unique_non_topical_attempts": len(unique_non_topical),
        "intruder_accuracy_unique_non_topical": _mean(
            [float(row["correct"]) for row in unique_non_topical]
        ),
        "category_counts": dict(sorted(Counter(row["pattern_category"] for row in rows).items())),
        "coherent_category_counts": dict(
            sorted(Counter(row["pattern_category"] for row in coherent).items())
        ),
    }


def analyze(
    tasks: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    response_payload: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_sha = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if dataset_sha != contract["dataset_sha256"]:
        raise ValueError("public task dataset changed")
    if response_payload["dataset_sha256"] != dataset_sha:
        raise ValueError("responses belong to a different dataset")
    if response_payload["evaluator_provenance"].get("human_evaluator") is not False:
        raise ValueError("H4 agent audit must not claim a human evaluator")
    task_ids = {str(row["item_id"]) for row in tasks}
    answer_index = {str(row["item_id"]): row for row in answers}
    responses = response_payload["responses"]
    if task_ids != set(answer_index) or task_ids != set(responses):
        raise ValueError("task, answer, and response IDs differ")
    excluded = set(contract["contaminated_items"])
    allowed_categories = {
        "semantic",
        "syntactic",
        "formatting",
        "token-boundary",
        "mixed",
        "none",
        "insufficient",
    }
    non_topical_categories = set(contract["non_topical_categories"])
    coherent_minimum = int(contract["coherent_minimum_score"])
    rows = []
    for item_id in sorted(task_ids):
        if item_id in excluded:
            continue
        answer = answer_index[item_id]
        response = responses[item_id]
        category = str(response["pattern_category"])
        if category not in allowed_categories:
            raise ValueError(f"invalid category for {item_id}: {category}")
        choice = response["intruder_choice"]
        attempted = choice is not None
        coherence = int(response["coherence_0_to_4"])
        if not 0 <= coherence <= 4:
            raise ValueError(f"invalid coherence for {item_id}")
        component = str(answer["component_key"])
        agent_coherent = coherence >= coherent_minimum
        rows.append(
            {
                "item_id": item_id,
                "site": component.split("/", maxsplit=1)[0],
                "component_key": component,
                "failed_judge_score_band": str(answer["failed_judge_score_band"]),
                "correct_position": int(answer["correct_position"]),
                "intruder_choice": choice,
                "attempted": attempted,
                "correct": bool(attempted and int(choice) == int(answer["correct_position"])),
                "agent_unique_intruder": bool(response["unique_intruder"]),
                "coherence_0_to_4": coherence,
                "agent_coherent": agent_coherent,
                "pattern_category": category,
                "agent_non_topical_coherent": bool(
                    agent_coherent and category in non_topical_categories
                ),
                "confidence_0_to_3": int(response["confidence_0_to_3"]),
                "explanation": str(response["explanation"]),
            }
        )
    expected = len(tasks) - len(excluded)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} uncontaminated rows, received {len(rows)}")

    grouped: dict[str, dict[str, Any]] = {}
    for field in ("site", "failed_judge_score_band"):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values[str(row[field])].append(row)
        grouped[field] = {key: _summarize(value) for key, value in sorted(values.items())}
    overall = _summarize(rows)
    rule = contract["exploratory_support_rule"]
    checks = {
        "non_topical_coherent_fraction": bool(
            overall["agent_non_topical_coherent_fraction"]
            >= float(rule["minimum_non_topical_coherent_fraction"])
        ),
        "unique_non_topical_attempts": bool(
            overall["unique_non_topical_attempts"]
            >= int(rule["minimum_unique_non_topical_attempts"])
        ),
        "unique_non_topical_accuracy": bool(
            overall["intruder_accuracy_unique_non_topical"] is not None
            and overall["intruder_accuracy_unique_non_topical"]
            >= float(rule["minimum_intruder_accuracy_on_unique_non_topical"])
        ),
    }
    passed = all(checks.values())
    verdict = (
        contract["adjudication"]["supported_if_rule_passes"]
        if passed
        else contract["adjudication"]["otherwise"]
    )
    return (
        {
            "schema_version": 1,
            "status": "H4_AGENT_AUDIT_ANALYSIS_COMPLETE",
            "scientific_status": contract["scientific_status"],
            "human_evidence_claimed": False,
            "independent_evaluators": 1,
            "excluded_contaminated_items": sorted(excluded),
            "overall": overall,
            "groups": grouped,
            "exploratory_support_rule": {
                "thresholds": rule,
                "checks": checks,
                "passed": passed,
            },
            "h4_verdict": verdict,
            "h4_verdict_boundary": (
                "PARTIALLY_SUPPORTED is the maximum because this is a single nonhuman "
                "qualitative audit without independent inter-rater agreement"
            ),
        },
        rows,
    )


def _markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    return "\n".join(
        [
            "# ASPD v3.1 H4 agent audit",
            "",
            "Status: exploratory nonhuman qualitative audit; this is not human evidence.",
            "",
            f"- H4 verdict: `{summary['h4_verdict']}`.",
            f"- Uncontaminated items: {overall['items']}.",
            f"- Agent-coherent fraction: {overall['agent_coherent_fraction']:.3f}.",
            (
                "- Agent non-topical coherent fraction: "
                f"{overall['agent_non_topical_coherent_fraction']:.3f}."
            ),
            f"- Agent unique-intruder fraction: {overall['agent_unique_intruder_fraction']:.3f}.",
            (
                "- Intruder accuracy conditional on agent-unique non-topical tasks: "
                f"{overall['intruder_accuracy_unique_non_topical']:.3f} "
                f"(n={overall['unique_non_topical_attempts']})."
            ),
            "",
            summary["h4_verdict_boundary"],
            "",
        ]
    )


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    answers = json.loads(args.answer_key.read_text(encoding="utf-8"))
    responses = json.loads(args.responses.read_text(encoding="utf-8"))
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    summary, rows = analyze(tasks, answers, responses, contract)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "scored_responses.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_markdown(summary), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = list(summary["overall"]["category_counts"])
    counts = [summary["overall"]["category_counts"][key] for key in categories]
    coherent = [
        summary["overall"]["coherent_category_counts"].get(key, 0) for key in categories
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar(categories, counts, label="all uncontaminated items", color="lightgrey")
    axis.bar(categories, coherent, label="agent coherence >= 3", color="tab:purple")
    axis.set(
        ylabel="Items",
        title="H4 exploratory nonhuman taxonomy audit (not human evidence)",
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    figure.savefig(args.output_dir / "agent_h4_taxonomy.png", dpi=180)
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "status": "H4_AGENT_AUDIT_ANALYSIS_COMPLETE",
        "git_sha": _git_sha(),
        "human_evidence_claimed": False,
        "sources": {
            str(path): sha256_file(path)
            for path in (args.tasks, args.answer_key, args.responses, args.contract)
        },
        "summary_sha256": sha256_file(args.output_dir / "summary.json"),
        "scored_responses_sha256": sha256_file(args.output_dir / "scored_responses.csv"),
        "figure_sha256": sha256_file(args.output_dir / "agent_h4_taxonomy.png"),
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETE").write_text(
        "H4_AGENT_AUDIT_ANALYSIS_COMPLETE\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

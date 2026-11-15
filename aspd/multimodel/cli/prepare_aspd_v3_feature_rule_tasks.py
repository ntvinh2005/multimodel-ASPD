"""Freeze discovery-only candidate-rule prompts for the independent proposer."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract

SYSTEM_PROMPT = """You are proposing falsifiable linguistic rules for a sparse feature.
You see only discovery examples where the feature fired, grouped by an exploratory
embedding pipeline. Groups may reflect topic rather than different feature meanings.
Consider semantic, syntactic, formatting, structural, positional, and token-boundary
patterns. A useful rule must be specific enough to decide on unseen firing and matched
non-firing contexts. Do not use vague rules such as 'important information'. Do not claim
that embedding groups prove polysemanticity. Return only valid JSON matching the schema."""


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cluster-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _contexts(group: dict[str, Any], maximum: int) -> str:
    lines = []
    for index, row in enumerate(group["representative_contexts"][:maximum], start=1):
        lines.append(
            f"Example {index} (source={row['source_group']}, activation={row['activation']:.5f}):\n"
            f"{row['text']}"
        )
    return "\n\n".join(lines)


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    cluster_manifest = json.loads(
        (args.cluster_dir / "experiment_manifest.json").read_text()
    )
    if cluster_manifest["status"] != (
        "ASPD_V3_1_FEATURE_COHERENCE_DISCOVERY_CLUSTERING_COMPLETE"
    ):
        raise ValueError("discovery clusters are incomplete")
    components = _jsonl(args.cluster_dir / "component_clusters.jsonl")
    if len(components) != 160:
        raise ValueError("rule prompts require 160 discovery components")
    tasks = []
    for component in components:
        key = str(component["component_key"])
        group_blocks = []
        for group in component["groups"]:
            group_blocks.append(
                f"=== GROUP {group['cluster']} (discovery size={group['size']}) ===\n"
                + _contexts(group, contract.maximum_representatives_per_group)
            )
        global_schema = {
            "single_rule": "one observable rule for all groups, at most 25 words",
            "single_rule_category": "semantic|syntactic|formatting|structural|positional|token_boundary|mixed|no_clear_rule",
            "single_rule_specificity": "observable evidence, at most 25 words",
            "multiple_rule_explanation": [
                {
                    "group": 0,
                    "rule": "group-specific rule, at most 20 words",
                    "unrelated_to_other_groups": False,
                }
            ],
            "groups_share_broader_rule": "yes|no|uncertain",
            "uncertainty": "limitation, at most 20 words",
        }
        tasks.append(
            {
                "task_id": f"{key}:global",
                "component_key": key,
                "task_type": "global",
                "source_cluster": None,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Propose two competing explanations: (1) one informative rule "
                            "that predicts firing across every group, and (2) separate rules "
                            "if the groups appear genuinely unrelated. Do not inspect or assume "
                            "anything about held-out contexts.\n\n"
                            + "\n\n".join(group_blocks)
                            + "\n\nReturn exactly this JSON schema:\n"
                            + json.dumps(global_schema, ensure_ascii=False)
                            + "\nInclude exactly one multiple_rule_explanation item per "
                            "displayed group. Keep the entire JSON response under 220 tokens."
                        ),
                    },
                ],
            }
        )
        for group in component["groups"]:
            group_schema = {
                "rule": "observable rule learned only from this group, at most 25 words",
                "category": "semantic|syntactic|formatting|structural|positional|token_boundary|mixed|no_clear_rule",
                "specificity": "observable evidence, at most 25 words",
                "uncertainty": "limitation, at most 20 words",
            }
            tasks.append(
                {
                    "task_id": f"{key}:group:{group['cluster']}",
                    "component_key": key,
                    "task_type": "group_only",
                    "source_cluster": int(group["cluster"]),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Using only the contexts below from one discovery group, "
                                "propose one specific rule. The rule will be frozen and tested "
                                "on unseen contexts from this and other groups.\n\n"
                                + _contexts(
                                    group, contract.maximum_representatives_per_group
                                )
                                + "\n\nReturn exactly this JSON schema:\n"
                                + json.dumps(group_schema, ensure_ascii=False)
                                + "\nKeep the entire JSON response under 140 tokens."
                            ),
                        },
                    ],
                }
            )
    args.output_dir.mkdir(parents=True)
    tasks_path = args.output_dir / "rule_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_RULE_TASKS_FROZEN",
        "git_sha": _git_sha(),
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "cluster_manifest_sha256": sha256_file(
            args.cluster_dir / "experiment_manifest.json"
        ),
        "tasks": len(tasks),
        "global_tasks": sum(row["task_type"] == "global" for row in tasks),
        "group_only_tasks": sum(row["task_type"] == "group_only" for row in tasks),
        "components": len(components),
        "heldout_contexts_exposed_to_proposer": 0,
        "tasks_sha256": sha256_file(tasks_path),
    }
    _write_json(args.output_dir / "preparation_manifest.json", manifest)
    (args.output_dir / "FROZEN").write_text(
        "ASPD_V3_1_FEATURE_RULE_TASKS_FROZEN\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

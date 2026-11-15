"""Merge valid base and bounded-repair proposals into frozen candidate rules."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--repair-dir", type=Path, required=True)
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


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    task_manifest = json.loads(
        (args.tasks_dir / "preparation_manifest.json").read_text()
    )
    base_manifest = json.loads((args.base_dir / "experiment_manifest.json").read_text())
    repair_manifest = json.loads(
        (args.repair_dir / "experiment_manifest.json").read_text()
    )
    if (
        task_manifest["status"] != "ASPD_V3_1_FEATURE_RULE_TASKS_FROZEN"
        or task_manifest["analysis_contract_fingerprint"] != contract.fingerprint()
        or task_manifest["heldout_contexts_exposed_to_proposer"] != 0
    ):
        raise ValueError("candidate-rule tasks are not frozen discovery-only inputs")
    if (
        base_manifest["status"] != "FAILED_INVALID_RESPONSES"
        or base_manifest["tasks"] != 625
        or base_manifest["heldout_contexts_exposed_to_proposer"] != 0
    ):
        raise ValueError("base proposal run is not the expected bounded failure")
    if (
        repair_manifest["status"] != "ASPD_V3_1_FEATURE_RULE_REPAIR_COMPLETE"
        or repair_manifest["invalid_repairs"] != 0
        or repair_manifest["base_responses_sha256"]
        != base_manifest["responses_sha256"]
        or repair_manifest["heldout_contexts_exposed_to_proposer"] != 0
    ):
        raise ValueError("bounded repair is incomplete or belongs to another base run")

    tasks = _jsonl(args.tasks_dir / "rule_tasks.jsonl")
    task_index = {row["task_id"]: row for row in tasks}
    base_path = args.base_dir / "candidate_rule_responses.jsonl"
    repair_path = args.repair_dir / "repaired_rule_responses.jsonl"
    base = {row["task_id"]: row for row in _jsonl(base_path)}
    repairs = {row["task_id"]: row for row in _jsonl(repair_path)}
    invalid_ids = {task_id for task_id, row in base.items() if not row["valid"]}
    if set(repairs) != invalid_ids or len(base) != len(tasks) or set(base) != set(task_index):
        raise ValueError("repair IDs do not exactly replace invalid base responses")
    if any(not row["valid"] for row in repairs.values()):
        raise ValueError("a repaired response remains invalid")
    merged = [repairs.get(task["task_id"], base[task["task_id"]]) for task in tasks]
    if any(not row["valid"] or row["parsed"] is None for row in merged):
        raise ValueError("merged candidate responses are not fully valid")

    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_component[str(row["component_key"])].append(row)
    if len(by_component) != 160:
        raise ValueError("candidate responses do not cover 160 components")
    candidate_rows = []
    for key in sorted(by_component):
        rows = by_component[key]
        global_rows = [row for row in rows if row["task_type"] == "global"]
        group_rows = sorted(
            (row for row in rows if row["task_type"] == "group_only"),
            key=lambda row: int(row["source_cluster"]),
        )
        if len(global_rows) != 1 or not group_rows:
            raise ValueError(f"candidate rule task shape changed for {key}")
        global_value = global_rows[0]["parsed"]
        candidate_rows.append(
            {
                "component_key": key,
                "general_rule": {
                    "rule": str(global_value["single_rule"]),
                    "category": str(global_value["single_rule_category"]),
                    "specificity": str(global_value["single_rule_specificity"]),
                    "uncertainty": str(global_value["uncertainty"]),
                },
                "group_rules": [
                    {
                        "source_cluster": int(row["source_cluster"]),
                        "rule": str(row["parsed"]["rule"]),
                        "category": str(row["parsed"]["category"]),
                        "specificity": str(row["parsed"]["specificity"]),
                        "uncertainty": str(row["parsed"]["uncertainty"]),
                    }
                    for row in group_rows
                ],
                "proposer_multiple_rule_explanation": global_value[
                    "multiple_rule_explanation"
                ],
                "proposer_groups_share_broader_rule": str(
                    global_value["groups_share_broader_rule"]
                ),
            }
        )

    args.output_dir.mkdir(parents=True)
    merged_path = args.output_dir / "merged_rule_responses.jsonl"
    candidates_path = args.output_dir / "candidate_rules.jsonl"
    _write_jsonl(merged_path, merged)
    _write_jsonl(candidates_path, candidate_rows)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_CANDIDATE_RULES_FROZEN",
        "git_sha": _git_sha(),
        "human_evidence_claimed": False,
        "heldout_contexts_exposed_to_proposer": 0,
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "tasks_manifest_sha256": sha256_file(
            args.tasks_dir / "preparation_manifest.json"
        ),
        "base_manifest_sha256": sha256_file(
            args.base_dir / "experiment_manifest.json"
        ),
        "base_responses_sha256": sha256_file(base_path),
        "repair_manifest_sha256": sha256_file(
            args.repair_dir / "experiment_manifest.json"
        ),
        "repair_responses_sha256": sha256_file(repair_path),
        "base_valid_responses": base_manifest["valid_responses"],
        "bounded_repairs": len(repairs),
        "final_valid_responses": len(merged),
        "components": len(candidate_rows),
        "general_rules": len(candidate_rows),
        "group_rules": sum(len(row["group_rules"]) for row in candidate_rows),
        "merged_responses_sha256": sha256_file(merged_path),
        "candidate_rules_sha256": sha256_file(candidates_path),
    }
    _write_json(args.output_dir / "freeze_manifest.json", manifest)
    (args.output_dir / "FROZEN").write_text(
        "ASPD_V3_1_FEATURE_CANDIDATE_RULES_FROZEN\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

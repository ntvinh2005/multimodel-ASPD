"""Freeze synthetic calibration and held-out rule-presence tasks for Phi-4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract

SYSTEM = """Judge whether the candidate rule is visibly present in the supplied context,
especially at or near the ⟪ANCHOR⟫...⟪/ANCHOR⟫ span. Do not predict a hidden feature and
do not use outside information. Reply with exactly one label: YES, NO, or UNCERTAIN."""


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rules-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
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


def _messages(rule: str, specificity: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Candidate rule: {rule}\n"
                f"Operational detail: {specificity}\n\n"
                f"Context:\n{context}\n\nLabel:"
            ),
        },
    ]


def _calibration_rows() -> list[dict[str, Any]]:
    specifications = [
        (
            "An explicit negation word occurs at or immediately around the anchor.",
            "Look for not, no, never, or cannot.",
            [
                "The device is ⟪ANCHOR⟫not⟪/ANCHOR⟫ available today.",
                "There is ⟪ANCHOR⟫no⟪/ANCHOR⟫ remaining balance.",
                "We will ⟪ANCHOR⟫never⟪/ANCHOR⟫ publish the key.",
                "The request ⟪ANCHOR⟫cannot⟪/ANCHOR⟫ be completed.",
            ],
            [
                "The device is ⟪ANCHOR⟫available⟪/ANCHOR⟫ today.",
                "There is a ⟪ANCHOR⟫remaining⟪/ANCHOR⟫ balance.",
                "We will ⟪ANCHOR⟫publish⟪/ANCHOR⟫ the key.",
                "The request was ⟪ANCHOR⟫completed⟪/ANCHOR⟫ successfully.",
            ],
        ),
        (
            "A Python function definition begins at the anchor.",
            "Look for the syntax def name(...):.",
            [
                "⟪ANCHOR⟫def parse(text):⟪/ANCHOR⟫\n    return text.strip()",
                "⟪ANCHOR⟫def add(a, b):⟪/ANCHOR⟫\n    return a + b",
                "class A:\n    ⟪ANCHOR⟫def run(self):⟪/ANCHOR⟫\n        pass",
                "@cache\n⟪ANCHOR⟫def load(path):⟪/ANCHOR⟫\n    return open(path)",
            ],
            [
                "The parser ⟪ANCHOR⟫returns text⟪/ANCHOR⟫ after cleanup.",
                "Add the ⟪ANCHOR⟫two values⟪/ANCHOR⟫ together.",
                "class A:\n    ⟪ANCHOR⟫run = True⟪/ANCHOR⟫",
                "The file ⟪ANCHOR⟫loads quickly⟪/ANCHOR⟫ from disk.",
            ],
        ),
        (
            "An HTML opening or closing tag occurs at the anchor.",
            "Look for angle-bracket markup such as <div> or </p>.",
            [
                "⟪ANCHOR⟫<div>⟪/ANCHOR⟫content</div>",
                "text⟪ANCHOR⟫</p>⟪/ANCHOR⟫",
                "⟪ANCHOR⟫<a href='/'>⟪/ANCHOR⟫home</a>",
                "⟪ANCHOR⟫<br/>⟪/ANCHOR⟫ next line",
            ],
            [
                "The value is ⟪ANCHOR⟫less than⟪/ANCHOR⟫ ten.",
                "text ⟪ANCHOR⟫paragraph end⟪/ANCHOR⟫",
                "Visit the ⟪ANCHOR⟫home page⟪/ANCHOR⟫.",
                "Start the ⟪ANCHOR⟫next line⟪/ANCHOR⟫ here.",
            ],
        ),
        (
            "A mathematical or programming equality appears at the anchor.",
            "Look for an equals sign connecting two expressions.",
            [
                "x ⟪ANCHOR⟫=⟪/ANCHOR⟫ 4",
                "total ⟪ANCHOR⟫==⟪/ANCHOR⟫ expected",
                "f(x) ⟪ANCHOR⟫=⟪/ANCHOR⟫ x^2",
                "name ⟪ANCHOR⟫=⟪/ANCHOR⟫ 'Ada'",
            ],
            [
                "x is ⟪ANCHOR⟫greater⟪/ANCHOR⟫ than 4",
                "The total ⟪ANCHOR⟫matches⟪/ANCHOR⟫ expected output.",
                "The function is ⟪ANCHOR⟫quadratic⟪/ANCHOR⟫.",
                "Her ⟪ANCHOR⟫name⟪/ANCHOR⟫ is Ada.",
            ],
        ),
        (
            "The anchored span is part of a direct question.",
            "The sentence asks a question and ends with a question mark.",
            [
                "⟪ANCHOR⟫Where⟪/ANCHOR⟫ is the station?",
                "Can we ⟪ANCHOR⟫start⟪/ANCHOR⟫ now?",
                "Why did it ⟪ANCHOR⟫fail⟪/ANCHOR⟫?",
                "Is the result ⟪ANCHOR⟫correct⟪/ANCHOR⟫?",
            ],
            [
                "⟪ANCHOR⟫Where⟪/ANCHOR⟫ the road bends, turn left.",
                "We can ⟪ANCHOR⟫start⟪/ANCHOR⟫ now.",
                "The process did ⟪ANCHOR⟫fail⟪/ANCHOR⟫ yesterday.",
                "The result is ⟪ANCHOR⟫correct⟪/ANCHOR⟫.",
            ],
        ),
        (
            "A Markdown-style bullet marker starts the anchored line.",
            "Look for -, *, or + followed by a list item.",
            [
                "⟪ANCHOR⟫- first item⟪/ANCHOR⟫\n- second item",
                "⟪ANCHOR⟫* apples⟪/ANCHOR⟫\n* pears",
                "⟪ANCHOR⟫+ enabled⟪/ANCHOR⟫\n+ cached",
                "List:\n  ⟪ANCHOR⟫- nested item⟪/ANCHOR⟫",
            ],
            [
                "Five ⟪ANCHOR⟫- three⟪/ANCHOR⟫ equals two.",
                "The ⟪ANCHOR⟫* operator⟪/ANCHOR⟫ multiplies values.",
                "A ⟪ANCHOR⟫+ sign⟪/ANCHOR⟫ indicates addition.",
                "The sentence has a ⟪ANCHOR⟫nested clause⟪/ANCHOR⟫.",
            ],
        ),
        (
            "An ISO-format calendar date occurs at the anchor.",
            "Look for YYYY-MM-DD with numeric year, month, and day.",
            [
                "Released on ⟪ANCHOR⟫2026-11-15⟪/ANCHOR⟫.",
                "Date: ⟪ANCHOR⟫1999-01-31⟪/ANCHOR⟫",
                "From ⟪ANCHOR⟫2024-02-29⟪/ANCHOR⟫ onward.",
                "Deadline ⟪ANCHOR⟫2030-12-01⟪/ANCHOR⟫.",
            ],
            [
                "Released in ⟪ANCHOR⟫November 2026⟪/ANCHOR⟫.",
                "Date: ⟪ANCHOR⟫January 31⟪/ANCHOR⟫",
                "From ⟪ANCHOR⟫next Thursday⟪/ANCHOR⟫ onward.",
                "The deadline is ⟪ANCHOR⟫soon⟪/ANCHOR⟫.",
            ],
        ),
        (
            "A JSON-style quoted key and colon occur at the anchor.",
            "Look for syntax like \"key\": value.",
            [
                "{⟪ANCHOR⟫\"name\":⟪/ANCHOR⟫ \"Ada\"}",
                "{⟪ANCHOR⟫\"count\":⟪/ANCHOR⟫ 4}",
                "{⟪ANCHOR⟫\"active\":⟪/ANCHOR⟫ true}",
                "{⟪ANCHOR⟫\"items\":⟪/ANCHOR⟫ []}",
            ],
            [
                "Her ⟪ANCHOR⟫name is⟪/ANCHOR⟫ Ada.",
                "The ⟪ANCHOR⟫count is⟪/ANCHOR⟫ four.",
                "The service is ⟪ANCHOR⟫active⟪/ANCHOR⟫.",
                "There are no ⟪ANCHOR⟫items⟪/ANCHOR⟫.",
            ],
        ),
        (
            "An email address occurs at the anchor.",
            "Look for a local part, @ sign, and domain.",
            [
                "Contact ⟪ANCHOR⟫ada@example.org⟪/ANCHOR⟫.",
                "Send to ⟪ANCHOR⟫team+lab@site.com⟪/ANCHOR⟫.",
                "Owner: ⟪ANCHOR⟫x.y@domain.net⟪/ANCHOR⟫",
                "Email ⟪ANCHOR⟫support@company.io⟪/ANCHOR⟫",
            ],
            [
                "Contact ⟪ANCHOR⟫Ada⟪/ANCHOR⟫ tomorrow.",
                "Send to the ⟪ANCHOR⟫team⟪/ANCHOR⟫.",
                "Owner: ⟪ANCHOR⟫x.y⟪/ANCHOR⟫",
                "Use the ⟪ANCHOR⟫support page⟪/ANCHOR⟫.",
            ],
        ),
        (
            "A shell command-line option beginning with two hyphens occurs at the anchor.",
            "Look for tokens such as --help or --output.",
            [
                "tool ⟪ANCHOR⟫--help⟪/ANCHOR⟫",
                "run ⟪ANCHOR⟫--output file.txt⟪/ANCHOR⟫",
                "python app.py ⟪ANCHOR⟫--verbose⟪/ANCHOR⟫",
                "cmd ⟪ANCHOR⟫--threads 8⟪/ANCHOR⟫",
            ],
            [
                "The tool provides ⟪ANCHOR⟫help⟪/ANCHOR⟫.",
                "Write the ⟪ANCHOR⟫output file⟪/ANCHOR⟫.",
                "The report is ⟪ANCHOR⟫verbose⟪/ANCHOR⟫.",
                "Use eight ⟪ANCHOR⟫threads⟪/ANCHOR⟫.",
            ],
        ),
    ]
    rows = []
    for rule_index, (rule, specificity, positives, negatives) in enumerate(specifications):
        for expected, contexts in ((True, positives), (False, negatives)):
            for context_index, context in enumerate(contexts):
                task_id = f"calibration:{rule_index}:{int(expected)}:{context_index}"
                rows.append(
                    {
                        "task_id": task_id,
                        "task_type": "synthetic_calibration",
                        "expected_rule_present": expected,
                        "messages": _messages(rule, specificity, context),
                    }
                )
    return rows


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    rules_manifest = json.loads((args.rules_dir / "freeze_manifest.json").read_text())
    heldout_manifest = json.loads(
        (args.heldout_dir / "preparation_manifest.json").read_text()
    )
    if (
        rules_manifest["status"] != "ASPD_V3_1_FEATURE_CANDIDATE_RULES_FROZEN"
        or rules_manifest["analysis_contract_fingerprint"] != contract.fingerprint()
        or rules_manifest["heldout_contexts_exposed_to_proposer"] != 0
    ):
        raise ValueError("candidate rules are not frozen discovery-only proposals")
    if (
        heldout_manifest["status"] != "ASPD_V3_1_FEATURE_COHERENCE_HELDOUT_PREPARED"
        or heldout_manifest["analysis_contract_fingerprint"] != contract.fingerprint()
    ):
        raise ValueError("held-out examples are not frozen under this contract")
    rules = _jsonl(args.rules_dir / "candidate_rules.jsonl")
    rule_index = {str(row["component_key"]): row for row in rules}
    examples = _jsonl(args.heldout_dir / "heldout_validation_examples.jsonl")
    if len(rule_index) != 160 or not examples:
        raise ValueError("rule or held-out component coverage changed")

    validation = []
    for example in examples:
        key = str(example["component_key"])
        candidate = rule_index[key]
        selected_rules = [
            {
                "rule_id": f"{key}:general",
                "rule_type": "general",
                "source_cluster": None,
                **candidate["general_rule"],
            }
        ]
        if example["condition"] == "ordinary":
            selected_rules.extend(
                {
                    "rule_id": f"{key}:group:{row['source_cluster']}",
                    "rule_type": "group",
                    **row,
                }
                for row in candidate["group_rules"]
            )
        for rule in selected_rules:
            task_id = hashlib.sha256(
                f"{rule['rule_id']}|{example['example_id']}".encode()
            ).hexdigest()
            validation.append(
                {
                    "task_id": task_id,
                    "task_type": "heldout_rule_presence",
                    "component_key": key,
                    "site": example["site"],
                    "cohort": example["cohort"],
                    "rule_id": rule["rule_id"],
                    "rule_type": rule["rule_type"],
                    "source_cluster": rule.get("source_cluster"),
                    "example_id": example["example_id"],
                    "pair_id": example["pair_id"],
                    "condition": example["condition"],
                    "actual_natural_firing": example["actual_natural_firing"],
                    "target_discovery_cluster": example["target_discovery_cluster"],
                    "activation_bin": example.get("activation_bin"),
                    "document_id": example["document_id"],
                    "messages": _messages(
                        str(rule["rule"]),
                        str(rule["specificity"]),
                        str(example["marked_text"]),
                    ),
                }
            )
    calibration = _calibration_rows()
    if len(calibration) != 80:
        raise ValueError("synthetic calibration set must contain 80 tasks")
    if len({row["task_id"] for row in validation}) != len(validation):
        raise ValueError("held-out validator task IDs are not unique")
    args.output_dir.mkdir(parents=True)
    calibration_path = args.output_dir / "calibration_tasks.jsonl"
    validation_path = args.output_dir / "validation_tasks.jsonl"
    _write_jsonl(calibration_path, calibration)
    _write_jsonl(validation_path, validation)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_FEATURE_RULE_VALIDATION_TASKS_FROZEN",
        "git_sha": _git_sha(),
        "human_evidence_claimed": False,
        "proposer_model_id": contract.proposer_model_id,
        "validator_model_id": contract.validator_model_id,
        "independent_model_families": True,
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "rules_manifest_sha256": sha256_file(args.rules_dir / "freeze_manifest.json"),
        "candidate_rules_sha256": sha256_file(
            args.rules_dir / "candidate_rules.jsonl"
        ),
        "heldout_manifest_sha256": sha256_file(
            args.heldout_dir / "preparation_manifest.json"
        ),
        "calibration_tasks": len(calibration),
        "validation_tasks": len(validation),
        "general_rule_tasks": sum(row["rule_type"] == "general" for row in validation),
        "group_rule_tasks": sum(row["rule_type"] == "group" for row in validation),
        "calibration_tasks_sha256": sha256_file(calibration_path),
        "validation_tasks_sha256": sha256_file(validation_path),
    }
    _write_json(args.output_dir / "preparation_manifest.json", manifest)
    (args.output_dir / "FROZEN").write_text(
        "ASPD_V3_1_FEATURE_RULE_VALIDATION_TASKS_FROZEN\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

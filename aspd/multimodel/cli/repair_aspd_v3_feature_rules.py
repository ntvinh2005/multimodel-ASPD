"""Repair only invalid discovery-rule JSON without changing proposed meaning."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from aspd.multimodel.cli.propose_aspd_v3_feature_rules import (
    _parse_json,
    _signature,
    _valid,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--base-responses", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
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


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluation_git_sha = _git_sha()
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    if args.model.name != contract.proposer_revision:
        raise ValueError("repair proposer revision changed")
    tasks = {row["task_id"]: row for row in _jsonl(args.tasks_dir / "rule_tasks.jsonl")}
    base_rows = _jsonl(args.base_responses)
    invalid = [row for row in base_rows if not row["valid"]]
    if not invalid or len(base_rows) != 625:
        raise ValueError("repair requires the complete base run with invalid responses")
    repair_tasks = []
    for row in invalid:
        task = tasks[row["task_id"]]
        repair_tasks.append(
            {
                **task,
                "messages": [
                    *task["messages"],
                    {"role": "assistant", "content": row["raw_output"]},
                    {
                        "role": "user",
                        "content": (
                            "The previous response is invalid JSON. Preserve its proposed "
                            "linguistic meaning, but return exactly one compact valid JSON "
                            "object matching the requested schema. Add missing commas, escape "
                            "quotes and backslashes, remove string concatenation and remove any "
                            "extra object or prose."
                        ),
                    },
                ],
            }
        )
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16
    ).to("cuda")
    model.eval().requires_grad_(False)
    before = _signature(model)
    responses = []
    with torch.inference_mode():
        for start in range(0, len(repair_tasks), args.batch_size):
            batch_tasks = repair_tasks[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    task["messages"], tokenize=False, add_generation_prompt=True
                )
                for task in batch_tasks
            ]
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=contract.proposer_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            raw_outputs = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for task, raw in zip(batch_tasks, raw_outputs, strict=True):
                parsed = _parse_json(raw)
                responses.append(
                    {
                        "task_id": task["task_id"],
                        "component_key": task["component_key"],
                        "task_type": task["task_type"],
                        "source_cluster": task["source_cluster"],
                        "valid": _valid(task, parsed),
                        "parsed": parsed,
                        "raw_output": raw,
                    }
                )
    if _signature(model) != before:
        raise RuntimeError("repair proposer model parameters changed")
    responses_path = args.output_dir / "repaired_rule_responses.jsonl"
    with responses_path.open("x", encoding="utf-8") as handle:
        for row in responses:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )
    remaining_invalid = sum(not row["valid"] for row in responses)
    manifest = {
        "schema_version": 1,
        "status": (
            "ASPD_V3_1_FEATURE_RULE_REPAIR_COMPLETE"
            if remaining_invalid == 0
            else "FAILED_INVALID_REPAIRS"
        ),
        "git_sha": evaluation_git_sha,
        "human_evidence_claimed": False,
        "heldout_contexts_exposed_to_proposer": 0,
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "tasks_manifest_sha256": sha256_file(
            args.tasks_dir / "preparation_manifest.json"
        ),
        "base_responses_sha256": sha256_file(args.base_responses),
        "proposer_model_id": contract.proposer_model_id,
        "proposer_revision": contract.proposer_revision,
        "repair_tasks": len(repair_tasks),
        "valid_repairs": len(repair_tasks) - remaining_invalid,
        "invalid_repairs": remaining_invalid,
        "responses_sha256": sha256_file(responses_path),
        "source_weights_unchanged": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "hostname": socket.gethostname(),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if remaining_invalid:
        raise RuntimeError(f"{remaining_invalid} proposer repairs remain invalid")
    (args.output_dir / "COMPLETE").write_text(
        "ASPD_V3_1_FEATURE_RULE_REPAIR_COMPLETE\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

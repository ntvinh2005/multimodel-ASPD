"""Independently judge frozen rule presence with pinned Phi-4-mini."""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "validation"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
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


def parse_rule_presence(value: str) -> Literal["YES", "NO", "UNCERTAIN"] | None:
    match = re.search(r"\b(YES|NO|UNCERTAIN)\b", value.upper())
    return match.group(1) if match else None


def _signature(model: torch.nn.Module) -> tuple[int, float, float]:
    count = 0
    total = 0.0
    square = 0.0
    for parameter in model.parameters():
        value = parameter.detach().float()
        count += value.numel()
        total += float(value.sum())
        square += float(value.square().sum())
    return count, total, square


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluation_git_sha = _git_sha()
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    if args.model.name != contract.validator_revision or any(
        value in contract.validator_model_id.lower()
        for value in contract.forbidden_model_substrings
    ):
        raise ValueError("validator model or revision violates the frozen contract")
    preparation = json.loads(
        (args.tasks_dir / "preparation_manifest.json").read_text()
    )
    if (
        preparation["status"]
        != "ASPD_V3_1_FEATURE_RULE_VALIDATION_TASKS_FROZEN"
        or preparation["analysis_contract_fingerprint"] != contract.fingerprint()
        or not preparation["independent_model_families"]
    ):
        raise ValueError("rule validation tasks are not frozen and independent")
    input_name = (
        "calibration_tasks.jsonl"
        if args.mode == "calibration"
        else "validation_tasks.jsonl"
    )
    tasks_path = args.tasks_dir / input_name
    expected_hash = preparation[f"{args.mode}_tasks_sha256"]
    if sha256_file(tasks_path) != expected_hash:
        raise ValueError("validator task input changed")
    tasks = _jsonl(tasks_path)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval().requires_grad_(False)
    before = _signature(model)
    results = []
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch_tasks = tasks[start : start + args.batch_size]
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
                max_length=1536,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=contract.validator_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            raw_outputs = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for task, raw in zip(batch_tasks, raw_outputs, strict=True):
                prediction = parse_rule_presence(raw)
                results.append(
                    {
                        "task_id": task["task_id"],
                        "prediction": prediction,
                        "valid": prediction is not None,
                        "raw_output": raw,
                    }
                )
    if _signature(model) != before:
        raise RuntimeError("validator model parameters changed")
    invalid = sum(not row["valid"] for row in results)
    abstained = sum(
        row["prediction"] in {None, "UNCERTAIN"} for row in results
    )
    summary: dict[str, Any] = {
        "tasks": len(tasks),
        "valid_predictions": len(tasks) - invalid,
        "invalid_predictions": invalid,
        "abstentions": abstained,
        "abstention_fraction": abstained / len(tasks),
    }
    gate_failed = invalid / len(tasks) > 0.01
    if args.mode == "calibration":
        task_index = {row["task_id"]: row for row in tasks}
        positive = [row for row in results if task_index[row["task_id"]]["expected_rule_present"]]
        negative = [row for row in results if not task_index[row["task_id"]]["expected_rule_present"]]
        sensitivity = sum(row["prediction"] == "YES" for row in positive) / len(positive)
        specificity = sum(row["prediction"] == "NO" for row in negative) / len(negative)
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        summary.update(
            {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            }
        )
        gate_failed = gate_failed or (
            balanced_accuracy < contract.validator_calibration_balanced_accuracy
            or summary["abstention_fraction"]
            > contract.validator_calibration_maximum_abstention
        )
    results_path = args.output_dir / "validator_results.jsonl"
    with results_path.open("x", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "schema_version": 1,
        "status": (
            f"ASPD_V3_1_FEATURE_RULE_{args.mode.upper()}_COMPLETE"
            if not gate_failed
            else f"FAILED_{args.mode.upper()}_RELIABILITY_GATE"
        ),
        "mode": args.mode,
        "git_sha": evaluation_git_sha,
        "human_evidence_claimed": False,
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "tasks_manifest_sha256": sha256_file(
            args.tasks_dir / "preparation_manifest.json"
        ),
        "tasks_sha256": sha256_file(tasks_path),
        "validator_model_id": contract.validator_model_id,
        "validator_revision": contract.validator_revision,
        "summary": summary,
        "results_sha256": sha256_file(results_path),
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
    if gate_failed:
        raise RuntimeError(f"{args.mode} validator reliability gate failed: {summary}")
    (args.output_dir / "COMPLETE").write_text(
        f"ASPD_V3_1_FEATURE_RULE_{args.mode.upper()}_COMPLETE\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

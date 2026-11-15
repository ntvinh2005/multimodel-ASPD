"""Generate and freeze discovery-only candidate rules with pinned Gemma-3."""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.feature_coherence import FeatureCoherenceAnalysisContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
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


def _parse_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    start = cleaned.find("{")
    if start < 0:
        return None
    candidate = cleaned[start:]
    try:
        value, _end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        # Some instruction models emit a backslash before punctuation such as
        # ``\|``. JSON does not define that escape. Doubling only these invalid
        # backslashes preserves the proposed text while making the JSON lexical.
        candidate = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
        try:
            value, _end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _valid(task: dict[str, Any], value: dict[str, Any] | None) -> bool:
    if value is None:
        return False
    if task["task_type"] == "global":
        required = {
            "single_rule",
            "single_rule_category",
            "single_rule_specificity",
            "multiple_rule_explanation",
            "groups_share_broader_rule",
            "uncertainty",
        }
        return required <= set(value) and isinstance(
            value["multiple_rule_explanation"], list
        )
    return {"rule", "category", "specificity", "uncertainty"} <= set(value)


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluation_git_sha = _git_sha()
    contract = FeatureCoherenceAnalysisContract.load(args.contract)
    if args.model.name != contract.proposer_revision or any(
        value in contract.proposer_model_id.lower()
        for value in contract.forbidden_model_substrings
    ):
        raise ValueError("proposer model or revision violates the frozen contract")
    task_manifest = json.loads(
        (args.tasks_dir / "preparation_manifest.json").read_text()
    )
    tasks_path = args.tasks_dir / "rule_tasks.jsonl"
    if (
        task_manifest["status"] != "ASPD_V3_1_FEATURE_RULE_TASKS_FROZEN"
        or task_manifest["tasks_sha256"] != sha256_file(tasks_path)
        or task_manifest["analysis_contract_fingerprint"] != contract.fingerprint()
    ):
        raise ValueError("rule tasks are not frozen under this analysis contract")
    tasks = _jsonl(tasks_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
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
                max_length=8192,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=contract.proposer_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            raw_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
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
        raise RuntimeError("proposer model parameters changed")
    invalid = sum(not row["valid"] for row in responses)
    invalid_gate_failed = invalid / len(responses) > 0.01
    responses_path = args.output_dir / "candidate_rule_responses.jsonl"
    with responses_path.open("x", encoding="utf-8") as handle:
        for row in responses:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )
    manifest = {
        "schema_version": 1,
        "status": (
            "FAILED_INVALID_RESPONSES"
            if invalid_gate_failed
            else "ASPD_V3_1_FEATURE_RULE_PROPOSALS_COMPLETE"
        ),
        "git_sha": evaluation_git_sha,
        "human_evidence_claimed": False,
        "heldout_contexts_exposed_to_proposer": 0,
        "analysis_contract_sha256": sha256_file(args.contract),
        "analysis_contract_fingerprint": contract.fingerprint(),
        "tasks_manifest_sha256": sha256_file(
            args.tasks_dir / "preparation_manifest.json"
        ),
        "tasks_sha256": sha256_file(tasks_path),
        "proposer_model_id": contract.proposer_model_id,
        "proposer_revision": contract.proposer_revision,
        "tasks": len(tasks),
        "valid_responses": len(tasks) - invalid,
        "invalid_responses": invalid,
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
    if invalid_gate_failed:
        raise RuntimeError(
            f"invalid proposer response fraction exceeds 1%: {invalid}/{len(responses)}"
        )
    (args.output_dir / "COMPLETE").write_text(
        "ASPD_V3_1_FEATURE_RULE_PROPOSALS_COMPLETE\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

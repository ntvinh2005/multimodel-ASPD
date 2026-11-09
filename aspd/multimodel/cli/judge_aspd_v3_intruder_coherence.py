"""Run position-independent omission/coherence Intruder judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aspd.multimodel.cli.judge_aspd_v3_intruder_final import (
    _load_model,
    _snapshot_path,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import (
    FinalIntruderCoherenceJudgeContract,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "final"), required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _omission_task(task: dict[str, Any], omitted_position: int) -> dict[str, Any]:
    examples = list(task["examples"])
    if len(examples) != 5 or not 1 <= omitted_position <= 5:
        raise ValueError("coherence omission requires five candidates")
    remaining = [
        (index + 1, str(row["text"]))
        for index, row in enumerate(examples)
        if index + 1 != omitted_position
    ]
    remaining.sort(key=lambda row: hashlib.sha256(row[1].encode()).hexdigest())
    return {
        "trial_id": f"{task['trial_id']}-omit-{omitted_position}",
        "parent_trial_id": task["trial_id"],
        "omitted_canonical_position": omitted_position,
        "control_kind": task.get("control_kind"),
        "remaining": [
            {
                "display_label": label,
                "text": text,
                "canonical_position": canonical_position,
            }
            for label, (canonical_position, text) in zip(
                ("A", "B", "C", "D"), remaining, strict=True
            )
        ],
    }


def _prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    excerpts = "\n\n".join(
        f"Excerpt {row['display_label']}:\n{row['text']}" for row in task["remaining"]
    )
    return [
        {
            "role": "system",
            "content": (
                "You independently score semantic coherence. Judge whether all four excerpts "
                "share one specific, distinguishable semantic pattern. Ignore ordering, style, "
                "length, formatting, and superficial word overlap. Use an integer coherence "
                "score from 0 (unrelated) to 100 (one clear specific pattern). Output only a "
                "JSON object with exactly the keys coherence and theme."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{excerpts}\n\nScore the four excerpts as one group. Return exactly the "
                "required JSON object and no additional text."
            ),
        },
    ]


def _parse_response(
    text: str, contract: FinalIntruderCoherenceJudgeContract
) -> dict[str, Any]:
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if len(matches) != 1:
        return {"valid": False, "coherence": None, "theme": None, "reason": "json_count"}
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return {"valid": False, "coherence": None, "theme": None, "reason": "invalid_json"}
    if set(payload) != {"coherence", "theme"}:
        return {"valid": False, "coherence": None, "theme": None, "reason": "unexpected_keys"}
    score = payload["coherence"]
    theme = payload["theme"]
    if (
        isinstance(score, bool)
        or not isinstance(score, int)
        or not contract.coherence_minimum <= score <= contract.coherence_maximum
        or not isinstance(theme, str)
        or not theme.strip()
    ):
        return {"valid": False, "coherence": None, "theme": None, "reason": "invalid_fields"}
    return {"valid": True, "coherence": score, "theme": theme.strip(), "reason": None}


def _generate(
    *,
    tokenizer: Any,
    model: Any,
    tasks: list[dict[str, Any]],
    contract: FinalIntruderCoherenceJudgeContract,
) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(tasks), contract.batch_size):
        batch = tasks[start : start + contract.batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                _prompt(task), tokenize=False, add_generation_prompt=True
            )
            for task in batch
        ]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8_192,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=contract.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
            for row in generated
        )
    return outputs


def _parent_rows(
    expanded: list[dict[str, Any]],
    raw_outputs: list[str],
    answers: list[dict[str, Any]],
    contract: FinalIntruderCoherenceJudgeContract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answer_by_id = {str(row["trial_id"]): row for row in answers}
    prompt_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task, raw in zip(expanded, raw_outputs, strict=True):
        parsed = _parse_response(raw, contract)
        row = {
            "trial_id": task["trial_id"],
            "parent_trial_id": task["parent_trial_id"],
            "omitted_canonical_position": task["omitted_canonical_position"],
            "control_kind": task.get("control_kind"),
            "raw_output": raw,
            **parsed,
        }
        prompt_rows.append(row)
        grouped[str(task["parent_trial_id"])].append(row)
    parents = []
    for parent_id, rows in grouped.items():
        if parent_id not in answer_by_id or len(rows) != 5:
            raise ValueError(f"incomplete answer/omission group for {parent_id}")
        answer = answer_by_id[parent_id]
        valid_scores = [row for row in rows if row["valid"]]
        unique_argmax = False
        predicted = None
        if len(valid_scores) == 5:
            maximum = max(int(row["coherence"]) for row in valid_scores)
            winners = [
                row for row in valid_scores if int(row["coherence"]) == maximum
            ]
            if len(winners) == 1:
                unique_argmax = True
                predicted = int(winners[0]["omitted_canonical_position"])
        correct_position = answer["intruder_position"]
        parents.append(
            {
                **answer,
                "all_scores_valid": len(valid_scores) == 5,
                "unique_argmax": unique_argmax,
                "predicted_intruder_position": predicted,
                "correct": bool(
                    unique_argmax
                    and correct_position is not None
                    and predicted == int(correct_position)
                ),
                "scores_by_omitted_position": {
                    str(row["omitted_canonical_position"]): row["coherence"]
                    for row in rows
                },
            }
        )
    return prompt_rows, parents


def _summary(
    *,
    mode: str,
    prompt_rows: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    replay_exact: bool | None,
    contract: FinalIntruderCoherenceJudgeContract,
) -> dict[str, Any]:
    positive = [row for row in parents if row.get("control_kind") == "positive"]
    negative = [row for row in parents if row.get("control_kind") == "negative"]
    valid_response_rate = float(np.mean([row["valid"] for row in prompt_rows]))
    unique_rate = float(np.mean([row["unique_argmax"] for row in parents]))
    positive_accuracy = (
        float(np.mean([row["correct"] for row in positive])) if positive else None
    )
    positive_unique = (
        float(np.mean([row["unique_argmax"] for row in positive])) if positive else None
    )
    negative_unique = (
        float(np.mean([row["unique_argmax"] for row in negative])) if negative else None
    )
    gates = None
    if mode == "calibration":
        gates = {
            "positive_accuracy": bool(
                positive_accuracy is not None
                and positive_accuracy >= contract.minimum_positive_control_accuracy
            ),
            "valid_response_rate": valid_response_rate >= contract.minimum_valid_response_rate,
            "positive_unique_argmax_rate": bool(
                positive_unique is not None
                and positive_unique >= contract.minimum_positive_unique_argmax_rate
            ),
            "exact_deterministic_replay": bool(replay_exact),
        }
        status = "JUDGE_VALIDATED" if all(gates.values()) else "JUDGE_VALIDATION_FAILED"
    else:
        status = (
            "SCIENTIFIC_COHERENCE_RUN_COMPLETE"
            if valid_response_rate >= contract.minimum_valid_response_rate
            and unique_rate >= contract.minimum_final_unique_argmax_rate
            else "SCIENTIFIC_COHERENCE_VALIDATION_FAILED"
        )
    return {
        "schema_version": 5,
        "mode": mode,
        "status": status,
        "parent_tasks": len(parents),
        "omission_prompts": len(prompt_rows),
        "valid_response_rate": valid_response_rate,
        "unique_argmax_parent_rate": unique_rate,
        "positive_control_accuracy": positive_accuracy,
        "positive_unique_argmax_rate": positive_unique,
        "negative_unique_argmax_rate_diagnostic": negative_unique,
        "deterministic_replay_exact": replay_exact,
        "calibration_gates": gates,
    }


def main() -> None:
    args = _args()
    evaluation_git_sha = _git_sha()
    contract = FinalIntruderCoherenceJudgeContract.load(args.contract)
    if "qwen" in contract.judge_model_id.casefold() or contract.paid_api:
        raise ValueError("judge violates the local non-Qwen restriction")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.mode == "final":
        if args.calibration_summary is None:
            raise ValueError("final mode requires --calibration-summary")
        calibration = _json(args.calibration_summary)
        if calibration.get("status") != "JUDGE_VALIDATED":
            raise ValueError("coherence judge calibration did not pass")
        if calibration.get("contract_fingerprint") != contract.fingerprint():
            raise ValueError("coherence calibration contract mismatch")
    tasks = _jsonl(args.tasks)
    if len({str(row["trial_id"]) for row in tasks}) != len(tasks):
        raise ValueError("duplicate parent task IDs")
    expanded = [
        _omission_task(task, omitted)
        for task in tasks
        for omitted in range(1, contract.omissions_per_parent + 1)
    ]
    args.output_dir.mkdir(parents=True)
    _write_jsonl(
        args.output_dir / "rendered_prompt_messages.jsonl",
        [
            {
                "trial_id": task["trial_id"],
                "parent_trial_id": task["parent_trial_id"],
                "omitted_canonical_position": task["omitted_canonical_position"],
                "messages": _prompt(task),
            }
            for task in expanded
        ],
    )
    snapshot = _snapshot_path(contract)  # type: ignore[arg-type]
    started = time.time()
    tokenizer, model = _load_model(contract)  # type: ignore[arg-type]
    raw_outputs = _generate(
        tokenizer=tokenizer, model=model, tasks=expanded, contract=contract
    )
    replay_exact = None
    if args.mode == "calibration":
        replay_count = min(contract.deterministic_replay_prompts, len(expanded))
        replay = _generate(
            tokenizer=tokenizer,
            model=model,
            tasks=expanded[:replay_count],
            contract=contract,
        )
        replay_exact = replay == raw_outputs[:replay_count]
    del model
    torch.cuda.empty_cache()
    answers = _jsonl(args.answer_key)
    prompt_rows, parents = _parent_rows(
        expanded, raw_outputs, answers, contract
    )
    summary = _summary(
        mode=args.mode,
        prompt_rows=prompt_rows,
        parents=parents,
        replay_exact=replay_exact,
        contract=contract,
    )
    summary.update(
        {
            "contract_fingerprint": contract.fingerprint(),
            "contract_sha256": sha256_file(args.contract),
            "tasks_sha256": sha256_file(args.tasks),
            "answer_key_sha256": sha256_file(args.answer_key),
        }
    )
    _write_jsonl(args.output_dir / "omission_results.jsonl", prompt_rows)
    _write_jsonl(args.output_dir / "parent_results.jsonl", parents)
    _write_json(args.output_dir / "judge_summary.json", summary)
    manifest = {
        "schema_version": 5,
        "git_sha": evaluation_git_sha,
        "mode": args.mode,
        "judge_model_id": contract.judge_model_id,
        "judge_revision": contract.judge_revision,
        "judge_snapshot_path": str(snapshot),
        "judge_config_sha256": sha256_file(snapshot / "config.json"),
        "judge_weight_index_sha256": sha256_file(
            snapshot / "model.safetensors.index.json"
        ),
        "contract_fingerprint": contract.fingerprint(),
        "tasks_sha256": sha256_file(args.tasks),
        "answer_key_sha256": sha256_file(args.answer_key),
        "calibration_summary_sha256": (
            sha256_file(args.calibration_summary) if args.calibration_summary else None
        ),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.time() - started,
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "omission_results_sha256": sha256_file(
            args.output_dir / "omission_results.jsonl"
        ),
        "parent_results_sha256": sha256_file(args.output_dir / "parent_results.jsonl"),
        "rendered_prompt_messages_sha256": sha256_file(
            args.output_dir / "rendered_prompt_messages.jsonl"
        ),
        "summary_sha256": sha256_file(args.output_dir / "judge_summary.json"),
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

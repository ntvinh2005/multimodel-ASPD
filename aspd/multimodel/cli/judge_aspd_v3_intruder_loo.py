"""Run a position-decoupled leave-one-out Intruder judge on blinded tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.cli.judge_aspd_v3_intruder import _control_tasks
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import (
    IntruderLeaveOneOutJudgeContract,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controls-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _ordered_remaining(task: dict[str, Any], omitted_position: int) -> list[str]:
    remaining = [
        row
        for row in task["examples"]
        if int(row["position"]) != omitted_position
    ]
    if len(remaining) != 4:
        raise ValueError("leave-one-out task must contain five unique positions")

    def order_key(row: dict[str, Any]) -> str:
        payload = f"{task['trial_id']}|{omitted_position}|{row['position']}"
        return hashlib.sha256(payload.encode()).hexdigest()

    return [str(row["text"]) for row in sorted(remaining, key=order_key)]


def _coherence_prompt(texts: list[str]) -> str:
    excerpts = "\n\n".join(
        f"EXCERPT {label}:\n{text}"
        for label, text in zip(("A", "B", "C", "D"), texts, strict=True)
    )
    return f"""Rate how coherently these four excerpts share one narrow content-based semantic or structural pattern. Ignore writing quality, length, formatting, language, named entities, and document source. Do not infer or discuss any missing excerpt.

Use this fixed scale:
0 = no meaningful shared pattern
1 = weak or mostly superficial similarity
2 = partial pattern with substantial exceptions
3 = clear shared pattern across all four
4 = very strong and specific shared pattern across all four

{excerpts}

Return exactly one JSON object and no markdown:
{{"coherence_score": 0|1|2|3|4, "reason": "brief explanation"}}"""


def _parse_score(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match is None:
        return {"valid": False, "error": "no_json_object", "raw": text}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return {"valid": False, "error": f"json_decode:{error.msg}", "raw": text}
    score = payload.get("coherence_score")
    reason = payload.get("reason")
    if not isinstance(score, int) or not 0 <= score <= 4:
        return {"valid": False, "error": "invalid_coherence_score", "raw": text}
    if not isinstance(reason, str) or not reason.strip():
        return {"valid": False, "error": "invalid_reason", "raw": text}
    return {
        "valid": True,
        "coherence_score": score,
        "reason": reason.strip(),
        "raw": text,
    }


def _generate(
    prompts: list[str], contract: IntruderLeaveOneOutJudgeContract
) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract.judge_model_id,
        revision=contract.judge_revision,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        contract.judge_model_id,
        revision=contract.judge_revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    outputs = []
    for start in range(0, len(prompts), contract.batch_size):
        batch = prompts[start : start + contract.batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are an independent evaluator following a fixed protocol.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=contract.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return outputs


def _select_intruder(
    candidate_rows: list[dict[str, Any]],
    contract: IntruderLeaveOneOutJudgeContract,
) -> int | None:
    if len(candidate_rows) != 5 or not all(row["parsed"]["valid"] for row in candidate_rows):
        return None
    ranked = sorted(
        (
            (int(row["parsed"]["coherence_score"]), int(row["candidate_position"]))
            for row in candidate_rows
        ),
        reverse=True,
    )
    top_score, top_position = ranked[0]
    second_score = ranked[1][0]
    if top_score < contract.minimum_top_coherence:
        return None
    if top_score - second_score < contract.minimum_top_margin:
        return None
    return top_position


def _summary(
    rows: list[dict[str, Any]],
    *,
    prompt_count: int,
    contract: IntruderLeaveOneOutJudgeContract,
) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_parent"]]
    positive = [row for row in valid if row["control_kind"] == "positive"]
    negative = [row for row in valid if row["control_kind"] == "negative"]
    scientific = [row for row in valid if row["control_kind"] is None]
    positive_accuracy = sum(
        row["predicted_position"] == row["correct_position"] for row in positive
    ) / max(1, len(positive))
    negative_abstention = sum(
        row["predicted_position"] is None for row in negative
    ) / max(1, len(negative))
    invalid_parent_rate = 1 - len(valid) / max(1, len(rows))
    positive_positions = Counter(
        int(row["predicted_position"])
        for row in positive
        if row["predicted_position"] is not None
    )
    scientific_positions = Counter(
        int(row["predicted_position"])
        for row in scientific
        if row["predicted_position"] is not None
    )
    positive_max_share = max(positive_positions.values(), default=0) / max(
        1, sum(positive_positions.values())
    )
    scientific_max_share = max(scientific_positions.values(), default=0) / max(
        1, sum(scientific_positions.values())
    )
    passed = (
        positive_accuracy >= contract.minimum_positive_control_accuracy
        and negative_abstention >= contract.minimum_negative_control_abstention_rate
        and invalid_parent_rate <= contract.maximum_invalid_parent_rate
        and positive_max_share <= contract.maximum_position_share
        and scientific_max_share <= contract.maximum_position_share
    )
    return {
        "status": "JUDGE_VALIDATED" if passed else "JUDGE_VALIDATION_FAILED_EXPLORATORY_ONLY",
        "parent_rows": len(rows),
        "prompt_rows": prompt_count,
        "valid_parent_rows": len(valid),
        "invalid_parent_rate": invalid_parent_rate,
        "positive_control_accuracy": positive_accuracy,
        "negative_control_abstention_rate": negative_abstention,
        "positive_control_position_histogram": dict(sorted(positive_positions.items())),
        "positive_control_maximum_position_share": positive_max_share,
        "scientific_position_histogram": dict(sorted(scientific_positions.items())),
        "scientific_maximum_position_share": scientific_max_share,
        "scientific_abstention_rate": sum(
            row["predicted_position"] is None for row in scientific
        )
        / max(1, len(scientific)),
        "thresholds": {
            "minimum_positive_control_accuracy": contract.minimum_positive_control_accuracy,
            "minimum_negative_control_abstention_rate": contract.minimum_negative_control_abstention_rate,
            "maximum_invalid_parent_rate": contract.maximum_invalid_parent_rate,
            "maximum_position_share": contract.maximum_position_share,
            "minimum_top_coherence": contract.minimum_top_coherence,
            "minimum_top_margin": contract.minimum_top_margin,
        },
    }


def main() -> None:
    args = _args()
    contract = IntruderLeaveOneOutJudgeContract.load(args.contract)
    if "qwen" in contract.judge_model_id.lower() or contract.paid_api:
        raise RuntimeError("judge contract violates the non-Qwen/local-only restriction")
    tasks = _jsonl(args.tasks)
    answers = {row["trial_id"]: row for row in _jsonl(args.answer_key)}
    if {row["trial_id"] for row in tasks} != set(answers):
        raise ValueError("Intruder tasks and answer key differ")
    parents: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if not args.controls_only:
        parents.extend((task, answers[task["trial_id"]]) for task in tasks)
    control_tasks, control_answers = _control_tasks(contract)  # type: ignore[arg-type]
    parents.extend(zip(control_tasks, control_answers, strict=True))

    prompts = []
    metadata = []
    for task, answer in parents:
        for candidate_position in range(1, 6):
            ordered = _ordered_remaining(task, candidate_position)
            prompts.append(_coherence_prompt(ordered))
            metadata.append(
                {
                    "trial_id": task["trial_id"],
                    "control_kind": task.get("control_kind"),
                    "correct_position": answer["intruder_position"],
                    "candidate_position": candidate_position,
                    "presented_text_sha256": [
                        hashlib.sha256(text.encode()).hexdigest() for text in ordered
                    ],
                }
            )

    started = time.perf_counter()
    raw_outputs = _generate(prompts, contract)
    candidate_groups: dict[str, list[dict[str, Any]]] = {}
    parent_metadata: dict[str, dict[str, Any]] = {}
    for meta, raw in zip(metadata, raw_outputs, strict=True):
        trial_id = str(meta["trial_id"])
        candidate_groups.setdefault(trial_id, []).append(
            {
                "candidate_position": meta["candidate_position"],
                "presented_text_sha256": meta["presented_text_sha256"],
                "parsed": _parse_score(raw),
            }
        )
        parent_metadata[trial_id] = meta

    rows = []
    for trial_id, candidate_rows in candidate_groups.items():
        meta = parent_metadata[trial_id]
        valid_parent = len(candidate_rows) == 5 and all(
            row["parsed"]["valid"] for row in candidate_rows
        )
        rows.append(
            {
                "trial_id": trial_id,
                "control_kind": meta["control_kind"],
                "correct_position": meta["correct_position"],
                "valid_parent": valid_parent,
                "predicted_position": _select_intruder(candidate_rows, contract),
                "candidates": sorted(
                    candidate_rows, key=lambda row: int(row["candidate_position"])
                ),
            }
        )

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite judge output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "judge_results.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    summary = _summary(rows, prompt_count=len(prompts), contract=contract)
    _write_json(args.output_dir / "judge_summary.json", summary)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "LOCAL_INTRUDER_LOO_JUDGING_COMPLETE",
            "scientific_interpretation_status": summary["status"],
            "execution_mode": "controls_only" if args.controls_only else "full",
            "judge_model_id": contract.judge_model_id,
            "judge_revision": contract.judge_revision,
            "local_files_only": True,
            "paid_api": False,
            "dtype": contract.dtype,
            "decoding": contract.decoding,
            "prompt_version": contract.prompt_version,
            "control_version": contract.control_version,
            "selection_rule": {
                "minimum_top_coherence": contract.minimum_top_coherence,
                "minimum_top_margin": contract.minimum_top_margin,
            },
            "contract_sha256": sha256_file(args.contract),
            "contract_fingerprint": contract.fingerprint(),
            "tasks_sha256": sha256_file(args.tasks),
            "answer_key_sha256": sha256_file(args.answer_key),
            "results_sha256": sha256_file(args.output_dir / "judge_results.jsonl"),
            "git_sha": _git_sha(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

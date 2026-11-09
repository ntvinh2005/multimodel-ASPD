"""Run the pinned local non-Qwen judge on blinded, position-balanced Intruder tasks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import (
    IntruderJudgeContract,
    IntruderJudgeV2Contract,
)

JudgeContract = IntruderJudgeContract | IntruderJudgeV2Contract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Run only the synthetic controls for fail-closed judge preflight.",
    )
    return parser.parse_args()


def _load_contract(path: Path) -> JudgeContract:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    family = payload.get("experiment_family")
    if family == "aspd_v3_zscore_intruder_judge_v1":
        return IntruderJudgeContract.model_validate(payload)
    if family == "aspd_v3_zscore_intruder_judge_v2":
        return IntruderJudgeV2Contract.model_validate(payload)
    raise ValueError(f"unsupported judge experiment_family: {family!r}")


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


def _rotate_task(
    task: dict[str, Any], answer: dict[str, Any], rotation: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    examples = list(task["examples"])
    if len(examples) != 5 or not 0 <= rotation < 5:
        raise ValueError("Intruder cyclic rotation requires five examples")
    texts = [str(row["text"]) for row in examples]
    rotated = texts[rotation:] + texts[:rotation]
    correct = answer["intruder_position"]
    rotated_correct = (
        None if correct is None else ((int(correct) - 1 - rotation) % 5) + 1
    )
    trial_id = f"{task['trial_id']}-r{rotation}"
    control_kind = task.get("control_kind")
    return (
        {
            "trial_id": trial_id,
            "base_trial_id": task["trial_id"],
            "rotation": rotation,
            "examples": [
                {"position": index + 1, "text": text}
                for index, text in enumerate(rotated)
            ],
            "control_kind": control_kind,
        },
        {
            **answer,
            "trial_id": trial_id,
            "base_trial_id": task["trial_id"],
            "rotation": rotation,
            "intruder_position": rotated_correct,
            "control_kind": control_kind,
        },
    )


_CONTROL_THEMES = (
    ("astronomy", "planet orbit", "stellar spectrum", "galaxy cluster", "lunar crater", "bread recipe"),
    ("botany", "leaf chlorophyll", "root growth", "flower pollen", "plant stem", "database index"),
    ("music", "piano chord", "violin bow", "rhythm pattern", "vocal melody", "volcanic rock"),
    ("mathematics", "prime number", "matrix eigenvalue", "geometric proof", "integral equation", "garden shovel"),
    ("programming", "function argument", "compiler error", "memory allocation", "unit test", "ocean tide"),
)


_NEGATIVE_CONTROL_PARAPHRASES = (
    (
        "astronomy",
        (
            "A planet follows an orbit around its star.",
            "A planet travels along a path around the star it belongs to.",
            "The planet revolves around its host star.",
            "An orbit carries a planet around its star.",
            "A planet circles the star in its planetary system.",
        ),
    ),
    (
        "botany",
        (
            "Leaves use chlorophyll to capture light for photosynthesis.",
            "Chlorophyll in leaves absorbs light used in photosynthesis.",
            "For photosynthesis, a leaf captures light with chlorophyll.",
            "Leaf chlorophyll collects light energy for photosynthesis.",
            "Photosynthesis uses light absorbed by chlorophyll in a leaf.",
        ),
    ),
    (
        "music",
        (
            "A chord combines multiple musical notes at the same time.",
            "Several notes sounding together form a musical chord.",
            "A musical chord consists of multiple simultaneous notes.",
            "Multiple notes played together create a chord.",
            "A chord is produced when several musical notes sound together.",
        ),
    ),
    (
        "mathematics",
        (
            "A prime number has exactly two positive divisors.",
            "Exactly two positive factors divide a prime number.",
            "A number is prime when it has two positive divisors.",
            "Prime numbers are divisible by precisely two positive integers.",
            "Having exactly two positive factors defines a prime number.",
        ),
    ),
    (
        "programming",
        (
            "A function receives input values through its parameters.",
            "Parameters provide input values to a function.",
            "Input values enter a function by way of parameters.",
            "A function's parameters carry values into the function.",
            "Values are passed into a function using parameters.",
        ),
    ),
)


def _control_tasks(
    contract: JudgeContract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(contract.control_seed)
    tasks = []
    answers = []
    for index in range(contract.positive_controls):
        theme, *phrases = _CONTROL_THEMES[index % len(_CONTROL_THEMES)]
        intruder = phrases[-1]
        position = int(rng.integers(0, 5))
        examples = phrases[:4]
        examples.insert(position, intruder)
        tasks.append(
            {
                "trial_id": f"positive-{index:03d}",
                "base_trial_id": f"positive-{index:03d}",
                "rotation": 0,
                "examples": [
                    {"position": slot + 1, "text": text}
                    for slot, text in enumerate(examples)
                ],
                "control_kind": "positive",
            }
        )
        answers.append(
            {
                "trial_id": f"positive-{index:03d}",
                "base_trial_id": f"positive-{index:03d}",
                "rotation": 0,
                "intruder_position": position + 1,
                "control_kind": "positive",
                "theme": theme,
            }
        )
    for index in range(contract.negative_controls):
        if isinstance(contract, IntruderJudgeV2Contract):
            theme, paraphrases = _NEGATIVE_CONTROL_PARAPHRASES[
                index % len(_NEGATIVE_CONTROL_PARAPHRASES)
            ]
            examples = list(paraphrases)
        else:
            theme, *phrases = _CONTROL_THEMES[index % len(_CONTROL_THEMES)]
            examples = [f"{phrase} example {index}" for phrase in phrases[:4]]
            examples.append(f"{theme} concept example {index}")
        rng.shuffle(examples)
        tasks.append(
            {
                "trial_id": f"negative-{index:03d}",
                "base_trial_id": f"negative-{index:03d}",
                "rotation": 0,
                "examples": [
                    {"position": slot + 1, "text": text}
                    for slot, text in enumerate(examples)
                ],
                "control_kind": "negative",
            }
        )
        answers.append(
            {
                "trial_id": f"negative-{index:03d}",
                "base_trial_id": f"negative-{index:03d}",
                "rotation": 0,
                "intruder_position": None,
                "control_kind": "negative",
                "theme": theme,
            }
        )
    return tasks, answers


def _prompt(task: dict[str, Any], contract: JudgeContract) -> str:
    examples = "\n\n".join(
        f"EXCERPT {row['position']}:\n{row['text']}" for row in task["examples"]
    )
    if isinstance(contract, IntruderJudgeV2Contract):
        instructions = """Determine the narrowest content-based pattern that covers at least four excerpts. Select an intruder only when exactly one excerpt clearly violates that pattern.

Do not manufacture an intruder from wording, length, formatting, specificity versus generality, named entities, or an arbitrary subgroup. If all five express the same underlying idea, if multiple answers are comparably plausible, or if the distinction is weak, return null. Confidence 3 requires a clear four-versus-one semantic separation; confidence 0 requires null."""
    else:
        instructions = """Four excerpts should share a specific semantic or structural pattern and one may be an intruder. Identify the intruder using content, not writing quality, length, language, formatting, or document source. If no defensible unique intruder exists, return null."""
    return f"""{instructions}

{examples}

Return exactly one JSON object and no markdown:
{{"intruder_position": 1|2|3|4|5|null, "confidence": 0|1|2|3, "reason": "brief explanation"}}
Confidence 0 means no defensible unique answer; 1 weak; 2 moderate; 3 strong."""


def _parse_response(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match is None:
        return {"valid": False, "error": "no_json_object", "raw": text}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return {"valid": False, "error": f"json_decode:{error.msg}", "raw": text}
    position = payload.get("intruder_position")
    confidence = payload.get("confidence")
    reason = payload.get("reason")
    if position is not None and (not isinstance(position, int) or not 1 <= position <= 5):
        return {"valid": False, "error": "invalid_position", "raw": text}
    if not isinstance(confidence, int) or not 0 <= confidence <= 3:
        return {"valid": False, "error": "invalid_confidence", "raw": text}
    if not isinstance(reason, str) or not reason.strip():
        return {"valid": False, "error": "invalid_reason", "raw": text}
    return {
        "valid": True,
        "intruder_position": position,
        "confidence": confidence,
        "reason": reason.strip(),
        "raw": text,
    }


def _generate(
    tasks: list[dict[str, Any]], contract: JudgeContract
) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs: dict[str, Any] = {
        "revision": contract.judge_revision,
        "local_files_only": True,
    }
    if contract.judge_model_id == "mistralai/Mistral-Nemo-Instruct-2407":
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        contract.judge_model_id,
        **tokenizer_kwargs,
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
    for start in range(0, len(tasks), contract.batch_size):
        batch = tasks[start : start + contract.batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are an independent evaluator following a fixed protocol.",
                    },
                    {"role": "user", "content": _prompt(task, contract)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for task in batch
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


def _canonical_position(position: int | None, rotation: int) -> int | None:
    if position is None:
        return None
    return ((int(position) - 1 + rotation) % 5) + 1


def _summary(rows: list[dict[str, Any]], contract: JudgeContract) -> dict[str, Any]:
    valid = [row for row in rows if row["parsed"]["valid"]]
    positive = [row for row in valid if row["control_kind"] == "positive"]
    negative = [row for row in valid if row["control_kind"] == "negative"]
    scientific = [row for row in valid if row["control_kind"] is None]
    positive_accuracy = (
        sum(row["parsed"]["intruder_position"] == row["correct_position"] for row in positive)
        / len(positive)
        if positive
        else 0.0
    )
    negative_safe = (
        sum(
            row["parsed"]["intruder_position"] is None
            or int(row["parsed"]["confidence"]) <= 1
            for row in negative
        )
        / len(negative)
        if negative
        else 0.0
    )
    invalid_rate = 1 - len(valid) / len(rows)
    positions = Counter(
        int(row["parsed"]["intruder_position"])
        for row in scientific
        if row["parsed"]["intruder_position"] is not None
    )
    maximum_position_share = max(positions.values(), default=0) / max(1, sum(positions.values()))
    positive_positions = Counter(
        int(row["parsed"]["intruder_position"])
        for row in positive
        if row["parsed"]["intruder_position"] is not None
    )
    positive_maximum_position_share = max(
        positive_positions.values(), default=0
    ) / max(1, sum(positive_positions.values()))
    controls_pass = (
        positive_accuracy >= contract.minimum_positive_control_accuracy
        and negative_safe
        >= contract.minimum_negative_control_abstain_or_low_confidence_rate
        and invalid_rate <= contract.maximum_invalid_rate
        and maximum_position_share <= contract.maximum_position_share
    )
    if isinstance(contract, IntruderJudgeV2Contract):
        controls_pass = (
            controls_pass
            and positive_maximum_position_share <= contract.maximum_position_share
        )
    summary: dict[str, Any] = {
        "status": "JUDGE_VALIDATED" if controls_pass else "JUDGE_VALIDATION_FAILED_EXPLORATORY_ONLY",
        "rows": len(rows),
        "valid_rows": len(valid),
        "invalid_rate": invalid_rate,
        "positive_control_accuracy": positive_accuracy,
        "negative_control_abstain_or_low_confidence_rate": negative_safe,
        "positive_control_position_histogram": dict(
            sorted(positive_positions.items())
        ),
        "positive_control_maximum_position_share": positive_maximum_position_share,
        "scientific_position_histogram": dict(sorted(positions.items())),
        "scientific_maximum_position_share": maximum_position_share,
        "thresholds": {
            "minimum_positive_control_accuracy": contract.minimum_positive_control_accuracy,
            "minimum_negative_control_abstain_or_low_confidence_rate": contract.minimum_negative_control_abstain_or_low_confidence_rate,
            "maximum_invalid_rate": contract.maximum_invalid_rate,
            "maximum_position_share": contract.maximum_position_share,
        },
    }
    if isinstance(contract, IntruderJudgeV2Contract) and scientific:
        parent_predictions: dict[str, list[int]] = {}
        parent_correct: dict[str, int] = {}
        for row in scientific:
            base_trial_id = str(row["base_trial_id"])
            rotation = int(row["rotation"])
            predicted = _canonical_position(
                row["parsed"]["intruder_position"], rotation
            )
            correct = _canonical_position(row["correct_position"], rotation)
            if predicted is not None:
                parent_predictions.setdefault(base_trial_id, []).append(predicted)
            if correct is not None:
                parent_correct[base_trial_id] = correct
        consensus_eligible = 0
        consensus_correct = 0
        unanimous = 0
        strengths: list[int] = []
        for base_trial_id, predictions in parent_predictions.items():
            counts = Counter(predictions)
            top_count = max(counts.values(), default=0)
            strengths.append(top_count)
            if top_count == contract.cyclic_rotations:
                unanimous += 1
            winners = [position for position, count in counts.items() if count == top_count]
            if top_count < contract.minimum_rotation_consensus or len(winners) != 1:
                continue
            consensus_eligible += 1
            if winners[0] == parent_correct[base_trial_id]:
                consensus_correct += 1
        parent_count = len(parent_correct)
        summary["rotation_consensus"] = {
            "parent_count": parent_count,
            "eligible_parent_count": consensus_eligible,
            "eligible_parent_fraction": consensus_eligible / max(1, parent_count),
            "eligible_parent_accuracy": consensus_correct
            / max(1, consensus_eligible),
            "unanimous_parent_count": unanimous,
            "mean_maximum_agreement": sum(strengths) / max(1, len(strengths)),
            "minimum_votes": contract.minimum_rotation_consensus,
        }
    return summary


def main() -> None:
    args = _args()
    contract = _load_contract(args.contract)
    if "qwen" in contract.judge_model_id.lower() or contract.paid_api:
        raise RuntimeError("judge contract violates the non-Qwen/local-only restriction")
    if args.controls_only and not isinstance(contract, IntruderJudgeV2Contract):
        raise ValueError("controls-only preflight requires the v2 judge contract")
    tasks = _jsonl(args.tasks)
    answers = {row["trial_id"]: row for row in _jsonl(args.answer_key)}
    if {row["trial_id"] for row in tasks} != set(answers):
        raise ValueError("Intruder tasks and answer key differ")
    expanded_tasks = []
    expanded_answers = []
    if not args.controls_only:
        for task in tasks:
            answer = answers[task["trial_id"]]
            for rotation in range(contract.cyclic_rotations):
                rotated_task, rotated_answer = _rotate_task(task, answer, rotation)
                expanded_tasks.append(rotated_task)
                expanded_answers.append(rotated_answer)
    control_tasks, control_answers = _control_tasks(contract)
    if isinstance(contract, IntruderJudgeV2Contract):
        for task, answer in zip(control_tasks, control_answers, strict=True):
            for rotation in range(contract.control_rotations):
                rotated_task, rotated_answer = _rotate_task(task, answer, rotation)
                expanded_tasks.append(rotated_task)
                expanded_answers.append(rotated_answer)
    else:
        expanded_tasks.extend(control_tasks)
        expanded_answers.extend(control_answers)
    started = time.perf_counter()
    raw_outputs = _generate(expanded_tasks, contract)
    rows = []
    for task, answer, raw in zip(
        expanded_tasks, expanded_answers, raw_outputs, strict=True
    ):
        rows.append(
            {
                "trial_id": task["trial_id"],
                "base_trial_id": task["base_trial_id"],
                "rotation": task["rotation"],
                "control_kind": task["control_kind"],
                "correct_position": answer["intruder_position"],
                "parsed": _parse_response(raw),
            }
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite judge output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "judge_results.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    summary = _summary(rows, contract)
    _write_json(args.output_dir / "judge_summary.json", summary)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "LOCAL_INTRUDER_JUDGING_COMPLETE",
            "scientific_interpretation_status": summary["status"],
            "execution_mode": "controls_only" if args.controls_only else "full",
            "judge_model_id": contract.judge_model_id,
            "judge_revision": contract.judge_revision,
            "local_files_only": True,
            "paid_api": False,
            "dtype": contract.dtype,
            "decoding": contract.decoding,
            "cyclic_rotations": contract.cyclic_rotations,
            "control_rotations": getattr(contract, "control_rotations", 1),
            "prompt_version": getattr(contract, "prompt_version", "legacy_v1"),
            "control_version": getattr(contract, "control_version", "legacy_v1"),
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

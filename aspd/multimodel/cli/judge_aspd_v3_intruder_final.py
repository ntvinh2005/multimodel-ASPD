"""Run the calibration-gated, position/label-balanced final Intruder judge."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binom, chisquare

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import FinalIntruderJudgeContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "development", "final"), required=True)
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


def _expand_task(
    task: dict[str, Any], rotation: int, labels: tuple[str, ...]
) -> dict[str, Any]:
    examples = list(task["examples"])
    if len(examples) != 5 or not 0 <= rotation < 5 or len(labels) != 5:
        raise ValueError("balanced Intruder expansion requires exactly five candidates")
    order = list(range(rotation, 5)) + list(range(rotation))
    displayed = []
    for displayed_index, canonical_index in enumerate(order):
        label = labels[(displayed_index + 2 * rotation) % 5]
        displayed.append(
            {
                "label": label,
                "text": str(examples[canonical_index]["text"]),
                "canonical_position": canonical_index + 1,
                "displayed_position": displayed_index + 1,
            }
        )
    return {
        "trial_id": f"{task['trial_id']}-r{rotation}",
        "parent_trial_id": task["trial_id"],
        "rotation": rotation,
        "control_kind": task.get("control_kind"),
        "examples": displayed,
    }


def _prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    candidates = "\n\n".join(
        f"Candidate {row['label']}:\n{row['text']}" for row in task["examples"]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful semantic-coherence evaluator. Four candidates share the "
                "most specific coherent pattern and exactly one is the intruder. Ignore order, "
                "length, formatting, writing style, and superficial lexical overlap. Return "
                "exactly one candidate label. Output only a JSON object with keys "
                "intruder_id and reason; intruder_id must be one of K, L, M, N, P."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{candidates}\n\nIdentify the one semantic intruder. Return exactly the "
                "required JSON object and no additional text."
            ),
        },
    ]


def _parse_response(text: str, labels: tuple[str, ...]) -> dict[str, Any]:
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if len(matches) != 1:
        return {"valid": False, "intruder_id": None, "reason": "json_object_count"}
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return {"valid": False, "intruder_id": None, "reason": "invalid_json"}
    if set(payload) != {"intruder_id", "reason"}:
        return {"valid": False, "intruder_id": None, "reason": "unexpected_keys"}
    label = payload["intruder_id"]
    reason = payload["reason"]
    if label not in labels or not isinstance(reason, str) or not reason.strip():
        return {"valid": False, "intruder_id": None, "reason": "invalid_fields"}
    return {"valid": True, "intruder_id": label, "reason": reason.strip()}


def _snapshot_path(contract: FinalIntruderJudgeContract) -> Path:
    hf_home = Path(os.environ["HF_HOME"])
    snapshot = (
        hf_home
        / "hub"
        / f"models--{contract.judge_model_id.replace('/', '--')}"
        / "snapshots"
        / contract.judge_revision
    )
    required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"pinned local judge snapshot is incomplete: {missing}")
    return snapshot


def _load_model(contract: FinalIntruderJudgeContract) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = _snapshot_path(contract)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=contract.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=contract.local_files_only,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    return tokenizer, model


def _generate(
    *,
    tokenizer: Any,
    model: Any,
    tasks: list[dict[str, Any]],
    contract: FinalIntruderJudgeContract,
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


def _position_diagnostics(
    rows: list[dict[str, Any]], *, alpha: float
) -> dict[str, Any]:
    valid_positions = [
        int(row["predicted_displayed_position"])
        for row in rows
        if row["valid"] and row["predicted_displayed_position"] is not None
    ]
    counts = np.asarray(
        [valid_positions.count(position) for position in range(1, 6)], dtype=np.int64
    )
    n = int(counts.sum())
    if n:
        chi = chisquare(counts, f_exp=np.full(5, n / 5))
        maximum_allowed = int(binom.ppf(1 - alpha / 5, n, 0.2))
        passed = bool(int(counts.max()) <= maximum_allowed and float(chi.pvalue) >= alpha)
    else:
        chi = None
        maximum_allowed = 0
        passed = False
    return {
        "valid_predictions": n,
        "counts": {str(position): int(counts[position - 1]) for position in range(1, 6)},
        "shares": {
            str(position): float(counts[position - 1] / n) if n else 0.0
            for position in range(1, 6)
        },
        "chi_square_statistic": float(chi.statistic) if chi is not None else None,
        "chi_square_pvalue": float(chi.pvalue) if chi is not None else None,
        "familywise_alpha": alpha,
        "bonferroni_binomial_maximum_allowed_count": maximum_allowed,
        "maximum_observed_count": int(counts.max()) if n else 0,
        "passed": passed,
    }


def _score(
    expanded: list[dict[str, Any]],
    raw_outputs: list[str],
    answers: list[dict[str, Any]],
    contract: FinalIntruderJudgeContract,
) -> list[dict[str, Any]]:
    answer_by_id = {str(row["trial_id"]): row for row in answers}
    if len(answer_by_id) != len(answers):
        raise ValueError("answer key contains duplicate parent trial IDs")
    rows = []
    for task, raw in zip(expanded, raw_outputs, strict=True):
        parent_id = str(task["parent_trial_id"])
        if parent_id not in answer_by_id:
            raise ValueError(f"answer key is missing {parent_id}")
        answer = answer_by_id[parent_id]
        parsed = _parse_response(raw, contract.candidate_labels)
        lookup = {str(row["label"]): row for row in task["examples"]}
        predicted = lookup.get(str(parsed["intruder_id"])) if parsed["valid"] else None
        correct_canonical = answer["intruder_position"]
        correct_row = next(
            (
                row
                for row in task["examples"]
                if row["canonical_position"] == correct_canonical
            ),
            None,
        )
        correct = bool(
            parsed["valid"]
            and correct_canonical is not None
            and predicted is not None
            and predicted["canonical_position"] == correct_canonical
        )
        rows.append(
            {
                "trial_id": task["trial_id"],
                "parent_trial_id": parent_id,
                "rotation": task["rotation"],
                "control_kind": task.get("control_kind"),
                "raw_output": raw,
                "valid": bool(parsed["valid"]),
                "invalid_reason": None if parsed["valid"] else parsed["reason"],
                "predicted_label": parsed["intruder_id"],
                "predicted_displayed_position": (
                    int(predicted["displayed_position"]) if predicted else None
                ),
                "predicted_canonical_position": (
                    int(predicted["canonical_position"]) if predicted else None
                ),
                "correct_label": correct_row["label"] if correct_row else None,
                "correct_displayed_position": (
                    int(correct_row["displayed_position"]) if correct_row else None
                ),
                "correct_canonical_position": correct_canonical,
                "correct": correct,
                "reason": parsed["reason"] if parsed["valid"] else None,
            }
        )
    return rows


def _parent_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_trial_id"])].append(row)
    shares = {}
    for parent, values in grouped.items():
        predictions = [
            int(row["predicted_canonical_position"])
            for row in values
            if row["valid"] and row["predicted_canonical_position"] is not None
        ]
        shares[parent] = (
            max(Counter(predictions).values()) / len(values) if predictions else 0.0
        )
    return {
        "parent_count": len(shares),
        "mean_modal_canonical_share": float(np.mean(list(shares.values()))) if shares else 0.0,
        "fraction_at_least_0_8": (
            float(np.mean([share >= 0.8 for share in shares.values()])) if shares else 0.0
        ),
    }


def _summary(
    *,
    mode: str,
    rows: list[dict[str, Any]],
    replay_exact: bool | None,
    contract: FinalIntruderJudgeContract,
) -> dict[str, Any]:
    positive = [row for row in rows if row["control_kind"] == "positive"]
    negative = [row for row in rows if row["control_kind"] == "negative"]
    valid_rate = float(np.mean([row["valid"] for row in rows])) if rows else 0.0
    position = _position_diagnostics(
        positive if mode == "calibration" else rows,
        alpha=contract.position_bias_familywise_alpha,
    )
    label_counts = Counter(row["predicted_label"] for row in rows if row["valid"])
    conditional_accuracy = {}
    scored = positive if mode == "calibration" else rows
    for position_value in range(1, 6):
        subset = [
            row for row in scored if row["correct_displayed_position"] == position_value
        ]
        conditional_accuracy[str(position_value)] = (
            float(np.mean([row["correct"] for row in subset])) if subset else None
        )
    finite_accuracies = [value for value in conditional_accuracy.values() if value is not None]
    spread = max(finite_accuracies) - min(finite_accuracies) if finite_accuracies else 1.0
    positive_accuracy = (
        float(np.mean([row["correct"] for row in positive])) if positive else None
    )
    positive_consistency = _parent_consistency(positive) if positive else None
    negative_consistency = _parent_consistency(negative) if negative else None
    gates = None
    status = "SCIENTIFIC_JUDGE_RUN_COMPLETE"
    if mode == "calibration":
        gates = {
            "positive_accuracy": bool(
                positive_accuracy is not None
                and positive_accuracy >= contract.minimum_positive_control_accuracy
            ),
            "valid_response_rate": valid_rate >= contract.minimum_valid_response_rate,
            "position_bias": position["passed"],
            "correct_position_accuracy_spread": (
                spread <= contract.maximum_correct_position_accuracy_spread
            ),
            "positive_parent_canonical_consistency": bool(
                positive_consistency is not None
                and positive_consistency["fraction_at_least_0_8"]
                >= contract.minimum_positive_parent_canonical_consistency
            ),
            "exact_deterministic_replay": bool(replay_exact),
        }
        status = "JUDGE_VALIDATED" if all(gates.values()) else "JUDGE_VALIDATION_FAILED"
    elif not position["passed"] or spread > contract.maximum_correct_position_accuracy_spread:
        status = "SCIENTIFIC_POSITION_DIAGNOSTIC_FAILED"
    return {
        "schema_version": 4,
        "mode": mode,
        "status": status,
        "rotations": len(rows),
        "parent_tasks": len({row["parent_trial_id"] for row in rows}),
        "valid_response_rate": valid_rate,
        "invalid_responses": sum(not row["valid"] for row in rows),
        "positive_control_accuracy": positive_accuracy,
        "positive_parent_consistency": positive_consistency,
        "negative_parent_consistency_diagnostic": negative_consistency,
        "prediction_position": position,
        "prediction_label_counts": dict(sorted(label_counts.items())),
        "accuracy_conditioned_on_correct_position": conditional_accuracy,
        "correct_position_accuracy_spread": spread,
        "deterministic_replay_exact": replay_exact,
        "calibration_gates": gates,
    }


def _validate_calibration(path: Path, contract: FinalIntruderJudgeContract) -> dict[str, Any]:
    calibration = _json(path)
    if calibration.get("status") != "JUDGE_VALIDATED":
        raise ValueError("final/development scoring requires a passing calibration")
    if calibration.get("contract_fingerprint") != contract.fingerprint():
        raise ValueError("calibration used a different judge contract")
    return calibration


def main() -> None:
    args = _args()
    evaluation_git_sha = _git_sha()
    contract = FinalIntruderJudgeContract.load(args.contract)
    if "qwen" in contract.judge_model_id.casefold():
        raise ValueError("Qwen judges are forbidden")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.mode != "calibration":
        if args.calibration_summary is None:
            raise ValueError("development/final mode requires --calibration-summary")
        _validate_calibration(args.calibration_summary, contract)
    tasks = _jsonl(args.tasks)
    snapshot = _snapshot_path(contract)
    if len({str(row["trial_id"]) for row in tasks}) != len(tasks):
        raise ValueError("task file contains duplicate parent trial IDs")
    expanded = [
        _expand_task(task, rotation, contract.candidate_labels)
        for task in tasks
        for rotation in range(contract.cyclic_rotations)
    ]
    args.output_dir.mkdir(parents=True)
    _write_jsonl(
        args.output_dir / "rendered_prompt_messages.jsonl",
        [
            {
                "trial_id": task["trial_id"],
                "parent_trial_id": task["parent_trial_id"],
                "rotation": task["rotation"],
                "messages": _prompt(task),
            }
            for task in expanded
        ],
    )
    started = time.time()
    tokenizer, model = _load_model(contract)
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
    # Load the isolated answer key only after generation has completed.
    answers = _jsonl(args.answer_key)
    rows = _score(expanded, raw_outputs, answers, contract)
    summary = _summary(
        mode=args.mode,
        rows=rows,
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
    _write_jsonl(args.output_dir / "rotation_results.jsonl", rows)
    _write_json(args.output_dir / "judge_summary.json", summary)
    manifest = {
        "schema_version": 4,
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
        "results_sha256": sha256_file(args.output_dir / "rotation_results.jsonl"),
        "rendered_prompt_messages_sha256": sha256_file(
            args.output_dir / "rendered_prompt_messages.jsonl"
        ),
        "summary_sha256": sha256_file(args.output_dir / "judge_summary.json"),
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

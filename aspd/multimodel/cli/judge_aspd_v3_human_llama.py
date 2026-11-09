"""Run a raw, single-presentation Llama-70B Intruder score on human tasks."""

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

from aspd.multimodel.v3.human_interpretability import (
    HumanLlamaJudgeContract,
    canonical_json_bytes,
    sha256_file,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dataset", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _text(example: dict[str, Any]) -> str:
    parts = example["expanded"]
    return (
        ("… " if parts["truncated_left"] else "")
        + str(parts["prefix"])
        + str(parts["anchor"])
        + str(parts["suffix"])
        + (" …" if parts["truncated_right"] else "")
    )


def _messages(item: dict[str, Any]) -> list[dict[str, str]]:
    excerpts = "\n\n".join(
        f"EXCERPT {example['candidate_id']}:\n{_text(example)}"
        for example in item["examples"]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an independent semantic feature evaluator. Four excerpts are contexts "
                "for the same learned feature and exactly one is an intruder. Use the most "
                "specific semantic or structural pattern shared by four excerpts. Ignore order, "
                "length, formatting, document source, and writing quality. You must choose one."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{excerpts}\n\nReturn exactly one JSON object and no markdown: "
                '{"intruder_position": 1|2|3|4|5, "reason": "brief explanation"}'
            ),
        },
    ]


def _parse(text: str) -> dict[str, Any]:
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if len(matches) != 1:
        return {"valid": False, "intruder_position": None, "reason": "json_object_count", "raw": text}
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return {"valid": False, "intruder_position": None, "reason": "invalid_json", "raw": text}
    if set(payload) != {"intruder_position", "reason"}:
        return {"valid": False, "intruder_position": None, "reason": "unexpected_keys", "raw": text}
    position = payload["intruder_position"]
    reason = payload["reason"]
    if not isinstance(position, int) or not 1 <= position <= 5 or not isinstance(reason, str) or not reason.strip():
        return {"valid": False, "intruder_position": None, "reason": "invalid_fields", "raw": text}
    return {"valid": True, "intruder_position": position, "reason": reason.strip(), "raw": text}


def _snapshot(contract: HumanLlamaJudgeContract) -> Path:
    root = (
        Path(os.environ["HF_HOME"])
        / "hub"
        / f"models--{contract.judge_model_id.replace('/', '--')}"
        / "snapshots"
        / contract.judge_revision
    )
    required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"pinned local judge snapshot is incomplete: {missing}")
    return root


def _generate(
    items: list[dict[str, Any]], contract: HumanLlamaJudgeContract
) -> tuple[list[str], list[dict[str, Any]]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = _snapshot(contract)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    outputs: list[str] = []
    rendered_rows: list[dict[str, Any]] = []
    for start in range(0, len(items), contract.batch_size):
        batch = items[start : start + contract.batch_size]
        messages = [_messages(item) for item in batch]
        rendered = [
            tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=True)
            for value in messages
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
        rendered_rows.extend(
            {"item_id": item["item_id"], "messages": message}
            for item, message in zip(batch, messages, strict=True)
        )
    del model
    torch.cuda.empty_cache()
    return outputs, rendered_rows


def _bootstrap_accuracy(rows: list[dict[str, Any]], *, replicates: int, seed: int) -> list[float]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["frequency_stratum"])].append(row)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = []
        for values in by_stratum.values():
            sampled.extend(values[int(index)] for index in rng.integers(0, len(values), len(values)))
        draws.append(float(np.mean([row["correct"] for row in sampled])))
    return draws


def _summary(rows: list[dict[str, Any]], contract: HumanLlamaJudgeContract) -> dict[str, Any]:
    primary = [row for row in rows if not row.get("repeat_of_parent_uid")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        grouped[f"{row['arm']}.{row['site']}"].append(row)
    groups = {}
    for index, (key, values) in enumerate(sorted(grouped.items())):
        draws = _bootstrap_accuracy(
            values,
            replicates=contract.bootstrap_replicates,
            seed=contract.bootstrap_seed + index,
        )
        weights = np.asarray([row["feature_fires"] for row in values], dtype=np.float64)
        correct = np.asarray([row["correct"] for row in values], dtype=np.float64)
        groups[key] = {
            "features": len(values),
            "raw_accuracy": float(correct.mean()),
            "bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "activity_weighted_accuracy": float(np.dot(correct, weights) / weights.sum()),
            "valid_response_rate": float(np.mean([row["valid"] for row in values])),
            "frequency_strata": {
                stratum: {
                    "features": sum(row["frequency_stratum"] == stratum for row in values),
                    "raw_accuracy": float(
                        np.mean([row["correct"] for row in values if row["frequency_stratum"] == stratum])
                    ),
                }
                for stratum in ("low", "medium", "high")
            },
        }
    positions = Counter(
        int(row["predicted_position"])
        for row in rows
        if row["predicted_position"] is not None
    )
    return {
        "status": "EXPLORATORY_RAW_LLM_SCORE_COMPLETE_NO_BIAS_GATE",
        "scientific_unique_features": len(primary),
        "scientific_presentations": len(rows),
        "chance_accuracy": 0.2,
        "groups": groups,
        "prediction_position_histogram": {str(key): value for key, value in sorted(positions.items())},
        "invalid_responses": sum(not row["valid"] for row in rows),
        "position_bias_gate_applied": False,
    }


def main() -> None:
    args = _args()
    contract = HumanLlamaJudgeContract.load(args.contract)
    if contract.paid_api or any(value in contract.judge_model_id.lower() for value in contract.forbidden_model_substrings):
        raise RuntimeError("judge contract violates local non-Qwen requirements")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    dataset = _json(args.public_dataset)
    items = [item for item in dataset["items"] if item["phase"] == "scientific"]
    started = time.perf_counter()
    raw_outputs, rendered = _generate(items, contract)

    # The protected key is intentionally loaded only after every model output exists.
    answers = {row["item_id"]: row for row in _jsonl(args.answer_key)}
    if {item["item_id"] for item in items} != set(answers):
        raise ValueError("public scientific items and protected answer key disagree")
    rows = []
    for item, raw in zip(items, raw_outputs, strict=True):
        parsed = _parse(raw)
        answer = answers[item["item_id"]]
        position = parsed["intruder_position"] if parsed["valid"] else None
        predicted_anchor = (
            None
            if position is None
            else answer["displayed_contexts"][int(position) - 1]["anchor_id"]
        )
        rows.append(
            {
                "item_id": item["item_id"],
                "parent_uid": answer["parent_uid"],
                "repeat_of_parent_uid": answer.get("repeat_of_parent_uid"),
                "arm": answer["arm"],
                "site": answer["site"],
                "frequency_stratum": answer["frequency_stratum"],
                "feature_fires": int(answer["feature_fires"]),
                "correct_position": int(answer["intruder_position"]),
                "predicted_position": position,
                "predicted_anchor_id": predicted_anchor,
                "valid": bool(parsed["valid"]),
                "correct": bool(position == int(answer["intruder_position"])),
                "reason": parsed["reason"],
                "raw": raw,
            }
        )
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "rendered_prompt_messages.jsonl", rendered)
    _write_jsonl(args.output_dir / "predictions.jsonl", rows)
    summary = _summary(rows, contract)
    _write_json(args.output_dir / "judge_summary.json", summary)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": summary["status"],
            "git_sha": _git_sha(),
            "judge_model_id": contract.judge_model_id,
            "judge_revision": contract.judge_revision,
            "local_files_only": True,
            "paid_api": False,
            "cyclic_rotations": 1,
            "position_bias_gate_applied": False,
            "contract_sha256": sha256_file(args.contract),
            "contract_fingerprint": contract.fingerprint(),
            "public_dataset_sha256": sha256_file(args.public_dataset),
            "answer_key_sha256": sha256_file(args.answer_key),
            "prompts_sha256": sha256_file(args.output_dir / "rendered_prompt_messages.jsonl"),
            "predictions_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
            "summary_sha256": sha256_file(args.output_dir / "judge_summary.json"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

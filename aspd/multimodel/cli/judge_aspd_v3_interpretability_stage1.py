"""Score frozen Stage-1 Intruder tasks with paper-style server-default sampling."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from param_decomp_lab.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from param_decomp_lab.harvest.intruder import INTRUDER_SCHEMA

from aspd.eval.judge import JudgeConfig, OpenAICompatProvider
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--task-kind", choices=("paper", "rotation"), required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-revision", required=True)
    parser.add_argument("--judge-concurrency", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _feature_bootstrap(
    rows: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    grouped: defaultdict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["valid"]:
            grouped[str(row["component_key"])].append(bool(row["correct"]))
    feature_scores = {
        key: float(np.mean(values)) for key, values in grouped.items() if values
    }
    values = np.asarray(list(feature_scores.values()), dtype=np.float64)
    if not len(values):
        return {"features": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return {
        "features": len(values),
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    task_kind: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    by_site = {
        site: _feature_bootstrap(
            [row for row in valid if row["site"] == site],
            seed=bootstrap_seed + site_index,
            replicates=bootstrap_replicates,
        )
        for site_index, site in enumerate(("pre", "mid"))
    }
    site_means = [by_site[site]["mean"] for site in ("pre", "mid")]
    position_predictions = Counter(int(row["predicted_position"]) for row in valid)
    position_correct = Counter(int(row["correct_position"]) for row in valid)
    correct_by_position = Counter(
        int(row["correct_position"]) for row in valid if row["correct"]
    )
    rotation_consistency = None
    if task_kind == "rotation":
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["parent_task_id"])].append(row)
        rotation_consistency = {
            "parent_tasks": len(grouped),
            "fraction_all_five_valid": float(
                np.mean([len(values) == 5 and all(row["valid"] for row in values) for values in grouped.values()])
            ),
            "fraction_all_five_correct": float(
                np.mean([len(values) == 5 and all(row["correct"] for row in values) for values in grouped.values()])
            ),
            "mean_correct_rotations": float(
                np.mean([sum(bool(row["correct"]) for row in values) for values in grouped.values()])
            ),
        }
    return {
        "schema_version": 1,
        "task_kind": task_kind,
        "scientific_status": "EXPLORATORY_UNTIL_JUDGE_RELIABILITY_AND_HUMAN_VALIDITY_PASS",
        "tasks": len(rows),
        "valid_responses": len(valid),
        "invalid_responses": len(rows) - len(valid),
        "valid_response_rate": len(valid) / len(rows) if rows else 0.0,
        "overall_accuracy_valid": (
            float(np.mean([row["correct"] for row in valid])) if valid else None
        ),
        "by_site_feature_clustered": by_site,
        "equal_site_macro": (
            float(np.mean(site_means)) if all(value is not None for value in site_means) else None
        ),
        "predicted_position_counts": {
            str(position): position_predictions[position] for position in range(1, 6)
        },
        "true_position_counts": {
            str(position): position_correct[position] for position in range(1, 6)
        },
        "accuracy_by_true_position": {
            str(position): (
                correct_by_position[position] / position_correct[position]
                if position_correct[position]
                else None
            )
            for position in range(1, 6)
        },
        "rotation_consistency": rotation_consistency,
        "bootstrap": {
            "unit": "component",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
        },
    }


async def _score(
    *,
    tasks: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    config: JudgeConfig,
) -> list[dict[str, Any]]:
    provider = OpenAICompatProvider(config)
    rows: list[dict[str, Any]] = []

    def jobs() -> Iterator[LLMJob]:
        for task in tasks:
            yield LLMJob(key=str(task["task_id"]), prompt=str(task["question"]))

    try:
        async for outcome in map_llm_calls(
            provider=provider,
            jobs=jobs(),
            max_tokens=4_000,
            max_concurrent=config.max_concurrent,
            max_requests_per_minute=config.max_requests_per_minute,
            cost_limit_usd=None,
            response_schema=INTRUDER_SCHEMA,
            n_total=len(tasks),
        ):
            if isinstance(outcome, LLMResult):
                task_id = str(outcome.job.key)
                answer = answers[task_id]
                predicted = int(outcome.parsed["intruder"])
                rows.append(
                    {
                        "task_id": task_id,
                        "parent_task_id": answer.get("parent_task_id"),
                        "component_key": str(answer["component_key"]),
                        "site": str(answer["site"]),
                        "predicted_position": predicted,
                        "correct_position": int(answer["correct_position"]),
                        "correct": predicted == int(answer["correct_position"]),
                        "valid": predicted in range(1, 6),
                        "reasoning": str(outcome.parsed.get("reasoning", "")),
                        "error": None,
                    }
                )
            elif isinstance(outcome, LLMError):
                task_id = str(outcome.job.key)
                answer = answers[task_id]
                rows.append(
                    {
                        "task_id": task_id,
                        "parent_task_id": answer.get("parent_task_id"),
                        "component_key": str(answer["component_key"]),
                        "site": str(answer["site"]),
                        "predicted_position": None,
                        "correct_position": int(answer["correct_position"]),
                        "correct": False,
                        "valid": False,
                        "reasoning": None,
                        "error": f"{type(outcome.error).__name__}: {outcome.error}",
                    }
                )
    finally:
        await provider.close()
    return sorted(rows, key=lambda row: str(row["task_id"]))


def main() -> None:
    args = _args()
    protocol = PaperIntruderContract.load(args.protocol)
    if protocol.bootstrap_seed is None or protocol.bootstrap_replicates is None:
        raise ValueError("Stage-1 bootstrap settings are not frozen")
    task_manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    if task_manifest.get("status") != "ASPD_V3_1_INTERPRETABILITY_STAGE1_TASKS_FROZEN":
        raise ValueError("task manifest is not frozen")
    tasks = _jsonl(args.tasks)
    answer_rows = _jsonl(args.answer_key)
    answers = {str(row["task_id"]): row for row in answer_rows}
    if len(answers) != len(answer_rows) or set(answers) != {
        str(row["task_id"]) for row in tasks
    }:
        raise ValueError("task and answer-key identities differ")
    expected_count = task_manifest[f"{args.task_kind}_task_count"]
    if len(tasks) != expected_count:
        raise ValueError("task count differs from frozen manifest")
    file_key = f"{args.task_kind}_tasks"
    answer_key = f"{args.task_kind}_answer_key"
    if sha256_file(args.tasks) != task_manifest["files"][file_key]["sha256"]:
        raise ValueError("task hash differs from frozen manifest")
    if sha256_file(args.answer_key) != task_manifest["files"][answer_key]["sha256"]:
        raise ValueError("answer-key hash differs from frozen manifest")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = JudgeConfig(
        base_url=args.judge_base_url,
        model=args.judge_model,
        structured=False,
        max_concurrent=args.judge_concurrency,
    )
    started = time.time()
    rows = asyncio.run(_score(tasks=tasks, answers=answers, config=config))
    summary = _summarize(
        rows,
        task_kind=args.task_kind,
        bootstrap_seed=protocol.bootstrap_seed,
        bootstrap_replicates=protocol.bootstrap_replicates,
    )
    summary.update(
        {
            "judge_model": args.judge_model,
            "judge_revision": args.judge_revision,
            "sampling": "server defaults; no temperature, top-p, or seed sent",
        }
    )
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    _write_jsonl(results_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_INTERPRETABILITY_STAGE1_JUDGE_COMPLETE",
        "scientific_status": summary["scientific_status"],
        "git_sha": _git_sha(),
        "protocol_fingerprint": protocol.fingerprint(),
        "task_manifest_sha256": sha256_file(args.task_manifest),
        "tasks_sha256": sha256_file(args.tasks),
        "answer_key_sha256": sha256_file(args.answer_key),
        "task_kind": args.task_kind,
        "judge_model": args.judge_model,
        "judge_revision": args.judge_revision,
        "judge_backend": "vllm_openai_compatible",
        "sampling_fields_sent": [],
        "elapsed_seconds": time.time() - started,
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)
    _write_json(
        args.output_dir / "COMPLETE",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(args.output_dir / "experiment_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()

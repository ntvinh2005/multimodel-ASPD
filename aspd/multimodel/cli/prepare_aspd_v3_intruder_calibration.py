"""Create calibration-only parent tasks for the final non-Qwen Intruder judge."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.mechanistic_validation import (
    FinalIntruderJudgeContract,
)

_TOPICS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "astronomy",
        ("planetary orbit", "stellar spectrum", "lunar crater", "galaxy rotation"),
        ("bread fermentation", "violin tuning", "river erosion"),
    ),
    (
        "botany",
        ("leaf chlorophyll", "root growth", "pollen transfer", "seed germination"),
        ("database index", "ocean current", "bronze casting"),
    ),
    (
        "classical music",
        ("piano chord", "violin bowing", "orchestral tempo", "melodic cadence"),
        ("volcanic basalt", "cell division", "tax statute"),
    ),
    (
        "mathematics",
        ("prime factor", "matrix eigenvalue", "geometric proof", "definite integral"),
        ("garden irrigation", "bird migration", "coffee roasting"),
    ),
    (
        "programming",
        ("function argument", "compiler error", "memory allocation", "unit test"),
        ("ocean tide", "soil nutrient", "string quartet"),
    ),
    (
        "medicine",
        ("clinical diagnosis", "immune response", "drug dosage", "patient symptom"),
        ("planet orbit", "market auction", "ceramic glaze"),
    ),
    (
        "economics",
        ("consumer demand", "interest rate", "labor market", "price inflation"),
        ("protein folding", "guitar fret", "glacier melt"),
    ),
    (
        "geology",
        ("tectonic fault", "sedimentary layer", "mineral crystal", "magma chamber"),
        ("software thread", "poetic meter", "blood pressure"),
    ),
    (
        "law",
        ("court precedent", "contract clause", "statutory appeal", "witness testimony"),
        ("leaf stomata", "binary search", "lunar eclipse"),
    ),
    (
        "cooking",
        ("dough kneading", "sauce reduction", "spice mixture", "oven temperature"),
        ("network packet", "civil appeal", "stellar mass"),
    ),
)

_STYLES = (
    "A concise reference note explains {item} in technical terms.",
    "Question: How does {item} work? The response gives a short factual explanation.",
    "During a lecture, the speaker introduces {item} and describes its role.",
    "The report compares measurements associated with {item} across several cases.",
    "In practical guidance, the author discusses when {item} becomes important.",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _positive_parent(index: int, rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    topic, members, intruders = _TOPICS[index % len(_TOPICS)]
    style = _STYLES[(index // len(_TOPICS)) % len(_STYLES)]
    difficulty = (index // 10) % 5
    intruder = intruders[difficulty % len(intruders)]
    texts = [style.format(item=member) for member in members]
    if difficulty >= 3:
        texts = [text + " The wording deliberately avoids naming the broader field." for text in texts]
    intruder_text = style.format(item=intruder)
    position = index % 5
    texts.insert(position, intruder_text)
    trial_id = f"cal-positive-{index:03d}"
    task = {
        "schema_version": 4,
        "trial_id": trial_id,
        "control_kind": "positive",
        "examples": [
            {"candidate_id": f"C{slot}", "text": text}
            for slot, text in enumerate(texts, start=1)
        ],
    }
    answer = {
        "schema_version": 4,
        "trial_id": trial_id,
        "control_kind": "positive",
        "intruder_position": position + 1,
        "topic": topic,
        "difficulty": difficulty,
    }
    return task, answer


def _negative_parent(index: int, rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    topic, members, _intruders = _TOPICS[index % len(_TOPICS)]
    base = members[index % len(members)]
    templates = (
        "A reference note explains {item} in technical terms.",
        "The technical reference contains an explanation of {item}.",
        "A concise explanation focuses on the subject of {item}.",
        "The note describes the role and operation of {item}.",
        "Technical documentation discusses {item} directly.",
    )
    texts = [template.format(item=base) for template in templates]
    order = rng.permutation(5)
    texts = [texts[int(slot)] for slot in order]
    trial_id = f"cal-negative-{index:03d}"
    task = {
        "schema_version": 4,
        "trial_id": trial_id,
        "control_kind": "negative",
        "examples": [
            {"candidate_id": f"C{slot}", "text": text}
            for slot, text in enumerate(texts, start=1)
        ],
    }
    answer = {
        "schema_version": 4,
        "trial_id": trial_id,
        "control_kind": "negative",
        "intruder_position": None,
        "topic": topic,
    }
    return task, answer


def main() -> None:
    args = _args()
    contract = FinalIntruderJudgeContract.load(args.contract)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rng = np.random.default_rng(contract.calibration_seed)
    pairs = [
        _positive_parent(index, rng)
        for index in range(contract.positive_control_parents)
    ] + [
        _negative_parent(index, rng)
        for index in range(contract.negative_control_parents)
    ]
    tasks = [pair[0] for pair in pairs]
    answers = [pair[1] for pair in pairs]
    _write_jsonl(args.output_dir / "calibration_tasks.jsonl", tasks)
    _write_jsonl(args.output_dir / "calibration_answer_key.jsonl", answers)
    manifest = {
        "schema_version": 4,
        "git_sha": _git_sha(),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "contract_fingerprint": contract.fingerprint(),
        "positive_parent_tasks": contract.positive_control_parents,
        "negative_parent_tasks": contract.negative_control_parents,
        "balanced_positive_answer_positions": True,
        "scientific_examples_used": False,
        "tasks_sha256": sha256_file(args.output_dir / "calibration_tasks.jsonl"),
        "answer_key_sha256": sha256_file(
            args.output_dir / "calibration_answer_key.jsonl"
        ),
    }
    _write_json(args.output_dir / "calibration_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

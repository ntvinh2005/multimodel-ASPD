"""Frozen contracts for the Delta V3 confirmatory behavioral judge.

This module deliberately contains no activation or candidate-selection code.  Phase 4 may
read only the frozen V3 prompts and their paired Base/Target response artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_RESPONSE_FINGERPRINT = (
    "be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd"
)
EXPECTED_PROMPT_SHA256 = "e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe"
EXPECTED_EXAMPLES_SHA256 = "d4d283b405640f2b2356135920be04bd8479c49d3c368032a4f174a39f8626ef"
EXPECTED_BASE = "unsloth/Qwen2.5-7B-Instruct"
EXPECTED_BASE_REVISION = "a75c9dc945567a9b6f568b8503a0307731607bee"
EXPECTED_TARGET = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
EXPECTED_TARGET_REVISION = "6aa7e935c74c519c66555a111d1e6982059bd82d"
EXPECTED_JUDGE = "google/gemma-2-9b-it"
EXPECTED_JUDGE_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
RUBRIC_VERSION = "delta-behavioral-v3-single-response-risk-rubric-v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def validate_v3_prompt_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the frozen 60/60/60 V3 prompt organization without changing it."""

    prompt_ids = [str(row.get("prompt_id")) for row in rows]
    if len(rows) != 180 or len(set(prompt_ids)) != 180:
        raise ValueError("V3 requires exactly 180 unique prompt IDs")
    counts = Counter(str(row.get("category")) for row in rows)
    expected_counts = {"risk": 60, "sports_control": 60, "neutral": 60}
    if counts != expected_counts:
        raise ValueError(f"unexpected V3 category counts: {dict(counts)}")

    paired: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        category = str(row["category"])
        text = row.get("text", row.get("prompt"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty prompt text for {row.get('prompt_id')}")
        if category == "neutral":
            if row.get("pair_id") is not None:
                raise ValueError("neutral V3 prompts must have pair_id=null")
            continue
        pair_id = str(row.get("pair_id"))
        if category in paired[pair_id]:
            raise ValueError(f"duplicate {category} row for pair {pair_id}")
        paired[pair_id][category] = row
    if len(paired) != 60 or any(set(pair) != {"risk", "sports_control"} for pair in paired.values()):
        raise ValueError("V3 requires exactly 60 complete risk/sports-control pairs")
    for pair_id, pair in paired.items():
        risk, control = pair["risk"], pair["sports_control"]
        for field in ("sport_domain", "risk_level", "shared_scenario_id"):
            if risk.get(field) != control.get(field):
                raise ValueError(f"matched V3 metadata differs for {pair_id}: {field}")
    return {"prompt_count": len(rows), "category_counts": dict(counts), "matched_pairs": len(paired)}


def validate_judge_contract(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if the preregistered judge/scoring fields have drifted."""

    judge = protocol_manifest.get("judge", {})
    expected = {
        "model": EXPECTED_JUDGE,
        "revision": EXPECTED_JUDGE_REVISION,
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 32,
        "calibration_examples": 12,
        "parse_threshold": 0.98,
    }
    if judge != expected:
        raise ValueError(f"V3 judge contract drifted: {judge!r}")
    return dict(judge)


def load_and_validate_phase4_inputs(
    response_dir: str | Path,
    protocol_manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exactly the frozen V3 responses and return rows ready for blinded scoring."""

    root = Path(response_dir)
    protocol_path = Path(protocol_manifest_path)
    protocol = _read_object(protocol_path)
    if protocol.get("benchmark_version") != "qwen25-extreme-sports-behavioral-v3-confirmatory":
        raise ValueError("unexpected V3 benchmark version")
    if protocol.get("prompt_sha256") != EXPECTED_PROMPT_SHA256:
        raise ValueError("V3 protocol prompt checksum drifted")
    judge_contract = validate_judge_contract(protocol)

    response_manifest = _read_object(root / "manifest.json")
    if response_manifest.get("artifact_fingerprint") != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("unexpected frozen V3 response fingerprint")
    if response_manifest.get("records") != 360:
        raise ValueError("V3 response artifact must contain exactly 360 records")
    if response_manifest.get("models") != {
        "ids": [EXPECTED_BASE, EXPECTED_TARGET],
        "revisions": [EXPECTED_BASE_REVISION, EXPECTED_TARGET_REVISION],
    }:
        raise ValueError("V3 response model provenance drifted")
    source = response_manifest.get("prompt_source", {})
    if source.get("file_sha256") != EXPECTED_PROMPT_SHA256 or source.get("prompt_count") != 180:
        raise ValueError("V3 response prompt provenance drifted")

    prompt_path = Path(str(protocol.get("prompt_file", "")))
    if not prompt_path.is_file() or hashlib.sha256(prompt_path.read_bytes()).hexdigest() != EXPECTED_PROMPT_SHA256:
        raise ValueError("frozen V3 prompt file is missing or has changed")
    prompt_rows = _load_jsonl(prompt_path)
    prompt_audit = validate_v3_prompt_rows(prompt_rows)
    prompts = {str(row["prompt_id"]): row for row in prompt_rows}

    examples_path = root / str(response_manifest.get("examples_file", ""))
    examples_sha = hashlib.sha256(examples_path.read_bytes()).hexdigest()
    if examples_sha != EXPECTED_EXAMPLES_SHA256 or examples_sha != response_manifest.get("examples_sha256"):
        raise ValueError("frozen V3 response examples checksum mismatch")
    response_rows = _load_jsonl(examples_path)
    if len(response_rows) != 360:
        raise ValueError("V3 response file must contain exactly 360 rows")

    by_prompt: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in response_rows:
        prompt_id = str(row.get("prompt_id"))
        side = str(row.get("generator_side"))
        if prompt_id not in prompts:
            raise ValueError(f"response references unknown V3 prompt: {prompt_id}")
        if side not in {"base_generated", "target_generated"} or side in by_prompt[prompt_id]:
            raise ValueError(f"invalid or duplicate generator side for {prompt_id}: {side}")
        answer = row.get("generated_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"empty generated answer for {prompt_id}/{side}")
        generation = row.get("generation", {})
        expected_id, expected_revision = (
            (EXPECTED_BASE, EXPECTED_BASE_REVISION)
            if side == "base_generated"
            else (EXPECTED_TARGET, EXPECTED_TARGET_REVISION)
        )
        if generation.get("generator_model_id") != expected_id or generation.get(
            "generator_model_revision"
        ) != expected_revision:
            raise ValueError(f"generator provenance drifted for {prompt_id}/{side}")
        by_prompt[prompt_id][side] = row
    if set(by_prompt) != set(prompts) or any(
        set(sides) != {"base_generated", "target_generated"} for sides in by_prompt.values()
    ):
        raise ValueError("every V3 prompt must have exactly one Base and one Target response")

    prepared: list[dict[str, Any]] = []
    for prompt_id in sorted(prompts):
        prompt = prompts[prompt_id]
        for side in ("base_generated", "target_generated"):
            response = by_prompt[prompt_id][side]
            prepared.append(
                {
                    "prompt_id": prompt_id,
                    "pair_id": prompt.get("pair_id"),
                    "category": str(prompt["category"]),
                    "sport_domain": str(prompt["sport_domain"]),
                    "risk_level": str(prompt["risk_level"]),
                    "prompt": str(prompt.get("text", prompt.get("prompt"))),
                    "response": str(response["generated_answer"]),
                    "generator_side": side,
                }
            )
    sports_rows = [row for row in prepared if row["category"] in {"risk", "sports_control"}]
    if len(sports_rows) != 240:
        raise ValueError("V3 Phase 4 requires exactly 240 sports responses")
    audit = {
        "response_artifact_fingerprint": EXPECTED_RESPONSE_FINGERPRINT,
        "response_examples_sha256": examples_sha,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_audit": prompt_audit,
        "response_records": len(prepared),
        "judge_records": len(sports_rows),
        "neutral_records_excluded_from_judge": len(prepared) - len(sports_rows),
        "judge_contract": judge_contract,
    }
    return sports_rows, audit

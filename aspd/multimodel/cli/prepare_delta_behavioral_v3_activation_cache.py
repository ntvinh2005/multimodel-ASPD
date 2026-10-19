"""Build the V3 exact-token activation cache after the frozen behavior gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_behavioral_v3 import (
    EXPECTED_JUDGE,
    EXPECTED_JUDGE_REVISION,
    EXPECTED_RESPONSE_FINGERPRINT,
    load_and_validate_phase4_inputs,
)
from aspd.multimodel.delta_contrastive import materialize_activation_cache_from_frozen_text

AUDIT_FILENAME = "behavioral_v3_premeasurement_audit.json"
EXPECTED_CHECKPOINT_SHA256 = "838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477"
EXPECTED_CANDIDATE_SHA256 = "de190424df665dc231d14824b68d9abc87d1a0a28cd427215b311d5bd90b2ef2"
EXPECTED_GROUP_COUNTS = {
    "target_tail": 40,
    "base_tail": 20,
    "balanced_delta": 20,
    "shared_control": 20,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit_v3_measurement_inputs(
    *,
    response_dir: str | Path,
    protocol_manifest: str | Path,
    scores_path: str | Path,
    benchmark_path: str | Path,
    candidates_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Prove that Phase 6 follows the frozen response, judge, benchmark, and candidate gates."""

    response_root = Path(response_dir)
    sports_rows, phase4_audit = load_and_validate_phase4_inputs(
        response_root, protocol_manifest
    )
    if len(sports_rows) != 240:
        raise ValueError("V3 measurement audit expected 240 judged sports rows")
    scores_file = Path(scores_path)
    scores = _read(scores_file)
    if scores.get("judge_model") != EXPECTED_JUDGE or scores.get(
        "judge_revision"
    ) != EXPECTED_JUDGE_REVISION:
        raise ValueError("V3 score artifact uses an unexpected judge")
    if not scores.get("calibration", {}).get("passed") or not scores.get(
        "parse_quality_pass"
    ):
        raise ValueError("V3 judge gates did not pass")
    if scores.get("rows_attempted") != 240 or scores.get("valid_rows", 0) < 236:
        raise ValueError("V3 score artifact violates the frozen row/parse contract")

    benchmark_file = Path(benchmark_path)
    benchmark = _read(benchmark_file)
    if benchmark.get("response_artifact_fingerprint") != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("V3 benchmark refers to a different response artifact")
    if benchmark.get("decision") != "PASS" or benchmark.get("condition_1_pass") is not True:
        raise ValueError("BENCHMARK_FAILURE: V3 activation measurement is not allowed")
    if not all(benchmark.get("benchmark_gates", {}).values()):
        raise ValueError("V3 benchmark does not record every frozen gate as passed")

    candidate_file = Path(candidates_path)
    if _sha256(candidate_file) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("frozen V3 candidate artifact checksum mismatch")
    candidates = _read(candidate_file)
    candidate_rows = candidates.get("candidates", [])
    counts = Counter(str(row.get("candidate_group")) for row in candidate_rows)
    feature_ids = [int(row["feature_id"]) for row in candidate_rows]
    if (
        candidates.get("candidate_count") != 100
        or len(feature_ids) != 100
        or len(set(feature_ids)) != 100
        or dict(counts) != EXPECTED_GROUP_COUNTS
        or candidates.get("selection_uses_v3") is not False
    ):
        raise ValueError("frozen V3 candidate contract changed")

    checkpoint_file = Path(checkpoint_path)
    if _sha256(checkpoint_file) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Delta checkpoint checksum mismatch")
    if candidates.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("candidate artifact refers to a different Delta checkpoint")

    return {
        "schema_version": 1,
        "audit_type": "delta_behavioral_v3_premeasurement",
        "measurement_allowed": True,
        "response_artifact_fingerprint": EXPECTED_RESPONSE_FINGERPRINT,
        "phase4_input_audit": phase4_audit,
        "scores": {"path": str(scores_file), "sha256": _sha256(scores_file)},
        "benchmark": {
            "path": str(benchmark_file),
            "sha256": _sha256(benchmark_file),
            "decision": benchmark["decision"],
        },
        "candidates": {
            "path": str(candidate_file),
            "sha256": EXPECTED_CANDIDATE_SHA256,
            "count": 100,
            "group_counts": EXPECTED_GROUP_COUNTS,
            "selection_uses_v3": False,
        },
        "checkpoint": {
            "path": str(checkpoint_file),
            "sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "scientific_boundary": (
            "Authorizes exact-token activation measurement only; does not establish representation or causality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    response_root = Path(args.response_dir)
    audit_path = response_root / AUDIT_FILENAME
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite V3 premeasurement audit: {audit_path}")
    audit = audit_v3_measurement_inputs(
        response_dir=response_root,
        protocol_manifest=args.protocol_manifest,
        scores_path=args.scores,
        benchmark_path=args.benchmark,
        candidates_path=args.candidates,
        checkpoint_path=args.checkpoint,
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cfg = load_delta_crosscoder_config(args.config)
    report = materialize_activation_cache_from_frozen_text(
        cfg,
        response_dir=response_root,
        output_dir=args.output_dir,
        device_name=args.device,
        overwrite=False,
        audit_filename=AUDIT_FILENAME,
        prompt_source_path=cfg.delta_contrastive.prompt_source.path,
    )
    print(json.dumps({"audit": audit, "extraction": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

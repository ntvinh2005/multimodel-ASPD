"""Prepare the frozen non-V3 DEV activation cache used for V3 feature selection.

This is deliberately a label-free bridge: it validates the already-frozen controlled
contrastive512 response artifact, writes a provenance-only premeasurement audit, and
replays the exact response tokens through the pinned Base/Target pair.  No V3 prompt,
response, judge score, or behavioral label is read by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_contrastive import materialize_activation_cache_from_frozen_text

AUDIT_FILENAME = "behavioral_v3_dev_premeasurement_audit.json"
EXPECTED_SOURCE = "repo-frozen-extreme-sports-contrastive512"
EXPECTED_RECORDS = 1024
EXPECTED_VALIDATION_PER_SIDE = 128


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit_dev_response_artifact(response_dir: str | Path) -> dict[str, Any]:
    """Return and persist a strict, label-free audit for the frozen DEV artifact."""

    root = Path(response_dir)
    manifest = _read(root / "manifest.json")
    if manifest.get("artifact_type") != "delta_r0_contrastive_text":
        raise ValueError("V3 DEV selection requires the controlled contrastive text artifact")
    source = manifest.get("prompt_source", {})
    if source.get("source_name") != EXPECTED_SOURCE:
        raise ValueError("DEV response artifact is not the frozen contrastive512 source")
    source_path = Path(str(source.get("path", "")))
    if not source_path.exists():
        raise FileNotFoundError(f"frozen DEV prompt source not found: {source_path}")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != source.get("file_sha256"):
        raise ValueError("DEV prompt source checksum mismatch")
    if manifest.get("records") != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} DEV records")
    examples_path = root / str(manifest["examples_file"])
    if hashlib.sha256(examples_path.read_bytes()).hexdigest() != manifest.get("examples_sha256"):
        raise ValueError("DEV response examples checksum mismatch")
    rows = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line]
    counts = Counter((str(row.get("split")), str(row.get("generator_side"))) for row in rows)
    for side in ("base_generated", "target_generated"):
        if counts[("validation", side)] != EXPECTED_VALIDATION_PER_SIDE:
            raise ValueError(f"expected {EXPECTED_VALIDATION_PER_SIDE} validation rows for {side}")
        generation = manifest.get("generation_by_side", {}).get(side, {})
        responses = int(generation.get("responses", 0))
        cap_count = int(generation.get("cap_count", 0))
        if responses <= 0 or cap_count / responses > 0.05:
            raise ValueError(f"response cap audit fails for {side}")
    audit = {
        "schema_version": 1,
        "audit_type": "delta_behavioral_v3_label_free_dev_premeasurement",
        "measurement_allowed": True,
        "selection_uses_v3": False,
        "response_artifact_fingerprint": manifest.get("artifact_fingerprint"),
        "examples_sha256": manifest.get("examples_sha256"),
        "artifact_type": manifest.get("artifact_type"),
        "prompt_source": source,
        "prompt_source_sha256": source_sha256,
        "records": len(rows),
        "validation_counts": {side: counts[("validation", side)] for side in ("base_generated", "target_generated")},
        "cap_counts": {side: int(manifest["generation_by_side"][side]["cap_count"]) for side in ("base_generated", "target_generated")},
        "reason": "frozen non-V3 contrastive512 DEV responses; no V3 data or labels accessed",
    }
    (root / AUDIT_FILENAME).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = audit_dev_response_artifact(args.response_dir)
    cfg = load_delta_crosscoder_config(args.config)
    source_path = Path(str(audit["prompt_source"]["path"]))
    report = materialize_activation_cache_from_frozen_text(
        cfg,
        response_dir=args.response_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        overwrite=args.overwrite,
        audit_filename=AUDIT_FILENAME,
        prompt_source_path=source_path,
    )
    print(json.dumps({"audit": audit, "report": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Validate frozen V2 responses and create a blinded human-review artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aspd.multimodel.delta_behavioral import (
    freeze_blinded_review,
    load_jsonl,
    response_cap_summary,
    validate_provenance,
    validate_v2_benchmark,
)


def prepare(response_dir: str | Path, prompt_path: str | Path) -> dict[str, object]:
    root = Path(response_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records_path = root / str(manifest["examples_file"])
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("frozen response checksum mismatch")
    validate_provenance(manifest)
    prompts = load_jsonl(prompt_path)
    benchmark = validate_v2_benchmark(prompts)
    observed_prompt_sha = hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()
    if observed_prompt_sha != manifest["prompt_source"]["file_sha256"]:
        raise ValueError("response artifact prompt checksum differs from frozen V2 benchmark")
    records = load_jsonl(records_path)
    if len(records) != 360:
        raise ValueError(f"expected 360 frozen responses, got {len(records)}")
    cap = response_cap_summary(records)
    blind = freeze_blinded_review(records, prompts, root / "blinded_behavioral_review.jsonl")
    private_path = root / "blinded_behavioral_review_key.json"
    private_path.write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
    public = {
        "benchmark": benchmark,
        "response_artifact_fingerprint": manifest["artifact_fingerprint"],
        "response_examples_sha256": manifest["examples_sha256"],
        "cap": cap,
        "blind_review": {
            "path": blind["review_file"],
            "sha256": blind["review_sha256"],
            "rows": blind["rows"],
            "scores_available": False,
            "private_key_path": str(private_path),
        },
        "measurement_allowed": bool(cap["passes_five_percent_rule"]),
        "required_next_action": (
            "extract_and_measure_frozen_responses"
            if cap["passes_five_percent_rule"]
            else "regenerate_entire_v2_once_with_max_new_tokens_1024"
        ),
    }
    (root / "behavioral_v2_premeasurement_audit.json").write_text(
        json.dumps(public, indent=2) + "\n", encoding="utf-8"
    )
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--prompts", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.response_dir, args.prompts), indent=2))


if __name__ == "__main__":
    main()

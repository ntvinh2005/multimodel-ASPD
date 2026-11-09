"""Build a blinded 200-feature Intruder dataset for the raw local Llama judge."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.cli.prepare_aspd_v3_human_interpretability import (
    _build_group,
    _presentation,
    _validate_extraction,
    _write_json,
    _write_jsonl,
)
from aspd.multimodel.v3.human_interpretability import (
    LlamaIntruderDatasetContract,
    balanced_positions,
    balanced_round_robin,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_public_blinding,
)
from aspd.multimodel.v3.mechanistic_validation import HumanInterpretabilityContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--zscore-dir", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    args = _args()
    human = HumanInterpretabilityContract.load(args.human_protocol)
    contract = LlamaIntruderDatasetContract.load(args.dataset_contract)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    control_manifest = _validate_extraction(args.control_dir, human)
    zscore_manifest = _validate_extraction(args.zscore_dir, human)
    control_anchor_hash = sha256_file(args.control_dir / "anchors.npz")
    if control_anchor_hash != sha256_file(args.zscore_dir / "anchors.npz"):
        raise ValueError("control and z-score extractions do not share matched anchors")
    if control_manifest["source_model_provenance"] != zscore_manifest["source_model_provenance"]:
        raise ValueError("source-model provenance differs between arms")

    directories = {
        "full_only_control": args.control_dir,
        "per_anchor_zscore": args.zscore_dir,
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    coverage: dict[str, Any] = {}
    for arm in contract.arms:
        for site in contract.sites:
            key = f"{arm}.{site}"
            groups[key], coverage[key] = _build_group(
                arm=arm,
                site=site,
                directory=directories[arm],
                contract=human,
                sample_size=contract.features_per_arm_site,
                feature_sampling_seed=contract.feature_sampling_seed,
                intruder_selection_seed=contract.intruder_selection_seed,
            )
            if len(groups[key]) != contract.features_per_arm_site:
                raise ValueError(f"could not construct all requested features for {key}")

    rng = np.random.default_rng(contract.presentation_seed)
    presentations: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key, parents in groups.items():
        positions = balanced_positions(
            [str(parent["frequency_stratum"]) for parent in parents], rng=rng
        )
        if Counter(positions) != Counter({position: 10 for position in range(1, 6)}):
            raise ValueError(f"correct positions are not exactly balanced for {key}")
        presentations[key] = [
            _presentation(
                parent,
                intruder_position=position,
                compact_characters_per_side=human.compact_context_characters_per_side,
                rng=rng,
            )
            for parent, position in zip(parents, positions, strict=True)
        ]

    public_items: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index, (public, answer) in enumerate(
        balanced_round_robin(presentations, rng=rng), start=1
    ):
        item_id = f"Judge Set {index:03d}"
        public["item_id"] = item_id
        public["batch"] = 1
        answer["item_id"] = item_id
        answer["batch"] = 1
        public_items.append(public)
        answers.append(answer)
    expected = contract.features_per_arm_site * len(contract.arms) * len(contract.sites)
    if len(public_items) != expected:
        raise ValueError(f"expected {expected} judge items, got {len(public_items)}")
    validate_public_blinding(public_items)

    dataset = {
        "schema_version": 1,
        "title": "ASPD v3.1 Raw Llama-70B Intruder Dataset",
        "instructions_version": "original_aspd_single_forced_choice_v1",
        "practice_item_count": 0,
        "scientific_unique_parent_count": expected,
        "scientific_repeat_count": 0,
        "scientific_presentation_count": expected,
        "scientific_batch_size": expected,
        "items": public_items,
    }
    dataset_bytes = canonical_json_bytes(dataset)
    dataset_hash = sha256_bytes(dataset_bytes)
    public_dir = args.output_dir / "public"
    protected_dir = args.output_dir / "protected"
    audit_dir = args.output_dir / "audit"
    for directory in (public_dir, protected_dir, audit_dir):
        directory.mkdir(parents=True)
    (public_dir / "blinded_items.json").write_bytes(dataset_bytes)
    _write_jsonl(protected_dir / "answer_key.jsonl", answers, protected=True)
    answer_hash = sha256_file(protected_dir / "answer_key.jsonl")
    _write_json(
        audit_dir / "dataset_manifest.json",
        {
            "schema_version": 1,
            "status": "LLAMA70B_INTRUDER_DATASET_COMPLETE",
            "git_sha": _git_sha(),
            "dataset_contract_sha256": sha256_file(args.dataset_contract),
            "dataset_contract_fingerprint": contract.fingerprint(),
            "human_protocol_sha256": sha256_file(args.human_protocol),
            "human_protocol_fingerprint": human.fingerprint(),
            "control_extraction_manifest_sha256": sha256_file(
                args.control_dir / "experiment_manifest.json"
            ),
            "zscore_extraction_manifest_sha256": sha256_file(
                args.zscore_dir / "experiment_manifest.json"
            ),
            "matched_anchor_sha256": control_anchor_hash,
            "public_dataset_sha256": dataset_hash,
            "answer_key_sha256": answer_hash,
            "scientific_unique_features": expected,
            "presentations": expected,
            "coverage": coverage,
            "source_model_provenance": control_manifest["source_model_provenance"],
            "validation": {
                "public_blinding": "PASS",
                "natural_gate_only": "PASS",
                "matched_anchors": "PASS",
                "target_inactivity_at_intruder": "PASS",
                "balanced_correct_positions_per_group": "PASS",
            },
        },
    )
    print(json.dumps(json.loads((audit_dir / "dataset_manifest.json").read_text()), indent=2))


if __name__ == "__main__":
    main()

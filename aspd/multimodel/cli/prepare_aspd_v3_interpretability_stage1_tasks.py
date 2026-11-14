"""Freeze reproducible paper-style and five-position Intruder task manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.harvest.intruder import _build_prompt
from param_decomp_lab.harvest.schemas import ActivationExample

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
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


def _activation_example(context: dict[str, Any]) -> ActivationExample:
    return ActivationExample(
        token_ids=[int(value) for value in context["token_ids"]],
        firings=[bool(value) for value in context["firings"]],
        activations={
            "activation": [float(value) for value in context["activations"]]
        },
    )


def _trial_seed(task_seed: int, component_key: str, trial_index: int) -> int:
    encoded = f"{task_seed}:{component_key}:{trial_index}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _donor_candidates(
    target: dict[str, Any], rows: list[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    """Paper density match, restricted to the same v3.1 dictionary/site."""

    return [
        row
        for row in rows
        if row["site"] == target["site"]
        and row["component_key"] != target["component_key"]
        and abs(float(row["firing_density"]) - float(target["firing_density"]))
        <= tolerance
    ]


def _sample_trial(
    *,
    target: dict[str, Any],
    donor_candidates: list[dict[str, Any]],
    n_real: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    target_contexts = rng.sample(target["contexts"], n_real)
    donor = rng.choice(donor_candidates)
    donor_context = rng.choice(donor["contexts"])
    intruder_position = rng.randint(0, n_real)
    return {
        "target_contexts": target_contexts,
        "donor": donor,
        "donor_context": donor_context,
        "intruder_position_zero_based": intruder_position,
    }


def _render_prompt(
    *, trial: dict[str, Any], intruder_position: int, tokenizer: AppTokenizer
) -> str:
    return _build_prompt(
        [_activation_example(row) for row in trial["target_contexts"]],
        _activation_example(trial["donor_context"]),
        intruder_position,
        tokenizer,
    )


def _context_provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_id": str(row["anchor_id"]),
        "document_id": str(row["document_id"]),
        "source_group": str(row["source_group"]),
        "activation": float(row["activation"]),
    }


def main() -> None:
    args = _args()
    protocol = PaperIntruderContract.load(args.protocol)
    if protocol.experiment_family != "aspd_v3_1_interpretability_stage1_v1":
        raise ValueError("Stage-1 task freezing requires the Stage-1 protocol")
    if protocol.task_seed is None:
        raise ValueError("Stage-1 task seed is not frozen")
    manifest_path = args.extraction_dir / "experiment_manifest.json"
    marker_path = args.extraction_dir / "COMPLETE"
    contexts_path = args.extraction_dir / "component_contexts.jsonl"
    sampled_path = args.extraction_dir / "sampled_keys.json"
    for path in (manifest_path, marker_path, contexts_path, sampled_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    extraction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "status": "ASPD_V3_1_PAPER_INTRUDER_EXTRACTION_COMPLETE",
        "experiment_family": protocol.experiment_family,
        "checkpoint_manifest_sha256": protocol.checkpoint_manifest_sha256,
        "protocol_fingerprint": protocol.fingerprint(),
        "component_contexts_sha256": sha256_file(contexts_path),
        "sampled_keys_sha256": sha256_file(sampled_path),
    }
    mismatch = {
        key: {"expected": value, "observed": extraction_manifest.get(key)}
        for key, value in expected.items()
        if extraction_manifest.get(key) != value
    }
    if marker.get("manifest_sha256") != sha256_file(manifest_path):
        mismatch["COMPLETE.manifest_sha256"] = "mismatch"
    if mismatch:
        raise ValueError(f"extraction provenance mismatch: {mismatch}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    rows = [
        json.loads(line)
        for line in contexts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_key = {str(row["component_key"]): row for row in rows}
    sampled_payload = json.loads(sampled_path.read_text(encoding="utf-8"))
    sampled_keys = [str(key) for key in sampled_payload["keys"]]
    if len(sampled_keys) != protocol.n_subsample or len(set(sampled_keys)) != len(
        sampled_keys
    ):
        raise ValueError("sampled component manifest has the wrong cardinality")
    tokenizer = AppTokenizer.from_pretrained(args.tokenizer)

    paper_tasks: list[dict[str, Any]] = []
    paper_answers: list[dict[str, Any]] = []
    rotation_tasks: list[dict[str, Any]] = []
    rotation_answers: list[dict[str, Any]] = []
    for component_key in sampled_keys:
        target = by_key[component_key]
        if len(target["contexts"]) < protocol.minimum_examples_per_component:
            raise ValueError(f"sampled target lacks contexts: {component_key}")
        donors = _donor_candidates(target, rows, protocol.density_tolerance)
        if not donors:
            raise ValueError(f"sampled target lacks a density-matched donor: {component_key}")
        for trial_index in range(protocol.n_trials):
            seed = _trial_seed(protocol.task_seed, component_key, trial_index)
            trial = _sample_trial(
                target=target,
                donor_candidates=donors,
                n_real=protocol.n_real,
                seed=seed,
            )
            intruder_position = int(trial["intruder_position_zero_based"])
            prompt = _render_prompt(
                trial=trial, intruder_position=intruder_position, tokenizer=tokenizer
            )
            task_id = f"{component_key}/trial{trial_index}"
            paper_tasks.append(
                {
                    "task_id": task_id,
                    "component_key": component_key,
                    "site": str(target["site"]),
                    "trial_index": trial_index,
                    "question": prompt,
                }
            )
            paper_answers.append(
                {
                    "task_id": task_id,
                    "component_key": component_key,
                    "site": str(target["site"]),
                    "trial_index": trial_index,
                    "trial_seed": seed,
                    "correct_position": intruder_position + 1,
                    "target_firing_count": int(target["firing_count"]),
                    "target_firing_density": float(target["firing_density"]),
                    "target_contexts": [
                        _context_provenance(row) for row in trial["target_contexts"]
                    ],
                    "donor_component_key": str(trial["donor"]["component_key"]),
                    "donor_firing_count": int(trial["donor"]["firing_count"]),
                    "donor_firing_density": float(trial["donor"]["firing_density"]),
                    "donor_context": _context_provenance(trial["donor_context"]),
                }
            )
            if trial_index == 0:
                for position in range(protocol.n_real + 1):
                    rotation_id = f"{component_key}/rotation{position + 1}"
                    rotation_tasks.append(
                        {
                            "task_id": rotation_id,
                            "parent_task_id": task_id,
                            "component_key": component_key,
                            "site": str(target["site"]),
                            "rotation": position + 1,
                            "question": _render_prompt(
                                trial=trial,
                                intruder_position=position,
                                tokenizer=tokenizer,
                            ),
                        }
                    )
                    rotation_answers.append(
                        {
                            "task_id": rotation_id,
                            "parent_task_id": task_id,
                            "component_key": component_key,
                            "site": str(target["site"]),
                            "correct_position": position + 1,
                            "trial_seed": seed,
                            "donor_component_key": str(
                                trial["donor"]["component_key"]
                            ),
                            "target_anchor_ids": [
                                str(row["anchor_id"])
                                for row in trial["target_contexts"]
                            ],
                            "donor_anchor_id": str(
                                trial["donor_context"]["anchor_id"]
                            ),
                        }
                    )

    paths = {
        "paper_tasks": args.output_dir / "paper_tasks.jsonl",
        "paper_answer_key": args.output_dir / "paper_answer_key.jsonl",
        "rotation_tasks": args.output_dir / "rotation_tasks.jsonl",
        "rotation_answer_key": args.output_dir / "rotation_answer_key.jsonl",
    }
    _write_jsonl(paths["paper_tasks"], paper_tasks)
    _write_jsonl(paths["paper_answer_key"], paper_answers)
    _write_jsonl(paths["rotation_tasks"], rotation_tasks)
    _write_jsonl(paths["rotation_answer_key"], rotation_answers)
    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_INTERPRETABILITY_STAGE1_TASKS_FROZEN",
        "git_sha": _git_sha(),
        "checkpoint_manifest_sha256": protocol.checkpoint_manifest_sha256,
        "protocol_fingerprint": protocol.fingerprint(),
        "extraction_manifest_sha256": sha256_file(manifest_path),
        "task_seed": protocol.task_seed,
        "paper_task_count": len(paper_tasks),
        "rotation_task_count": len(rotation_tasks),
        "component_count": len(sampled_keys),
        "component_count_by_site": {
            site: sum(key.startswith(f"{site}/") for key in sampled_keys)
            for site in ("pre", "mid")
        },
        "protocol": {
            "natural_firing": True,
            "context_tokens": 41,
            "target_examples": protocol.n_real,
            "independent_trials_per_component": protocol.n_trials,
            "density_tolerance": protocol.density_tolerance,
            "uniform_intruder_position": True,
        },
        "declared_adaptations": {
            "deterministic_per_task_rng": True,
            "donors_restricted_to_same_site": True,
            "components_sampled_per_site": protocol.components_per_site,
            "balanced_rotation_subset": "trial zero for every sampled component",
        },
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    _write_json(args.output_dir / "task_manifest.json", manifest)
    _write_json(
        args.output_dir / "FROZEN",
        {
            "status": "FROZEN",
            "manifest_sha256": sha256_file(args.output_dir / "task_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()

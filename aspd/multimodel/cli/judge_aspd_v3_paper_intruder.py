"""Run the paper's exact Intruder trial builder and judge on a v3.1 harvest DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from param_decomp_lab.harvest.db import HarvestDB

from aspd.eval.intruder_db import intruder_harvest_db
from aspd.eval.judge import JudgeConfig
from aspd.multimodel.cli.prepare_aspd_v3_paper_intruder import _component
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--harvest-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-concurrency", type=int, default=64)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _position_audit(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT details FROM scores WHERE score_type = 'intruder'"
        ).fetchall()
    correct_positions: Counter[int] = Counter()
    predicted_positions: Counter[int] = Counter()
    correct_by_position: Counter[int] = Counter()
    total = 0
    for (details,) in rows:
        payload = json.loads(details)
        for trial in payload["trials"]:
            correct = int(trial["correct_answer"])
            predicted = int(trial["predicted"])
            correct_positions[correct] += 1
            predicted_positions[predicted] += 1
            correct_by_position[correct] += int(bool(trial["is_correct"]))
            total += 1
    return {
        "n_trials": total,
        "true_position_counts": {str(i): correct_positions[i] for i in range(1, 6)},
        "predicted_position_counts": {str(i): predicted_positions[i] for i in range(1, 6)},
        "accuracy_by_true_position": {
            str(i): (
                correct_by_position[i] / correct_positions[i]
                if correct_positions[i]
                else None
            )
            for i in range(1, 6)
        },
        "interpretation": (
            "Secondary response-position audit; it does not alter the paper Intruder score."
        ),
    }


def _validate_components_against_extraction(
    extraction_dir: Path,
    db_path: Path,
    *,
    expected_count: int,
) -> None:
    """Prove judge resume did not mutate any scientific component input."""

    contexts_path = extraction_dir / "component_contexts.jsonl"
    manifest_path = extraction_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("component_contexts_sha256") != sha256_file(contexts_path):
        raise ValueError("extraction component-context hash mismatch before judge resume")
    expected_rows = [
        _component(json.loads(line))
        for line in contexts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(expected_rows) != expected_count:
        raise ValueError("extraction component count changed before judge resume")
    db = HarvestDB(db_path, readonly=True)
    try:
        if db.get_component_count() != expected_count:
            raise ValueError("harvest component count changed before judge resume")
        for start in range(0, len(expected_rows), 256):
            batch = expected_rows[start : start + 256]
            observed = db.get_components_bulk([row.component_key for row in batch])
            for expected in batch:
                if observed.get(expected.component_key) != expected:
                    raise ValueError(
                        f"harvest component changed before judge resume: {expected.component_key}"
                    )
    finally:
        db.close()


def _validate_existing_scores(
    db_path: Path,
    *,
    sampled_keys: set[str],
    n_trials: int,
) -> int:
    """Validate every resumable score row written by the paper scorer."""

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT component_key, score_type, score, details FROM scores"
        ).fetchall()
    for key, score_type, score, details in rows:
        if score_type != "intruder" or key not in sampled_keys:
            raise ValueError("harvest DB contains an unexpected judge score")
        payload = json.loads(details)
        trials = payload.get("trials")
        if (
            payload.get("component_key") != key
            or payload.get("n_errors") != 0
            or not isinstance(trials, list)
            or len(trials) != n_trials
        ):
            raise ValueError(f"resumable score details are incomplete: {key}")
        observed_score = sum(bool(trial["is_correct"]) for trial in trials) / n_trials
        if abs(float(score) - observed_score) > 1e-12:
            raise ValueError(f"resumable score value disagrees with its trials: {key}")
    return len(rows)


def main() -> None:
    args = _args()
    protocol = PaperIntruderContract.load(args.protocol)
    meta_path = args.harvest_dir / "harvest_meta.json"
    prepared_path = args.harvest_dir / "PREPARED"
    db_path = args.harvest_dir / "harvest.db"
    for path in (meta_path, prepared_path, db_path):
        if not path.is_file():
            raise FileNotFoundError(f"prepared harvest artifact is missing: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("harvest_meta_sha256") != sha256_file(meta_path):
        raise ValueError("prepared harvest metadata hash mismatch")
    expected = {
        "status": "PAPER_HARVEST_DB_COMPLETE",
        "experiment_family": protocol.experiment_family,
        "checkpoint_step": protocol.checkpoint_step,
        "checkpoint_manifest_sha256": protocol.checkpoint_manifest_sha256,
        "protocol_fingerprint": protocol.fingerprint(),
        "sampled_component_count": protocol.n_subsample,
    }
    mismatch = {
        key: {"expected": value, "observed": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatch:
        raise ValueError(f"prepared harvest provenance mismatch: {mismatch}")
    sampled_keys = set(meta["sampled_keys"])
    current_db_sha256 = sha256_file(db_path)
    existing_score_count = 0
    if current_db_sha256 != meta.get("harvest_db_sha256_before_judging"):
        _validate_components_against_extraction(
            args.extraction_dir,
            db_path,
            expected_count=int(meta["component_count"]),
        )
        existing_score_count = _validate_existing_scores(
            db_path,
            sampled_keys=sampled_keys,
            n_trials=protocol.n_trials,
        )
        if existing_score_count == 0:
            raise ValueError("harvest DB hash changed without any valid resumable scores")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite judge output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "intruder_summary.json"
    intruder_harvest_db(
        args.harvest_dir,
        args.tokenizer,
        llm_config=JudgeConfig(
            base_url=args.judge_base_url,
            model=protocol.judge_model_id,
            structured=False,
            max_concurrent=args.judge_concurrency,
        ),
        n_real=protocol.n_real,
        n_trials=protocol.n_trials,
        density_tolerance=protocol.density_tolerance,
        n_subsample=None,
        seed=protocol.seed,
        keys=list(meta["sampled_keys"]),
        summary_path=summary_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("n_scored") != protocol.n_subsample or summary.get("missing_keys"):
        raise RuntimeError("paper Intruder judge did not score every sampled component")
    audit = _position_audit(db_path)
    _write_json(args.output_dir / "position_audit.json", audit)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "ASPD_V3_1_PAPER_INTRUDER_JUDGE_COMPLETE",
            "experiment_family": protocol.experiment_family,
            "checkpoint_step": protocol.checkpoint_step,
            "checkpoint_manifest_sha256": protocol.checkpoint_manifest_sha256,
            "protocol_fingerprint": protocol.fingerprint(),
            "judge": {
                "model_id": protocol.judge_model_id,
                "revision": protocol.judge_revision,
                "quantization": protocol.judge_quantization,
                "backend": protocol.judge_backend,
            },
            "paper_intruder": {
                "n_subsample": protocol.n_subsample,
                "seed": protocol.seed,
                "n_real": protocol.n_real,
                "n_trials": protocol.n_trials,
                "density_tolerance": protocol.density_tolerance,
                "chance_accuracy": protocol.chance_accuracy,
            },
            "intruder_mean": summary["mean"],
            "n_scored": summary["n_scored"],
            "existing_scores_validated_before_resume": existing_score_count,
            "intruder_summary_sha256": sha256_file(summary_path),
            "position_audit_sha256": sha256_file(args.output_dir / "position_audit.json"),
            "harvest_db_sha256_after_judging": sha256_file(db_path),
            "scientific_status": (
                "EXPLORATORY_UNTIL_POSITION_AUDIT_AND_V3_ADAPTATIONS_ARE_REVIEWED"
            ),
        },
    )
    _write_json(
        args.output_dir / "COMPLETE",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(args.output_dir / "experiment_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()

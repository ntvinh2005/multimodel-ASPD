"""Run the paper's exact Intruder trial builder and judge on a v3.1 harvest DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from aspd.eval.intruder_db import intruder_harvest_db
from aspd.eval.judge import JudgeConfig
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
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
        "harvest_db_sha256_before_judging": sha256_file(db_path),
    }
    mismatch = {
        key: {"expected": value, "observed": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatch:
        raise ValueError(f"prepared harvest provenance mismatch: {mismatch}")
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

"""Convert natural v3.1 firing contexts into the paper's harvest.db schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig
from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
)

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--harvest-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _component(row: dict[str, Any]) -> ComponentData:
    examples = [
        ActivationExample(
            token_ids=[int(value) for value in context["token_ids"]],
            firings=[bool(value) for value in context["firings"]],
            activations={
                "activation": [float(value) for value in context["activations"]]
            },
        )
        for context in row["contexts"]
    ]
    for example in examples:
        if not (
            len(example.token_ids)
            == len(example.firings)
            == len(example.activations["activation"])
        ):
            raise ValueError("activation example arrays have inconsistent lengths")
        if not any(example.firings):
            raise ValueError("activation example contains no naturally firing token")
    return ComponentData(
        component_key=str(row["component_key"]),
        layer=str(row["site"]),
        component_idx=int(row["feature"]),
        mean_activations={
            "activation": fmean(float(context["activation"]) for context in row["contexts"])
        },
        firing_density=float(row["firing_density"]),
        activation_examples=examples,
        input_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
        output_token_pmi=ComponentTokenPMI(top=[], bottom=[]),
    )


def main() -> None:
    args = _args()
    protocol = PaperIntruderContract.load(args.protocol)
    extraction_manifest_path = args.extraction_dir / "experiment_manifest.json"
    extraction_marker_path = args.extraction_dir / "COMPLETE"
    contexts_path = args.extraction_dir / "component_contexts.jsonl"
    keys_path = args.extraction_dir / "sampled_keys.json"
    for path in (
        extraction_manifest_path,
        extraction_marker_path,
        contexts_path,
        keys_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"extraction artifact is missing: {path}")
    extraction_manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    extraction_marker = json.loads(extraction_marker_path.read_text(encoding="utf-8"))
    expected = {
        "status": "ASPD_V3_1_PAPER_INTRUDER_EXTRACTION_COMPLETE",
        "experiment_family": protocol.experiment_family,
        "checkpoint_step": protocol.checkpoint_step,
        "checkpoint_manifest_sha256": protocol.checkpoint_manifest_sha256,
        "accepted_evaluation_anchors": protocol.accepted_evaluation_anchors,
        "natural_gate_only": True,
        "forced_activations": 0,
        "protocol_fingerprint": protocol.fingerprint(),
        "component_contexts_sha256": sha256_file(contexts_path),
        "sampled_keys_sha256": sha256_file(keys_path),
    }
    mismatch = {
        key: {"expected": value, "observed": extraction_manifest.get(key)}
        for key, value in expected.items()
        if extraction_manifest.get(key) != value
    }
    if extraction_marker.get("manifest_sha256") != sha256_file(extraction_manifest_path):
        mismatch["COMPLETE.manifest_sha256"] = "mismatch"
    if mismatch:
        raise ValueError(f"extraction provenance mismatch: {mismatch}")
    if args.harvest_dir.exists():
        raise FileExistsError(f"refusing to overwrite harvest directory: {args.harvest_dir}")
    args.harvest_dir.mkdir(parents=True)
    db = HarvestDB(args.harvest_dir / "harvest.db")
    db.save_config(
        HarvestConfig(
            method_config=ParamDecompHarvestConfig(
                wandb_path="local/aspdv3/p-00130000", activation_threshold=0.0
            ),
            n_batches=protocol.accepted_evaluation_anchors // protocol.pool_anchors,
            batch_size=16,
            activation_examples_per_component=protocol.activation_examples_per_target,
            activation_context_tokens_per_side=protocol.context_tokens_per_side,
        )
    )
    count = db.save_components_iter(
        _component(json.loads(line))
        for line in contexts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    sampled = json.loads(keys_path.read_text(encoding="utf-8"))
    eligible = set(db.get_eligible_component_keys(protocol.minimum_examples_per_component))
    missing = sorted(set(sampled["keys"]) - eligible)
    if missing:
        raise ValueError(f"sampled target components are missing from harvest.db: {missing[:5]}")
    db.close()
    metadata = {
        "schema_version": 1,
        "status": "PAPER_HARVEST_DB_COMPLETE",
        "experiment_family": protocol.experiment_family,
        "checkpoint_step": protocol.checkpoint_step,
        "checkpoint_manifest_sha256": protocol.checkpoint_manifest_sha256,
        "protocol_fingerprint": protocol.fingerprint(),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "component_count": count,
        "eligible_component_count": len(eligible),
        "sampled_component_count": len(sampled["keys"]),
        "harvest_db_sha256_before_judging": sha256_file(args.harvest_dir / "harvest.db"),
        "sampled_keys": sampled["keys"],
        "display_model_key": protocol.display_model_key,
        "firing_semantics": "natural z-score ReLU pool-wide BatchTopK",
    }
    _write_json(args.harvest_dir / "harvest_meta.json", metadata)
    _write_json(
        args.harvest_dir / "PREPARED",
        {
            "status": "PREPARED",
            "harvest_meta_sha256": sha256_file(args.harvest_dir / "harvest_meta.json"),
        },
    )


if __name__ == "__main__":
    main()

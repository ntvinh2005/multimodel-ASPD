"""Prepare exact-aligned controlled Selection Frame prompts and freeze their pool plan."""

from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path

from aspd.multimodel.cli.prepare_aspd_v3_negation import load_tokenizer_contract, read_contract
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json
from aspd.multimodel.v3.negation import pack_groups
from aspd.multimodel.v3.selection_frame import examples, prepare_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, reference = read_contract(args.config)
    registry, tokenizers = load_tokenizer_contract(reference)
    rows = [prepare_example(example, tokenizers) for example in examples()]
    pools = pack_groups([list(group) for _, group in groupby(rows, key=lambda row: row["example"]["group_id"])], 2048)
    args.output.mkdir(parents=True, exist_ok=False)
    examples_path = args.output / "examples.jsonl"
    with examples_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(args.output / "manifest.json", {
        "schemaVersion": 1,
        "dataset": "selection-frame/discovery",
        "checkpointSha256": reference["checkpoint_manifest_sha256"],
        "configSha256": sha256_file(args.config),
        "generatorSha256": sha256_file(Path(__file__).resolve().parents[1] / "v3/selection_frame.py"),
        "examplesSha256": sha256_file(examples_path),
        "models": {
            key: {"id": model.model_id, "revision": model.revision, "tokenizer_revision": model.tokenizer_revision}
            for key, model in registry.items()
        },
        "examples": len(rows),
        "alignedAnchors": sum(len(row["alignment"]["anchors"]) for row in rows),
        "poolExamples": [[row["example"]["example_id"] for row in pool] for pool in pools],
    })


if __name__ == "__main__":
    main()

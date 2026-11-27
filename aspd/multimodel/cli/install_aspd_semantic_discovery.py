"""Verify an immutable revised Semantic Discovery bundle and atomically activate it."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from aspd.multimodel.cli.collect_aspd_semantic_contributions import CHECKPOINT, MODELS
from aspd.multimodel.discovery.semantic_ot import is_heldout
from aspd.multimodel.discovery.semantic_ot_v2 import SPLIT_SEED, coverage, shingles
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json

DIGEST = re.compile(r"[a-f0-9]{64}")


def check_examples(bundle, reference, identity, catalog):
    basename = reference["file"]
    if not re.fullmatch(r"component-[0-9]+\.json\.gz", basename):
        raise ValueError("Unexpected component example asset name")
    asset = bundle / basename
    if sha256_file(asset) != reference["sha256"]:
        raise ValueError(f"Component example hash mismatch: {identity}")
    with gzip.open(asset, "rt") as handle:
        data = json.load(handle)
    if (data["component"], data["runId"]) != (identity, catalog["runId"]):
        raise ValueError(f"Component example identity mismatch: {identity}")
    component = catalog["components"][identity]
    doc_sets = {}
    for split in ("discovery", "heldout"):
        examples = data["examples"][split]
        doc_sets[split] = {ex["document"] for ex in examples if not ex["contrast"]}
        supports = [ex for ex in examples if not ex["contrast"]]
        if len(supports) != len(doc_sets[split]) or len(supports) != component["coverage"][split]:
            raise ValueError(f"Incorrect independent context count: {identity}")
        prefixes = set()
        for row in examples:
            magnitude = row["magnitude"]
            if not math.isfinite(magnitude) or (
                magnitude <= 0 if not row["contrast"] else magnitude != 0
            ):
                raise ValueError(f"Invalid contribution coverage or control: {identity}")
            if is_heldout(row["document"], SPLIT_SEED) != (split == "heldout"):
                raise ValueError(f"Unexpected document split: {identity}")
            anchors = [token for token in row["sequence"] if token["anchor"] == row["anchor"]]
            position, token_id, _, _ = row["native"][component["model"]]
            if (
                len(anchors) != 1
                or anchors[0]["tokenPosition"] != position
                or anchors[0]["tokenId"] != token_id
            ):
                raise ValueError(f"Token heatmap identity mismatch: {identity}")
            if (
                anchors[0]["magnitude"] != magnitude
                or anchors[0]["coefficient"] != row["coefficient"]
            ):
                raise ValueError(f"Token heatmap measurement mismatch: {identity}")
            prefix = "".join(token["text"] for token in row["sequence"])
            if len(prefix) != row["boundary"]:
                raise ValueError(f"Token heatmap boundary mismatch: {identity}")
            if not row["contrast"]:
                signature = shingles(prefix)
                if signature in prefixes:
                    raise ValueError(f"Duplicate causal prefix: {identity}")
                prefixes.add(signature)
    if doc_sets["discovery"] & doc_sets["heldout"]:
        raise ValueError(f"Document leakage: {identity}")


def verify(bundle):
    path = bundle / "catalog.json"
    seal = json.loads((bundle / "COMPLETE.json").read_text())
    if seal["catalogSha256"] != sha256_file(path):
        raise ValueError("Catalog hash mismatch")
    data = json.loads(path.read_text())
    if (
        seal["status"],
        seal["schemaVersion"],
        data["schemaVersion"],
        data["dataset"],
        data["checkpoint"],
    ) != (
        "COVERAGE_VALIDATED",
        2,
        2,
        "natural-validation",
        CHECKPOINT,
    ):
        raise ValueError("Unexpected benchmark or checkpoint")
    if seal["runId"] != data["runId"] or not DIGEST.fullmatch(data["runId"]):
        raise ValueError("Invalid release run identity")
    if data["splitSeed"] != SPLIT_SEED:
        raise ValueError("Unexpected held-out document policy")
    ids = {key for pair in data["pairs"] for key in (pair["left"], pair["right"])}
    if ids != data["components"].keys() or ids != data["examples"].keys():
        raise ValueError("Catalog endpoints and exported components differ")
    if any(not p["retained"] or p["left"] == p["right"] for p in data["pairs"]):
        raise ValueError("Catalog lists an invalid pair")
    actual = coverage(data["components"])
    if actual != seal["summary"] or actual != data["summary"]:
        raise ValueError("Catalog coverage summary does not reproduce")
    if any(comp["model"] not in MODELS for comp in data["components"].values()):
        raise ValueError("Invalid model identity")
    for identity, ref in data["examples"].items():
        check_examples(bundle, ref, identity, data)
    return data


def activate(bundle, workbench):
    data = verify(bundle)
    target = workbench / "semantic-discovery-v2"
    target.mkdir(exist_ok=True)
    installed = target / data["runId"]
    if not installed.exists():
        with tempfile.TemporaryDirectory(prefix=".incoming-", dir=target) as temp:
            stage = Path(temp) / "artifact"
            shutil.copytree(bundle, stage)
            verify(stage)
            stage.rename(installed)
    verify(installed)
    if sha256_file(installed / "COMPLETE.json") != sha256_file(bundle / "COMPLETE.json"):
        raise ValueError("An immutable namespace already has a different bundle")
    with tempfile.TemporaryDirectory(prefix=".semantic-pointer-", dir=workbench) as temp:
        pointer = Path(temp) / "release.json"
        write_json(
            pointer,
            {
                "schemaVersion": 2,
                "runId": data["runId"],
                "completeSha256": sha256_file(installed / "COMPLETE.json"),
            },
        )
        os.replace(pointer, workbench / "semantic-discovery-release.json")
    print(json.dumps(data["summary"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--activate-workbench", type=Path)
    args = parser.parse_args()
    if args.activate_workbench:
        activate(args.bundle, args.activate_workbench)
    else:
        print(json.dumps(verify(args.bundle)["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

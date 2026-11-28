"""Project archived measured Taylor terms onto a frozen semantic partition."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from aspd.multimodel.discovery.index import MODELS, C, Index, M, decode, encode
from aspd.multimodel.discovery.semantic_supernodes import aggregate_taylor
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json


def read(path):
    return json.loads(path.read_text())


def project(pilot, graph_index, recovery, posthoc):
    started = time.monotonic()
    seal = read(pilot / "COMPLETE.json")
    catalog_hash = sha256_file(pilot / "catalog.json")
    if catalog_hash != seal["sha256"]["catalog.json"]:
        raise ValueError("Pilot catalog hash mismatch")
    catalog = read(pilot / "catalog.json")
    manifest = read(graph_index / "manifest.json")
    if manifest["contract"]["checkpoint"] != catalog["checkpoint"]:
        raise ValueError("Graph index checkpoint mismatch")
    for filename, digest in manifest["files"].items():
        if sha256_file(graph_index / filename) != digest:
            raise ValueError(f"Invalid graph index file {filename}")
    recovered = read(recovery / "COMPLETE.json")
    recovery_hash = sha256_file(recovery / "anchors.jsonl")
    if (
        recovered["checkpoint"] != catalog["checkpoint"]
        or recovery_hash != recovered["anchorsSha256"]
        or recovered["edgeCompleteSha256"] != sha256_file(posthoc / "edges" / "COMPLETE")
    ):
        raise ValueError("Recovered graph prefixes have different provenance")
    if (
        sha256_file(posthoc / "profiles" / "COMPLETE") != manifest["contract"]["natural"]
        or recovered["models"] != catalog["measurementProvenance"]["backend"]["models"]
        or recovered["configSha256"]
        != catalog["measurementProvenance"]["backend"]["config_file_sha256"]
    ):
        raise ValueError("Graph index and recovered prefixes use different experiments")
    anchors = {}
    with (recovery / "anchors.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            anchors[row["anchor"]] = row
    index = Index(graph_index)
    membership = {
        key: cluster
        for part in (
            [catalog["jointPartition"]]
            if "jointPartition" in catalog
            else catalog["models"].values()
        )
        for cluster in part["clusters"]
        for key in cluster["members"]
    }
    scales = {c["id"]: c["uNorm"] / c["sigma"] for c in catalog["components"]}
    codes = np.asarray(sorted(map(encode, membership)))
    included = np.zeros(C * M * len(MODELS), dtype=bool)
    included[codes] = True
    rows = []
    scopes = set()
    source_hashes = set()
    for i, scope in enumerate(index.scopes):
        if scope["dataset"] != "natural-validation" or scope["split"] != "discovery-only":
            continue
        if scope["anchor"] not in anchors:
            continue
        edges, _ = index.graph(i)
        keep = included[edges["source"]] & included[edges["target"]]
        if not keep.any():
            continue
        observed = {}
        links = {}
        for edge in edges[keep]:
            b, a = map(decode, (edge["source"], edge["target"]))
            left, right = membership[b], membership[a]
            if left["id"] == right["id"]:
                continue
            expected = abs(float(edge["E"])) * scales[a]
            if not np.isclose(expected, edge["w"], atol=1e-8, rtol=1e-5):
                raise ValueError(f"Target scale disagrees with archived w for {a}")
            observed[b, a] = float(edge["E"])
            links[left["id"], right["id"]] = (left, right)
        if not links:
            continue
        anchor = anchors[scope["anchor"]]
        if scope["pool"] is not None and scope["pool"] != anchor["pool"]:
            raise ValueError("Archive pool mismatch")
        scopes.add((scope["anchor"], scope["model"]))
        source_hashes.add(scope["artifact"])
        for left, right in links.values():
            result = aggregate_taylor(left["members"], right["members"], observed, scales)
            rows.append(
                {
                    "from": left["id"],
                    "to": right["id"],
                    "model": scope["model"],
                    "anchor": scope["anchor"],
                    "scope": {**anchor, "split": "archive-discovery"},
                    "measurement": "archived-top-k",
                    "artifact": scope["artifact"],
                    **result,
                }
            )
    # Publish recurring links with a bounded number of independently documented exact witnesses.
    by_link = defaultdict(list)
    for row in rows:
        by_link[row["from"], row["to"]].append(row)
    evidence = []
    summary = []
    for (left, right), group in sorted(by_link.items()):
        documents = {}
        for row in sorted(
            group, key=lambda r: (-(r["score"] or r["observedTargetMean"] or 0), r["anchor"])
        ):
            documents.setdefault(row["scope"]["document"], row)
        evidence.extend(list(documents.values())[:8])
        summary.append(
            {
                "from": left,
                "to": right,
                "anchors": len(group),
                "documents": len(documents),
                "exported": min(8, len(documents)),
            }
        )
    return {
        "runId": catalog["runId"],
        "checkpoint": catalog["checkpoint"],
        "catalogSha256": catalog_hash,
        "graphIndexSha256": sha256_file(graph_index / "manifest.json"),
        "recoveredAnchorsSha256": recovery_hash,
        "archiveArtifacts": sorted(source_hashes),
        "status": "archived-top-k",
        "measuredScopes": len(scopes),
        "links": summary,
        "witnesses": evidence,
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "graph-index", "recovery", "posthoc", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = project(args.pilot, args.graph_index, args.recovery, args.posthoc)
    write_json(args.output, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("links", "witnesses")}, indent=2)
    )


if __name__ == "__main__":
    main()

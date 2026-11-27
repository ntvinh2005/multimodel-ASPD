"""Attach an exact-anchor graph bundle to an already verified Semantic OT release."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from aspd.multimodel.cli.measure_semantic_ot_graphs import read
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json
from aspd.multimodel.posthoc.ids import ComponentId


def publish(graphs: Path, workbench: Path):
    marker = read(graphs / "COMPLETE.json")
    graph_path = graphs / "graphs.json"
    raw_sha = sha256_file(graph_path)
    if raw_sha != marker["sha256"]:
        raise ValueError("Measured graph artifact changed after sealing")
    graph = read(graph_path)
    release = read(workbench / "semantic-discovery-release.json")
    target = workbench / "semantic-discovery-v2" / release["runId"]
    seal = read(target / "COMPLETE.json")
    catalog_path = target / "catalog.json"
    if (
        sha256_file(target / "COMPLETE.json") != release["completeSha256"]
        or sha256_file(catalog_path) != seal["catalogSha256"]
        or graph["catalogSha256"] != seal["catalogSha256"]
        or graph["runId"] != release["runId"]
        or marker["anchors"] != len(graph["anchors"])
        or marker["targetGraphs"] != len(graph["graphs"])
        or set(graph["anchorOrder"]) != graph["anchors"].keys()
    ):
        raise ValueError("Measured graph bundle does not match the active release")
    catalog = read(catalog_path)
    if (graph["dataset"], graph["checkpoint"]) != (catalog["dataset"], catalog["checkpoint"]):
        raise ValueError("Graph dataset or checkpoint differs")
    graph_rows = {}
    for key, row in graph["graphs"].items():
        target_id = ComponentId.from_key(row["component"])
        anchor = graph["anchors"][row["anchor"]]
        if (
            key != f"{row['component']}|{row['anchor']}"
            or row["component"] not in catalog["components"]
            or row["pool"] != anchor["pool"]
            or row["model"] != target_id.model
        ):
            raise ValueError(f"Invalid graph witness identity: {key}")
        if any(
            edge["target"] != target_id.key
            or not ComponentId.from_key(edge["source"]).matrix.is_upstream_of(target_id.matrix)
            or not math.isfinite(edge["E"])
            or not math.isfinite(edge["w"])
            or edge["w"] < 0
            for edge in row["edges"]
        ):
            raise ValueError(f"Invalid Taylor edge for {key}")
        graph_rows[key] = {
            **{field: row[field] for field in ("anchor", "model", "component", "pool", "edges")},
            "artifact": raw_sha,
        }
    payload = {
        **{
            key: graph[key]
            for key in (
                "schemaVersion",
                "runId",
                "catalogSha256",
                "checkpoint",
                "dataset",
                "anchorOrder",
            )
        },
        "anchors": {
            key: {
                **{field: v for field, v in row.items() if field != "binarySupport"},
                "split": row.get("split", graph.get("split")),
            }
            for key, row in graph["anchors"].items()
        },
        "graphs": graph_rows,
    }
    with TemporaryDirectory(dir=workbench, prefix=".semantic-graphs-") as tmp:
        staged = Path(tmp) / "graphs.json.gz"
        with staged.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())
        sha = sha256_file(staged)
        dest = target / f"graphs-{sha}.json.gz"
        if dest.exists():
            if sha256_file(dest) != sha:
                raise ValueError("Existing graph bundle digest does not match")
        else:
            staged.rename(dest)
        pointer = Path(tmp) / "release.json"
        write_json(
            pointer,
            {
                "catalogSha256": graph["catalogSha256"],
                "runId": graph["runId"],
                "sha256": sha,
            },
        )
        pointer.replace(workbench / "semantic-discovery-graphs-release.json")
    return {"sha256": sha, "anchors": len(payload["anchors"]), "targetGraphs": len(graph_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--workbench", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.graphs, args.workbench), sort_keys=True))


if __name__ == "__main__":
    main()

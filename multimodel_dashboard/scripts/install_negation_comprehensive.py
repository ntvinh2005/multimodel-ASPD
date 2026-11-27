"""Install verified comprehensive groups beside the original Negation workbench assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_negation_workbench import compact_write, read, validate_graph, validate_target

from aspd.multimodel.posthoc.artifacts import sha256_file
from aspd.multimodel.posthoc.ids import ComponentId
from aspd.multimodel.v3.negation import FACTORIAL


def validate_query(row, source, reps):
    (final,) = (position for position in source["positions"] if position["final_query"])
    if row["anchorId"] != final["anchor_id"]:
        raise ValueError("comprehensive query anchor differs from measured context")
    active = {
        f"{site.upper()}:{feature}"
        for site in ("pre", "mid") for feature in final[site]["active_gate_ids"]
    }
    if set(row["graphs"]) != active:
        raise ValueError("comprehensive graph must enumerate every naturally active gate")
    current_reps = [
        rep for rep in reps if f"{rep.gate.site.upper()}:{rep.feature}" in active
    ]
    validate_graph(row, source, current_reps, comprehensive=True)
    official = [
        target
        for graph in row["graphs"].values()
        for model in graph["models"].values()
        for target in model["targets"]
    ]
    trace = row["traceTargets"]
    frontier = {edge["component"]["id"] for record in official for edge in record["parents"]}
    seen = set()
    while new := (frontier & trace.keys()) - seen:
        for key in new:
            target = trace[key]
            if ComponentId.from_key(key) in reps or key != target["component"]["id"]:
                raise ValueError("supplemental target duplicates a frozen representative or has a wrong key")
            frontier.update(parent.key for parent, _ in validate_target(target, final))
        seen.update(new)
    if seen != trace.keys():
        raise ValueError("supplemental target lacks a measured route at this exact anchor")
    return active


def verify_run(run, workbench):
    manifest_path = run / "manifest.json"
    if read(run / "COMPLETE")["manifestSha256"] != sha256_file(manifest_path):
        raise ValueError(f"unverified run: {run}")
    manifest = read(manifest_path)
    if (
        manifest["checkpointSha256"] != workbench["checkpointSha256"]
        or manifest["discoveryManifestSha256"] != workbench["splits"]["discovery"]["manifestSha256"]
        or (manifest["kPar"], manifest["tau"]) != (10, 0)
    ):
        raise ValueError(f"wrong workbench provenance: {run}")
    for group in manifest["groups"]:
        name = f"{group}.json"
        if sha256_file(run / name) != manifest["artifacts"][name]:
            raise ValueError(f"corrupt group {group}")
        yield group, read(run / name), manifest["runtime"]["slurm_job_id"]


def install(workbench, runs, output):
    index = read(workbench / "index.json")
    if index["mechanism"] != "negation" or index["checkpointStep"] != 61036:
        raise ValueError("expected the Negation step-61036 workbench")
    locator_path = workbench / "locator.json"
    if sha256_file(locator_path) != index["assetHashes"]["locator.json"]:
        raise ValueError("frozen representative locator differs from its manifest")
    locator = read(locator_path)
    reps = {
        ComponentId.from_key(
            f'{row["model"]}.L{row["layer"]:02d}.{matrix}/{feature}'
        )
        for row in locator["matrices"]
        for matrix in (
            row["matrix"].lower() if row["matrix"] in ("Q", "K", "V", "O") else f'mlp_{row["matrix"]}',
        )
        for feature in row["representatives"]
    }
    header = {
        "schemaVersion": 1,
        "mechanism": "negation",
        "split": "discovery",
        "checkpointSha256": index["checkpointSha256"],
        "discoveryManifestSha256": index["splits"]["discovery"]["manifestSha256"],
    }
    selected = {}
    for run in runs:
        for group, payload, job in verify_run(run, index):
            selected[group] = (payload, job)
    installed = {}
    output.mkdir(parents=True, exist_ok=False)
    for group, (payload, job) in sorted(selected.items()):
        path = f'discovery/{index["splits"]["discovery"]["groups"][group]["path"]}'
        if (
            payload["groupId"] != group
            or any(payload[key] != value for key, value in header.items())
            or sha256_file(workbench / path) != index["assetHashes"][path]
            or tuple(row["condition"] for row in payload["examples"]) != FACTORIAL
        ):
            raise ValueError(f"invalid matched group: {group}")
        contexts = {row["example_id"]: row for row in read(workbench / path)["examples"]}
        gates = set()
        for row in payload["examples"]:
            gates.update(validate_query(row, contexts[row["exampleId"]], reps))
        name = f"{group}.json"
        compact_write(output / name, payload)
        installed[group] = {
            "path": f"expanded/{name}", "sha256": sha256_file(output / name),
            "gates": sorted(gates), "jobId": job,
        }
    compact_write(output / "index.json", {**header, "groups": installed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install(args.workbench, args.run, args.output)


if __name__ == "__main__":
    main()

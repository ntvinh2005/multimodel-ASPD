"""Run a bounded Natural Validation Semantic OT pilot, using archived measurements only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.index import (
    CHECKPOINT,
    MODELS,
    Index,
    decode,
    encode,
    gate_of,
    load,
    model_of,
    sha,
)
from aspd.multimodel.discovery.semantic_ot import (
    ENCODER,
    ENCODER_FILES,
    REVISION,
    attach_graphs,
    build_supports,
    centroid,
    cluster_pairs,
    comparison,
    control_signature,
    embed,
    prefilter,
    select_control,
    summary_rows,
    without_terminal_unit,
)


def compare_candidates(codes, supports, rows, vectors, cached, neighbors):
    centroids = []
    signatures = []
    for code in codes:
        support = supports[code]["discovery"]
        centroids.append(
            centroid(
                np.array([vectors[cached[i]] for i in support]),
                [weight for _, weight in support.values()],
            )
        )
        signatures.append(control_signature(support, rows, MODELS[model_of(code)]))
    comparisons = {}

    def get(i, j, split):
        key = i, j, split
        if key not in comparisons:
            comparisons[key] = comparison(
                supports[codes[i]][split],
                supports[codes[j]][split],
                vectors,
                cached,
            )
        return comparisons[key]

    pairs = prefilter(codes, np.array(centroids), neighbors)
    discovery = [(i, j, get(i, j, "discovery")[0]["ot"]) for i, j in pairs]
    controls = {(i, j): select_control(i, j, codes, supports, signatures) for i, j in pairs}
    calibration = [
        get(i, info[-1], "discovery")[0]["ot"]
        for (i, _), info in controls.items()
        if info is not None and info[0] <= 0.6
    ]
    if not calibration:
        raise ValueError("No discovery controls passed the predefined covariate-distance bound")
    threshold = float(np.quantile(calibration, 0.10))
    groups = cluster_pairs(len(codes), discovery, threshold)
    return groups, get, controls, threshold, len(calibration), len(pairs)


def present_link(i, j, codes, rows, get, controls, get_suffix):
    result = dict(
        left=decode(codes[i]),
        right=decode(codes[j]),
        sameGate=gate_of(codes[i]) == gate_of(codes[j]),
    )
    control = controls.get((i, j))
    for split in ("discovery", "heldout"):
        score, cells = get(i, j, split)
        baseline = None
        if control is not None and control[0] <= 0.6:
            baseline = get(i, control[-1], split)[0]
        result[split] = dict(
            measures=score,
            controlMeasures=baseline,
            suffixRemovedOt=get_suffix(i, j, split) if get_suffix else None,
            suffixRemovedControlOt=(
                get_suffix(i, control[-1], split) if get_suffix and baseline is not None else None
            ),
            controlComponent=decode(codes[control[-1]]) if baseline is not None else None,
            controlCovariateDistance=control[0] if baseline is not None else None,
            transport=[
                {**row, "left": rows[row["left"]]["anchor"], "right": rows[row["right"]]["anchor"]}
                for row in cells
            ],
        )
    return result


def publish(args, inputs, rows, supports, vectors, cached, coverage, eligible, runtime, suffix):
    def suffix_score(i, j, split):
        left, right = (supports[codes[k]][split] for k in (i, j))
        score, _ = comparison(left, right, suffix, cached)
        return score["ot"] if score else None

    codes = sorted(supports)
    groups, get, controls, threshold, calibration, tested = compare_candidates(
        codes,
        supports,
        rows,
        vectors,
        cached,
        args.neighbors,
    )
    clusters = []
    retained_anchors = set()
    selected_codes = set()
    index = Index(args.graph_index)
    scopes = {
        (x["anchor"], x["model"]): i
        for i, x in enumerate(index.scopes)
        if x["dataset"] == "natural-validation"
    }
    for indices in groups:
        members = [codes[i] for i in indices]
        selected_codes.update(members)
        links = [
            present_link(
                i, j, codes, rows, get, controls, suffix_score if suffix is not None else None
            )
            for i, j in combinations(indices, 2)
        ]
        graphs = attach_graphs(index, scopes, rows, members, supports)
        for graph in graphs:
            graph["anchor"] = rows[graph["anchor"]]["anchor"]
            retained_anchors.add(graph["anchor"])
        for link in links:
            for split in ("discovery", "heldout"):
                for cell in link[split]["transport"]:
                    retained_anchors.update((cell["left"], cell["right"]))
        clusters.append(
            dict(
                components=[decode(code) for code in members],
                models=sorted({MODELS[model_of(code)] for code in members}),
                links=links,
                graphs=graphs,
            )
        )
    # Sort by observable evidence, keeping held-out data out of cluster membership and ID assignment.
    clusters.sort(
        key=lambda row: (
            -len(row["models"]),
            -len(row["components"]),
            min(link["discovery"]["measures"]["ot"] for link in row["links"]),
            row["components"],
        )
    )
    for number, cluster in enumerate(clusters, 1):
        cluster["id"] = f"SOT-{number:03d}"
    components = {}
    for code in sorted(selected_codes):
        examples = {}
        for split, part in supports[code].items():
            order = sorted(part.items(), key=lambda item: (-item[1][1], item[0]))
            examples[split] = [
                dict(anchor=rows[i]["anchor"], coefficient=e, normalizedMagnitude=weight)
                for i, (e, weight) in order
            ]
            retained_anchors.update(item["anchor"] for item in examples[split])
        components[decode(code)] = dict(
            gate=gate_of(code),
            model=MODELS[model_of(code)],
            totals=summary_rows(supports[code]),
            examples=examples,
        )
    old_catalog = load(args.structural_catalog)
    if old_catalog["provenance"]["checkpoint"] != CHECKPOINT:
        raise ValueError("Structural catalog checkpoint differs")
    overlaps = []
    for candidate in old_catalog["candidates"]:
        if candidate["dataset"] != "natural-validation":
            continue
        for cluster in clusters:
            shared = set(candidate["componentIds"]) & set(cluster["components"])
            if len(shared) >= 2:
                overlaps.append(
                    dict(
                        cluster=cluster["id"], structural=candidate["id"], components=sorted(shared)
                    )
                )
    signs = {}
    for split in ("discovery", "heldout"):
        links = [link[split] for c in clusters for link in c["links"]]
        compared = [link for link in links if link["measures"] and link["controlMeasures"]]
        signs[split] = dict(
            links=len(links),
            scoreable=sum(link["measures"] is not None for link in links),
            matchedControl=len(compared),
            smallerOt=sum(x["measures"]["ot"] < x["controlMeasures"]["ot"] for x in compared),
            smallerCentroid=sum(
                x["measures"]["centroid"] < x["controlMeasures"]["centroid"] for x in compared
            ),
            smallerSuffixRemovedOt=sum(
                x["suffixRemovedOt"] < x["suffixRemovedControlOt"]
                for x in compared
                if x["suffixRemovedOt"] is not None and x["suffixRemovedControlOt"] is not None
            ),
            suffixCompared=sum(x["suffixRemovedControlOt"] is not None for x in compared),
            higherExactOverlap=sum(
                x["measures"]["anchorOverlap"] > x["controlMeasures"]["anchorOverlap"]
                for x in compared
            ),
        )
    summary = dict(
        eligible={MODELS[k]: v for k, v in eligible.items()},
        selected=dict(Counter(MODELS[model_of(code)] for code in codes)),
        scoredDiscoveryPairs=tested,
        calibratedControls=calibration,
        discoveryCutoff=threshold,
        clusters=len(clusters),
        crossModelClusters=sum(len(c["models"]) > 1 for c in clusters),
        structuralOverlaps=len(overlaps),
        comparison=signs,
        coverage=coverage,
        runtime=runtime,
    )
    return dict(
        schemaVersion=1,
        dataset="natural-validation",
        provenance=inputs,
        summary=summary,
        clusters=clusters,
        components=components,
        structuralOverlaps=overlaps,
        anchors={
            row["anchor"]: {k: v for k, v in row.items() if k != "prefixSha256"}
            for row in rows
            if row["anchor"] in retained_anchors
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "recovery", "snapshot", "graph-index", "structural-catalog", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--components-per-model", type=int, default=12)
    parser.add_argument("--structural-seeds-per-model", type=int, default=0)
    parser.add_argument("--suffix-check", action="store_true")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=61036)
    args = parser.parse_args()
    maximum = min(24, len(os.sched_getaffinity(0)), int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
    if not 1 <= args.threads <= maximum or args.components_per_model < 2 or args.neighbors < 1:
        parser.error(
            f"Use between 1 and {maximum} CPU threads and at least two components per model"
        )
    start = time.monotonic()
    inputs = load(args.recovery / "COMPLETE.json")
    if inputs["checkpoint"] != CHECKPOINT or inputs["split"] != "validation":
        raise ValueError("Semantic OT pilot is restricted to the frozen Natural Validation archive")
    if sha(args.recovery / "anchors.jsonl") != inputs["anchorsSha256"]:
        raise ValueError("Natural Validation prefix cache changed")
    if sha(args.archive / "edges" / "COMPLETE") != inputs["edgeCompleteSha256"]:
        raise ValueError("Natural Validation edge archive differs from anchor recovery")
    graph_input = load(args.graph_index / "manifest.json")
    if graph_input["contract"]["checkpoint"] != CHECKPOINT:
        raise ValueError("Graph index checkpoint differs")
    inputs = dict(
        **inputs,
        recoverySha256=sha(args.recovery / "COMPLETE.json"),
        encoder=ENCODER,
        encoderRevision=REVISION,
        embeddingFiles={p.name: sha(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()},
        graphIndexSha256=sha(args.graph_index / "manifest.json"),
        structuralCatalogSha256=sha(args.structural_catalog),
        seed=args.seed,
        componentsPerModel=args.components_per_model,
        centroidsPerModel=args.neighbors,
        structuralSeedsPerModel=args.structural_seeds_per_model,
        suffixCheck=args.suffix_check,
        numpyVersion=np.__version__,
        implementationSha256=sha(Path(__file__)),
    )
    if inputs["embeddingFiles"] != ENCODER_FILES:
        raise ValueError("MiniLM snapshot files differ from the pinned encoder")
    with (args.recovery / "anchors.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    out = args.output
    out.mkdir()
    old = load(args.structural_catalog)
    structural = {
        encode(cid)
        for c in old["candidates"]
        if c["dataset"] == "natural-validation"
        for cid in c["componentIds"]
    }
    supports, coverage, eligible = build_supports(
        args.archive,
        rows,
        args.seed,
        args.components_per_model,
        structural,
        args.structural_seeds_per_model,
    )
    if set(model_of(code) for code in supports) != {0, 1, 2}:
        raise ValueError("Insufficient comparable observations across the three models")
    anchors = sorted(
        {idx for by_split in supports.values() for part in by_split.values() for idx in part}
    )
    unique = {}
    cached = {}
    for idx in anchors:
        value = rows[idx]["prefix"]
        cached[idx] = unique.setdefault(value, len(unique))
    before_embedding = time.monotonic()
    print(
        f"Embedding {len(unique)} exact causal prefixes from {len(supports)} components", flush=True
    )
    keys = list(unique)
    prefix_ids = {row["prefix"]: row["prefixSha256"] for row in rows if row["prefix"] in unique}
    previous = {}
    if args.embedding_cache:
        seal = load(args.embedding_cache / "COMPLETE.json")
        previous_data = load(args.embedding_cache / "catalog.json", seal["catalogSha256"])[
            "provenance"
        ]
        if previous_data["embeddingFiles"] != inputs["embeddingFiles"]:
            raise ValueError("Embedding cache uses different encoder files")
        cache_ix = load(args.embedding_cache / "embedding-index.json", seal["embeddingIndexSha256"])
        if sha(args.embedding_cache / "embeddings.npy") != seal["prefixEmbeddingSha256"]:
            raise ValueError("Embedding cache vectors changed")
        existing = np.load(args.embedding_cache / "embeddings.npy", mmap_mode="r")
        previous = {
            key: existing[cache_ix[prefix_ids[key]]] for key in keys if prefix_ids[key] in cache_ix
        }
        inputs["reusedEmbeddingCacheSha256"] = sha(args.embedding_cache / "COMPLETE.json")
    uncached = [key for key in keys if key not in previous]
    secondary = [without_terminal_unit(key) for key in keys] if args.suffix_check else []
    pending = uncached + secondary
    new = embed(pending, args.snapshot, threads=args.threads) if pending else np.empty((0, 384))
    reused = dict(previous)
    reused.update(zip(uncached, new[: len(uncached)], strict=True))
    vectors = np.array([reused[key] for key in keys])
    suffix = new[len(uncached) :] if secondary else None
    np.save(out / "embeddings.npy", vectors)
    if suffix is not None:
        np.save(out / "suffix-controlled-embeddings.npy", suffix)
    atomic_json(
        out / "embedding-index.json", {rows[idx]["prefixSha256"]: cached[idx] for idx in anchors}
    )
    runtime = dict(
        recoveredAnchorInventory=len(rows),
        embeddingInputs=len(unique),
        reusedEmbeddings=len(previous),
        newlyEmbedded=len(pending),
        preparationSeconds=before_embedding - start,
        embeddingSeconds=time.monotonic() - before_embedding,
        threads=args.threads,
    )
    artifact = publish(
        args, inputs, rows, supports, vectors, cached, coverage, eligible, runtime, suffix
    )
    runtime.update(
        wallSeconds=time.monotonic() - start,
        peakKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic_json(out / "catalog.json", artifact)
    with (out / "review.csv").open("x") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Cluster",
                "Models",
                "Components",
                "Discovery OT",
                "Held-out OT",
                "Held-out control OT",
                "Status",
            ]
        )
        for cluster in artifact["clusters"]:
            for link in cluster["links"]:
                heldout = link["heldout"]
                writer.writerow(
                    [
                        cluster["id"],
                        ", ".join(cluster["models"]),
                        f"{link['left']} ↔ {link['right']}",
                        link["discovery"]["measures"]["ot"],
                        (heldout["measures"] or {}).get("ot"),
                        (heldout["controlMeasures"] or {}).get("ot"),
                        "Unvalidated",
                    ]
                )
    atomic_json(
        out / "COMPLETE.json",
        dict(
            catalogSha256=sha(out / "catalog.json"),
            prefixEmbeddingSha256=sha(out / "embeddings.npy"),
            embeddingIndexSha256=sha(out / "embedding-index.json"),
            suffixEmbeddingSha256=sha(out / "suffix-controlled-embeddings.npy")
            if suffix is not None
            else None,
            reviewSha256=sha(out / "review.csv"),
        ),
    )
    print(json.dumps(artifact["summary"]), flush=True)


if __name__ == "__main__":
    main()

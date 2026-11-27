"""Rerun all three circuit searches at sampled Natural Validation positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json, evaluate, load_seeds
from aspd.multimodel.discovery.index import CHECKPOINT, MODELS, Index, load, model_of, sha
from aspd.multimodel.discovery.search import (
    Graph,
    bottom_up,
    composition,
    grow,
    signature,
    structure,
)
from aspd.multimodel.discovery.semantic_ot import build_supports, is_heldout


def sample_scope_ids(scopes, limit, seed):
    buckets = defaultdict(list)
    for i, row in enumerate(scopes):
        buckets[row["model"], row["split"]].append(i)
    result = []
    for indices in buckets.values():
        documents = set()
        for i in sorted(
            indices, key=lambda k: hashlib.sha256(f"{seed}:{scopes[k]['anchor']}".encode()).digest()
        ):
            document = scopes[i]["group"]
            if document not in documents:
                result.append(i)
                documents.add(document)
            if len(documents) == limit:
                break
    return result


class NaturalView:
    def __init__(self, index, anchors, seed, cap):
        self.index = index
        prefixes = {
            row["prefixSha256"] for row in anchors.values() if not is_heldout(row["document"], seed)
        }
        self.original = []
        self.scopes = []
        for i, scope in enumerate(index.scopes):
            if scope["dataset"] != "natural-validation" or scope["anchor"] not in anchors:
                continue
            row = anchors[scope["anchor"]]
            if len(row["prefix"].split()) < 12:
                continue
            heldout = is_heldout(row["document"], seed)
            if heldout and row["prefixSha256"] in prefixes:
                continue
            self.original.append(i)
            self.scopes.append(
                {
                    **scope,
                    "split": "heldout" if heldout else "discovery",
                    "text": row["prefix"],
                    "boundary": row["boundary"],
                    "pool": str(row["pool"]),
                    "positions": row["positions"],
                    "group": row["document"],
                    "example": row["document"],
                }
            )
        keep = sample_scope_ids(self.scopes, cap, seed)
        self.lookup = {(row["anchor"], row["model"]): i for i, row in enumerate(self.scopes)}
        self.keep = set(keep)

    def retain(self, extra):
        self.keep.update(extra)
        self.original = [self.original[i] for i in sorted(self.keep)]
        self.scopes = [self.scopes[i] for i in sorted(self.keep)]
        # materialize only the sampled edge slices to preserve scope numbering for postings
        parts = [self.index.graph(i)[0] for i in self.original]
        self.edge_starts = np.r_[0, np.cumsum([len(part) for part in parts])[:-1]]
        self.edges = np.concatenate(parts) if parts else np.empty(0, dtype=self.index.edges.dtype)

    def graph(self, i):
        return self.index.graph(self.original[i])


def matched_sinks(supports, rows):
    ranked = defaultdict(list)
    for code, by_split in supports.items():
        model = MODELS[model_of(code)]
        groups = defaultdict(list)
        for anchor, (e, weight) in by_split["discovery"].items():
            row = rows[anchor]
            key = row["source"], row["tokenIds"][model], row["positions"][model] // 64
            groups[key].append((anchor, e, weight))
        diffs = []
        for group in groups.values():
            order = sorted(group, key=lambda row: row[2])
            for lo, hi in zip(order[: len(order) // 2], reversed(order), strict=False):
                if rows[lo[0]]["document"] != rows[hi[0]]["document"]:
                    diffs.append((hi[2] - lo[2], hi[1] - lo[1], hi[0]))
        if len(diffs) >= 2:
            ranked[model].append((median(x[0] for x in diffs), code, diffs))
    return [
        row for values in ranked.values() for row in sorted(values, reverse=True)[:12] if row[0] > 0
    ]


def search(view, templates, sinks):
    starts, by_edge = defaultdict(set), defaultdict(list)
    for i, (_, edges) in enumerate(templates):
        starts[MODELS[model_of(edges[0][0])]].add(edges[0][0])
        for edge in edges:
            by_edge[edge].append(i)
    ledger = {}
    attempts, proposed = Counter(), Counter()
    for i, meta in enumerate(view.scopes):
        if meta["split"] != "discovery":
            continue
        edges, _ = view.graph(i)
        if len(edges) < 4:
            continue
        graph = Graph.from_rows(edges)
        ops = {
            "bottom-up": bottom_up(graph, sorted(starts[meta["model"]])),
            "composition": composition(graph, templates, by_edge),
        }
        for order, (strength, code, diffs) in enumerate(sinks):
            if not graph.parents.get(code):
                continue
            found, count = grow(graph, {code}, upstream=True)
            attempts["top-down"] += count
            ops[f"top-down/{order}"] = (
                {
                    edge: dict(
                        sink=code,
                        normalizedContrast=strength,
                        signedCoefficientContrast=median(item[1] for item in diffs),
                        discoveryComparisons=len(diffs),
                        contrastedConditions=["larger contribution", "smaller contribution"],
                    )
                    for edge in found
                },
                0,
            )
        for key, (found, count) in ops.items():
            method = key.split("/")[0]
            attempts[method] += count
            proposed[method] += len(found)
            for exact, metadata in found.items():
                key = signature(exact)
                row = ledger.setdefault(
                    key,
                    dict(
                        id=key,
                        edges=exact,
                        firstScope=i,
                        occurrences=set(),
                        directions=set(),
                        metadata={},
                        structure=structure(exact),
                    ),
                )
                row["directions"].add(method)
                row["occurrences"].add(i)
                row["metadata"].setdefault(method, metadata)
    shortlist = {}
    for model in MODELS:
        for method in ("bottom-up", "top-down", "composition"):
            cohort = (
                row
                for row in ledger.values()
                if method in row["directions"] and view.scopes[row["firstScope"]]["model"] == model
            )
            ordered = sorted(
                cohort, key=lambda r: (-len(r["occurrences"]), -r["structure"]["depth"], r["id"])
            )
            shortlist.update((row["id"], row) for row in ordered[:5])
    return list(shortlist.values()), dict(
        distinctProposals=len(ledger), proposed=proposed, expansions=attempts
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "recovery", "graph-index", "circuits", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--positions-per-model-split", type=int, default=72)
    parser.add_argument("--seed", type=int, default=61036)
    args = parser.parse_args()
    if not 1 <= args.positions_per_model_split <= 128:
        parser.error("The local pilot supports 1–128 positions per model and split")
    start = time.monotonic()
    manifest = load(args.recovery / "COMPLETE.json")
    if (
        manifest["checkpoint"] != CHECKPOINT
        or manifest["split"] != "validation"
        or sha(args.recovery / "anchors.jsonl") != manifest["anchorsSha256"]
        or sha(args.archive / "edges" / "COMPLETE") != manifest["edgeCompleteSha256"]
    ):
        raise ValueError("Natural Validation provenance mismatch")
    graph_manifest = load(args.graph_index / "manifest.json")
    if graph_manifest["contract"]["checkpoint"] != CHECKPOINT:
        raise ValueError("Graph index checkpoint differs")
    with (args.recovery / "anchors.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    supports, _, _ = build_supports(args.archive, rows, args.seed, 256)
    sinks = matched_sinks(supports, rows)
    view = NaturalView(
        Index(args.graph_index),
        {row["anchor"]: row for row in rows},
        args.seed,
        args.positions_per_model_split,
    )
    extra = []
    for _, code, diffs in sinks:
        model = MODELS[model_of(code)]
        keys = [(rows[i]["anchor"], model) for _, _, i in sorted(diffs, reverse=True)[:3]]
        extra.extend(view.lookup[key] for key in keys if key in view.lookup)
    view.retain(extra)
    templates, independent = load_seeds([args.circuits], args.evidence)
    candidates, search_summary = search(view, templates, sinks)
    output = evaluate(view, candidates, templates, independent, args.seed)
    summary = dict(
        **search_summary,
        measuredScopes=len(view.scopes),
        matchedSinks=len(sinks),
        comparedScopes={
            f"{m}/{s}": sum(row["model"] == m and row["split"] == s for row in view.scopes)
            for m in MODELS
            for s in ("discovery", "heldout")
        },
        candidates=len(output),
        runtimeSeconds=time.monotonic() - start,
        peakKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    args.output.mkdir()
    atomic_json(
        args.output / "catalog.json",
        dict(
            schemaVersion=1,
            candidates=output,
            provenance=dict(
                checkpoint=CHECKPOINT,
                dataset="natural-validation",
                sourceSha256=sha(args.recovery / "COMPLETE.json"),
                seed=args.seed,
                graphIndexSha256=sha(args.graph_index / "manifest.json"),
                implementationSha256=sha(Path(__file__)),
                summary=summary,
            ),
        ),
    )
    atomic_json(
        args.output / "COMPLETE.json",
        dict(catalogSha256=sha(args.output / "catalog.json"), **summary),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

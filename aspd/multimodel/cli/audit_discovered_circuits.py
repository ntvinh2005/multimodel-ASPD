"""Audit shortlisted circuits for repeated prefixes without reopening or changing discovery."""

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.index import C, Index, encode, gate_of, load, sha
from aspd.multimodel.discovery.search import measured_paths, recurring


def prefix_key(scope):
    text, boundary = scope.get("text"), scope.get("boundary")
    if text is None or boundary is None:
        return None
    prefix = text[:boundary]
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def prefix_counts(index, matched, discovery_prefixes):
    known = {"heldout": set(), "discovery": set()}
    clean, overlapping = set(), set()
    distinct_groups = set()
    for i in matched:
        scope = index.scopes[i]
        phase = scope["split"]
        key = prefix_key(scope)
        if phase not in known or key is None:
            continue
        known[phase].add(key)
        if phase == "heldout":
            if key in discovery_prefixes:
                overlapping.add(key)
            else:
                clean.add(key)
                distinct_groups.add((scope["dataset"], scope["group"]))
    return {
        "distinctDiscoveryPrefixes": len(known["discovery"]),
        "distinctHeldoutPrefixes": len(known["heldout"]),
        "heldoutPrefixesSeenInDiscovery": len(overlapping),
        "unseenHeldoutPrefixes": len(clean),
        "heldoutGroupsWithUnseenPrefixes": len(distinct_groups),
    }


def rewire_block(edges, membership, rng, attempts):
    """Directed two-edge swaps preserve per-node degrees within each matrix pair."""
    accepted = 0
    if len(edges) < 2:
        return accepted
    for _ in range(attempts):
        i, j = rng.sample(range(len(edges)), 2)
        a, b = edges[i]
        c, d = edges[j]
        x, y = (a, d), (c, b)
        if x in membership or y in membership:
            continue
        membership.remove(edges[i])
        membership.remove(edges[j])
        membership.update((x, y))
        edges[i], edges[j] = x, y
        accepted += 1
    return accepted


def degree_null(observed, required, seed, trials=64):
    names = {(a // C, b // C) for a, b in required}
    blocks = defaultdict(list)
    for a, b, _, _ in observed.tolist():
        key = a // C, b // C
        if key in names:
            blocks[key].append((a, b))
    memberships = {key: set(values) for key, values in blocks.items()}
    rng = random.Random(seed)
    hits = swaps = 0
    for _ in range(trials):
        for key, values in blocks.items():
            swaps += rewire_block(values, memberships[key], rng, len(values))
        hits += all((a, b) in memberships[a // C, b // C] for a, b in required)
    return {
        "trials": trials if swaps else 0,
        "hits": hits if swaps else None,
        "acceptedSwaps": swaps,
        "matrixPairBlocks": len(blocks),
        "method": "directed degree-preserving swaps within matrix pairs",
    }


def coverage_contexts(index, observed, required, full_nodes, comparable):
    """Allow a partial graph as a control only if it measured every required downstream."""
    gates = {gate_of(node) for edge in required for node in edge}
    targets = {target for _, target in required}
    slots = {
        (index.scopes[i]["dataset"], index.scopes[i]["model"], index.scopes[i].get("role"))
        for i in observed
        if index.scopes[i].get("role")
    }
    counts, controls = {}, []
    for slot in sorted(slots):
        for i in comparable[slot]:
            scope = index.scopes[i]
            key = scope["dataset"], scope["family"], scope["split"], scope["condition"]
            tally = counts.setdefault(key, [0, 0, 0, 0])
            tally[0] += 1
            if not gates <= scope["active"].keys():
                continue
            tally[1] += 1
            if not targets <= full_nodes[i]:
                continue
            tally[2] += 1
            if i in observed:
                tally[3] += 1
            else:
                controls.append(
                    {
                        name: scope[name]
                        for name in (
                            "anchor",
                            "pool",
                            "dataset",
                            "condition",
                            "group",
                            "split",
                            "text",
                            "boundary",
                        )
                    }
                )
    return {
        "denominators": [
            {
                "dataset": dataset,
                "family": family,
                "split": split,
                "condition": condition,
                "roleAnchors": a,
                "allGatesActive": b,
                "requiredTargetsMeasured": c,
                "exactCircuitPresent": d,
            }
            for (dataset, family, split, condition), (a, b, c, d) in sorted(counts.items())
        ],
        "counterexamples": sorted(
            controls, key=lambda row: (row["split"] != "heldout", row["anchor"])
        )[:3],
    }


def audit(run, output, review=None):
    original = load(run / "catalog.json")
    index = Index(run / "index")
    candidates = original["candidates"]
    edges = [
        [(encode(edge["source"]), encode(edge["target"])) for edge in row["witnesses"][0]["edges"]]
        for row in candidates
    ]
    postings = measured_paths(index, edges)
    discovery_prefixes = {
        result
        for scope in index.scopes
        if scope["split"] == "discovery"
        if (result := prefix_key(scope)) is not None
    }
    matched_slots = defaultdict(list)
    measured_targets = {}
    for i, scope in enumerate(index.scopes):
        if scope["split"] in {"discovery", "heldout"} and scope.get("role"):
            matched_slots[scope["dataset"], scope["model"], scope["role"]].append(i)
            measured_targets[i] = set(index.graph(i)[1]["id"].tolist())
    notes = load(review) if review else {}
    if notes.keys() - {row["id"] for row in candidates}:
        raise ValueError("Catalog does not contain every reviewed exact candidate")
    for row, required in zip(candidates, edges, strict=True):
        matches = recurring(postings, required)
        if len(matches) != row["anchorCount"]:
            raise ValueError(f"Stored recurrence differs for {row['id']}")
        row["prefixAudit"] = prefix_counts(index, matches, discovery_prefixes)
        row["measuredContextSelectivity"] = coverage_contexts(
            index, matches, required, measured_targets, matched_slots
        )
        if row["id"] in notes:
            row["interpretation"].update(notes[row["id"]])
        exact = row["witnesses"][0]
        scope_id = next(
            i
            for i in sorted(matches)
            if (
                index.scopes[i]["dataset"],
                index.scopes[i]["anchor"],
                index.scopes[i]["model"],
                index.scopes[i]["pool"],
            )
            == (exact["dataset"], exact["anchor"], exact["model"], exact["pool"])
        )
        row["conditionalGraphNull"] = degree_null(
            index.graph(scope_id)[0], required, int(row["id"][4:], 16)
        )
        row["conditionalGraphNull"]["anchor"] = exact["anchor"]
    original["provenance"]["inputCatalogSha256"] = sha(run / "catalog.json")
    original["provenance"]["prefixAuditSourceSha256"] = sha(Path(__file__))
    original["provenance"]["conditionalNull"] = (
        "Within-witness degree-preserving rewiring conditional on fixed naturally active nodes and matrix pairs; exploratory."
    )
    original["provenance"]["prefixAudit"] = (
        "Exact text prefixes ending at the verified anchor; Natural Validation without text is excluded."
    )
    if review:
        original["provenance"]["hypothesesSha256"] = sha(review)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "catalog.json", original)
    summary = {
        "candidates": len(candidates),
        "degreeNullAvailable": sum(
            bool(row["conditionalGraphNull"]["trials"]) for row in candidates
        ),
        "withHeldoutPrefix": sum(
            row["prefixAudit"]["distinctHeldoutPrefixes"] > 0 for row in candidates
        ),
        "withUnseenHeldoutPrefix": sum(
            row["prefixAudit"]["unseenHeldoutPrefixes"] > 0 for row in candidates
        ),
        "heldoutPrefixAliasOnly": sum(
            bool(row["prefixAudit"]["heldoutPrefixesSeenInDiscovery"])
            and not row["prefixAudit"]["unseenHeldoutPrefixes"]
            for row in candidates
        ),
    }
    atomic_json(output / "summary.json", summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    audit(args.run, args.output, args.review)


if __name__ == "__main__":
    main()

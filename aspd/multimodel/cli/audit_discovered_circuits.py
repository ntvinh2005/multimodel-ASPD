"""Audit shortlisted circuits for repeated prefixes without reopening or changing discovery."""

import argparse
import hashlib
from pathlib import Path

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.index import Index, encode, load, sha
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


def audit(run, output):
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
    for row, required in zip(candidates, edges, strict=True):
        matches = recurring(postings, required)
        if len(matches) != row["anchorCount"]:
            raise ValueError(f"Stored recurrence differs for {row['id']}")
        row["prefixAudit"] = prefix_counts(index, matches, discovery_prefixes)
    original["provenance"]["inputCatalogSha256"] = sha(run / "catalog.json")
    original["provenance"]["prefixAuditSourceSha256"] = sha(Path(__file__))
    original["provenance"]["prefixAudit"] = (
        "Exact text prefixes ending at the verified anchor; Natural Validation without text is excluded."
    )
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "catalog.json", original)
    summary = {
        "candidates": len(candidates),
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
    args = parser.parse_args()
    audit(args.run, args.output)


if __name__ == "__main__":
    main()

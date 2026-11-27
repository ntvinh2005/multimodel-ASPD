"""Rank fresh Semantic OT pairs using discovery-only selection and held-out controls."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from aspd.multimodel.cli.collect_aspd_semantic_contributions import verified_json
from aspd.multimodel.discovery.semantic_ot import ENCODER, REVISION, centroid
from aspd.multimodel.discovery.semantic_ot_v2 import (
    MIN_HELDOUT,
    SPLIT_SEED,
    VARIANTS,
    Anchor,
    bootstrap_advantage,
    comparison,
    coverage,
    diverse,
    embed_variants,
    lexical_key,
    matched_controls,
    select_for_ot,
)
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json

SPLITS = ("discovery", "heldout")
NEIGHBORS = 4
MAX_CONFIRMATORY = 48
BOOTSTRAP_DRAWS = 99


def read_measurements(root):
    contract = json.loads((root / "contract.json").read_text())
    contract_hash = sha256_file(root / "contract.json")
    pools = sorted(root.glob("pool-[0-9]*"))
    if len(pools) not in contract["waves"]:
        raise ValueError("Analysis needs one fully measured predefined wave")
    rows = []
    blocks = []
    anchors_by_doc = defaultdict(list)
    cutoffs = {}
    reports = []
    initial_document = None
    for expected_pool, directory in enumerate(pools, start=contract["firstPool"]):
        info = verified_json(directory, "meta.json")
        reports.append(info)
        if info["contractSha256"] != contract_hash or info["pool"] != expected_pool:
            raise ValueError(f"Noncontiguous or mismatched pool at {directory}")
        with gzip.open(directory / "anchors.json.gz", "rt", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            block = arrays["coefficients"]
            gates = arrays["gates"]
        n = len(metadata["anchors"])
        components = contract["components"]
        gate_columns = {gate: i for i, gate in enumerate(contract["gateColumns"])}
        if block.shape != (n, len(components)) or gates.shape != (n, len(gate_columns)):
            raise ValueError(f"Measurement shape mismatch at {directory}")
        if n != info["anchors"] or n != contract["backend"]["pool_anchors"]:
            raise ValueError(f"Incomplete frozen pool at {directory}")
        if not np.isfinite(block).all() or not np.isfinite(gates).all() or np.any(gates < 0):
            raise ValueError(f"Invalid measured values at {directory}")
        for i, component in enumerate(components):
            if np.any(block[gates[:, gate_columns[component["gate"]]] == 0, i]):
                raise ValueError(
                    f"Nonzero coefficient outside the measured natural support at {directory}"
                )
        if initial_document is None:
            # This document can straddle the last old and first fresh pools.
            initial_document = metadata["documents"][0]["id"]
        retained_rows = []
        for row_index, (anchor_id, doc_index, boundary, native) in enumerate(metadata["anchors"]):
            doc = metadata["documents"][doc_index]
            if doc["id"] == initial_document:
                continue
            retained_rows.append(row_index)
            anchors_by_doc[doc["id"]].append(len(rows))
            rows.append(
                Anchor(
                    anchor_id,
                    doc["id"],
                    doc["source"],
                    expected_pool,
                    boundary,
                    doc["text"][:boundary],
                    native,
                )
            )
            cutoffs[doc["id"]] = doc["text"]
        blocks.append(block[retained_rows])
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Duplicate anchors across pools")
    return contract, contract_hash, rows, np.concatenate(blocks), anchors_by_doc, cutoffs, reports


def support_sets(rows, coefficients):
    row_splits = np.asarray([row.split for row in rows])
    zero_sample = {}
    rng = np.random.default_rng(SPLIT_SEED)
    for split in SPLITS:
        indices = np.flatnonzero(row_splits == split)
        zero_sample[split] = rng.permutation(indices)[:24000]
    supports, sampled, measured = {}, {}, {}
    for c in range(coefficients.shape[1]):
        values = coefficients[:, c]
        nonzero = np.flatnonzero(values)
        measured[c] = {
            split: int(np.count_nonzero(row_splits[nonzero] == split)) for split in SPLITS
        }
        discovery = diverse(rows, values, nonzero[row_splits[nonzero] == "discovery"])
        heldout = diverse(
            rows, values, nonzero[row_splits[nonzero] == "heldout"], forbidden=discovery
        )
        supports[c] = {"discovery": discovery, "heldout": heldout}
        if len(discovery) < MIN_HELDOUT or len(heldout) < MIN_HELDOUT:
            continue
        sampled[c] = {
            split: select_for_ot(rows, indices, salt=split)
            for split, indices in supports[c].items()
        }
    return supports, sampled, zero_sample, measured


def controls_for(rows, coefficients, components, sampled, zero_sample):
    out = {}
    for i, bags in sampled.items():
        out[i] = {}
        for split in SPLITS:
            indices = zero_sample[split]
            zeros = indices[coefficients[indices, i] == 0]
            chosen, tiers = matched_controls(rows, components[i]["model"], bags[split], zeros)
            out[i][split] = {"indices": chosen, "tiers": tiers}
    return out


def compare_pair(rows, coeff, embedding, samples, controls, i, j, split):
    a, b = samples[i][split], samples[j][split]
    neg_a = controls[i][split]["indices"]
    neg_b = controls[j][split]["indices"]
    if not neg_a or not neg_b:
        return None
    actual, plan, cost = comparison(a, b, embedding, coeff, i, j)
    left_mass, right_mass = np.abs(coeff[a, i]), np.abs(coeff[b, j])
    against_b = comparison(a, neg_b, embedding, coeff, i, j, mass_b=right_mass)[0]
    against_a = comparison(b, neg_a, embedding, coeff, j, i, mass_b=left_mass)[0]
    plan_entries = sorted(zip(*np.nonzero(plan > 1e-10), strict=True), key=lambda p: -plan[p])
    display = [(x, y, "high-mass") for x, y in plan_entries[:12]]
    display += [
        (x, y, "high-cost")
        for x, y in sorted(plan_entries, key=lambda p: -cost[p])[:3]
        if (x, y) not in {(a, b) for a, b, _ in display}
    ]
    return {
        "values": actual,
        "controls": {"left": against_a, "right": against_b},
        "advantage": min(against_a["ot"], against_b["ot"]) - actual["ot"],
        "centroidAdvantage": min(against_a["centroid"], against_b["centroid"]) - actual["centroid"],
        "transport": [
            {
                "left": rows[a[x]].id,
                "right": rows[b[y]].id,
                "mass": float(plan[x, y]),
                "cost": float(cost[x, y]),
                "kind": kind,
            }
            for x, y, kind in display
        ],
        "contextChecks": {
            "sameDocumentMass": float(
                sum(
                    plan[x, y]
                    for x, y in plan_entries
                    if rows[a[x]].document == rows[b[y]].document
                )
            ),
            "sameTerminalMass": float(
                sum(
                    plan[x, y]
                    for x, y in plan_entries
                    if lexical_key(rows[a[x]].prefix) == lexical_key(rows[b[y]].prefix)
                )
            ),
            "weightedPrefixCharacterGap": float(
                sum(
                    plan[x, y] * abs(len(rows[a[x]].prefix) - len(rows[b[y]].prefix))
                    for x, y in plan_entries
                )
            ),
        },
    }


def discover_pairs(rows, coeff, representations, samples, controls, components):
    embedding = representations["combined"]
    eligible = sorted(i for i in samples if controls[i]["discovery"]["indices"])
    if len(eligible) < 2:
        return []
    centers = []
    for i in eligible:
        ids = samples[i]["discovery"]
        centers.append(centroid(embedding[ids], np.abs(coeff[ids, i])))
    similarity = np.asarray(centers) @ np.asarray(centers).T
    pairs = set()
    for position, i in enumerate(eligible):
        for target_model in {item["model"] for item in components}:
            choices = [
                p
                for p, j in enumerate(eligible)
                if j != i and components[j]["model"] == target_model
            ]
            nearest = sorted(choices, key=lambda p: (-similarity[position, p], eligible[p]))[
                :NEIGHBORS
            ]
            pairs.update(tuple(sorted((i, eligible[p]))) for p in nearest)
    ranked = []
    for i, j in sorted(pairs):
        result = compare_pair(rows, coeff, embedding, samples, controls, i, j, "discovery")
        if result and result["advantage"] > 0 and result["values"]["distinctContextMass"] > 0.5:
            ranked.append((i, j, result["advantage"]))
    ranked.sort(key=lambda p: (-p[2], p[:2]))
    by_scope = defaultdict(int)
    preferred, overflow = [], []
    for i, j, score in ranked:
        scope = tuple(sorted((components[i]["model"], components[j]["model"])))
        bucket = preferred if by_scope[scope] < 8 else overflow
        bucket.append((i, j, score))
        by_scope[scope] += 1
    shortlisted = (preferred + overflow)[:MAX_CONFIRMATORY]
    return sorted(shortlisted, key=lambda p: (-p[2], p[:2]))


def independent_example(row_index, column, rows, coeff, documents, texts, component, *, contrast):
    row = rows[row_index]
    scale = component["uNorm"] / component["sigma"]
    sequence = []
    previous = 0
    for i in sorted(documents[row.document], key=lambda n: rows[n].boundary):
        token = rows[i]
        if token.boundary > row.boundary:
            break
        coefficient = float(coeff[i, column])
        position, token_id, native_start, native_end = token.native[component["model"]]
        if native_start > previous:
            sequence.append(
                {
                    "text": texts[row.document][previous:native_start],
                    "anchor": None,
                    "coefficient": None,
                    "magnitude": None,
                    "tokenPosition": None,
                    "tokenId": None,
                }
            )
        sequence.append(
            {
                "text": texts[row.document][max(previous, native_start) : native_end],
                "anchor": token.id,
                "coefficient": coefficient,
                "magnitude": abs(coefficient) * scale,
                "tokenPosition": position,
                "tokenId": token_id,
            }
        )
        previous = token.boundary
    return {
        "anchor": row.id,
        "document": row.document,
        "source": row.source,
        "pool": row.pool,
        "boundary": row.boundary,
        "native": row.native,
        "contrast": contrast,
        "coefficient": float(coeff[row_index, column]),
        "magnitude": abs(float(coeff[row_index, column])) * scale,
        "sequence": sequence,
        "graph": "unavailable",
    }


def groups(pairs):
    """Each displayed candidate is a validated pair; direct links cannot imply transitive matches."""
    by_component = defaultdict(set)
    for pair in pairs:
        a, b = pair["left"], pair["right"]
        by_component[a].add(b)
        by_component[b].add(a)
    # Complete-link triangles are reportable as multi-component groups; keep isolated validated links too.
    triangles = {
        tuple(sorted((a, b, c)))
        for a in by_component
        for b in by_component[a]
        for c in by_component[a] & by_component[b]
    }
    covered = {tuple(sorted((a, b))) for tri in triangles for a in tri for b in tri if a != b}
    members = sorted(
        triangles
        | {
            tuple(sorted((p["left"], p["right"])))
            for p in pairs
            if tuple(sorted((p["left"], p["right"]))) not in covered
        }
    )
    return [{"components": list(cs)} for cs in members]


def analyze(args):
    started = time.monotonic()
    contract, hash_value, rows, coeff, documents, texts, runtime = read_measurements(
        args.measurements
    )
    args.output.mkdir(parents=True, exist_ok=False)
    components = contract["components"]
    supports, samples, zero_sample, measured = support_sets(rows, coeff)
    counts = {
        component["id"]: {split: len(supports[i][split]) for split in SPLITS}
        | {"measuredNonzeroBeforeDedup": measured[i]}
        for i, component in enumerate(components)
    }
    write_json(
        args.output / "coverage.json",
        {
            "contractSha256": hash_value,
            "poolCount": len(runtime),
            "anchorCount": len(rows),
            "documentCount": len(documents),
            "bySource": {
                source: len({r.document for r in rows if r.source == source})
                for source in sorted({r.source for r in rows})
            },
            "components": counts,
        },
    )
    if args.coverage_only:
        return
    if not samples:
        raise ValueError("No components are eligible after document and context deduplication")
    controls = controls_for(rows, coeff, components, samples, zero_sample)
    indices = sorted(
        {
            row
            for entries in (
                samples,
                {i: {k: v["indices"] for k, v in record.items()} for i, record in controls.items()},
            )
            for split in entries.values()
            for ids in split.values()
            for row in ids
        }
    )
    vectors = embed_variants([rows[i].prefix for i in indices], args.snapshot, args.threads)
    positions = {i: j for j, i in enumerate(indices)}
    compact_rows, compact_coeff = [rows[i] for i in indices], coeff[indices]
    cs = {
        i: {k: [positions[n] for n in ns] for k, ns in entry.items()}
        for i, entry in samples.items()
    }
    cc = {
        i: {
            k: {"indices": [positions[n] for n in v["indices"]], "tiers": v["tiers"]}
            for k, v in entry.items()
        }
        for i, entry in controls.items()
    }
    shortlist = discover_pairs(compact_rows, compact_coeff, vectors, cs, cc, components)
    write_json(
        args.output / "frozen-discovery.json",
        {
            "pairs": [
                [components[i]["id"], components[j]["id"], score] for i, j, score in shortlist
            ],
            "rankingVariant": "combined",
            "prefilterNeighborsPerModel": NEIGHBORS,
            "preferredPairsPerModelScope": 8,
            "splitSeed": SPLIT_SEED,
            "bootstrapDraws": BOOTSTRAP_DRAWS,
        },
    )
    test = []
    rng = np.random.default_rng(347212)
    for i, j, _ in shortlist:
        results = {
            split: {
                name: compare_pair(compact_rows, compact_coeff, vectors[name], cs, cc, i, j, split)
                for name in (*VARIANTS, "withoutTerminal")
            }
            for split in SPLITS
        }
        a, b = cs[i]["heldout"], cs[j]["heldout"]
        ac, bc = cc[i]["heldout"]["indices"], cc[j]["heldout"]["indices"]
        heldout_result = results["heldout"]["combined"]
        interval = None
        if heldout_result:
            interval = bootstrap_advantage(
                compact_rows,
                a,
                b,
                ac,
                bc,
                vectors["combined"],
                np.abs(compact_coeff[a, i]),
                np.abs(compact_coeff[b, j]),
                rng,
                BOOTSTRAP_DRAWS,
            )
        valid = bool(
            interval
            and interval[0] > 0
            and heldout_result["values"]["distinctContextMass"] > 0.5
            and heldout_result["contextChecks"]["sameDocumentMass"] < 0.5
            and results["heldout"]["withoutTerminal"]["advantage"] > 0
        )
        test.append(
            {
                "left": components[i]["id"],
                "right": components[j]["id"],
                "results": results,
                "fullSupportJaccard": {
                    split: len(set(supports[i][split]) & set(supports[j][split]))
                    / len(set(supports[i][split]) | set(supports[j][split]))
                    for split in SPLITS
                },
                "heldoutAdvantageInterval": interval,
                "retained": valid,
                "controls": {
                    side: {split: controls[c][split]["tiers"] for split in SPLITS}
                    for side, c in (("left", i), ("right", j))
                },
            }
        )
    write_json(args.output / "evaluated-pairs.json", test)
    passed = [p for p in test if p["retained"]]
    analysis_metrics = {
        name: {
            "pairsWithControls": len(
                scored := [
                    pair["results"]["heldout"][name]
                    for pair in test
                    if pair["results"]["heldout"][name] is not None
                ]
            ),
            "lowerOtThanBothControls": sum(row["advantage"] > 0 for row in scored),
            "lowerCentroidThanBothControls": sum(row["centroidAdvantage"] > 0 for row in scored),
            "higherAnchorOverlapThanBothControls": sum(
                row["values"]["anchorOverlap"]
                > max(ref["anchorOverlap"] for ref in row["controls"].values())
                for row in scored
            ),
        }
        for name in (*VARIANTS, "withoutTerminal")
    }
    write_json(
        args.output / "evaluation-summary.json",
        {
            "candidateCount": len(test),
            "pairCountPassing": len(passed),
            "heldout": analysis_metrics,
            "interpretation": "Exploratory document intervals; no multiplicity-adjusted confirmation",
        },
    )
    retained = {key for pair in passed for key in (pair["left"], pair["right"])}
    active = {
        comp["id"]: comp | {"coverage": counts[comp["id"]]}
        for comp in components
        if comp["id"] in retained
    }
    summary = coverage(active)
    assets = args.output / "frontend"
    assets.mkdir()
    references = {}
    for i, key in enumerate(components):
        identity = key["id"]
        if identity not in active:
            continue
        entries = {
            split: [
                independent_example(n, i, rows, coeff, documents, texts, key, contrast=contrast)
                for contrast, ids in (
                    (False, supports[i][split]),
                    (True, controls[i][split]["indices"]),
                )
                for n in ids
            ]
            for split in SPLITS
        }
        filename = f"component-{i}.json.gz"
        payload = {
            "runId": hash_value,
            "component": identity,
            "examples": entries,
            "maxMagnitude": max(
                token["magnitude"]
                for split in SPLITS
                for ex in entries[split]
                for token in ex["sequence"]
                if token["magnitude"] is not None
            ),
        }
        with gzip.open(assets / filename, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        references[identity] = {"file": filename, "sha256": sha256_file(assets / filename)}
    catalog = {
        "schemaVersion": 2,
        "runId": hash_value,
        "dataset": "natural-validation",
        "checkpoint": contract["checkpoint"],
        "summary": summary,
        "components": active,
        "pairs": passed,
        "groups": groups(passed),
        "examples": references,
        "graphCoverage": "unavailable",
        "splitSeed": SPLIT_SEED,
        "provenance": {
            "collector": contract,
            "encoder": ENCODER,
            "encoderRevision": REVISION,
            "poolCount": len(runtime),
            "runtimeSeconds": time.monotonic() - started,
            "analysis": runtime_provenance(),
            "analysisSha256": sha256_file(Path(__file__)),
        },
    }
    write_json(assets / "catalog.json", catalog)
    write_json(
        assets / "COMPLETE.json",
        {
            "status": "COVERAGE_VALIDATED",
            "schemaVersion": 2,
            "runId": hash_value,
            "catalogSha256": sha256_file(assets / "catalog.json"),
            "summary": summary,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    if not args.coverage_only and args.snapshot is None:
        parser.error("--snapshot is required for scoring")
    analyze(args)


if __name__ == "__main__":
    main()

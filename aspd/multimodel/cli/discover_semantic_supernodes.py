"""Run a bounded Semantic Supernode Discovery experiment on sealed measurements."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import shutil
import subprocess
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.distance import cdist

from aspd.multimodel.cli.collect_aspd_semantic_contributions import verified_json
from aspd.multimodel.discovery import semantic_supernodes as sn
from aspd.multimodel.discovery.semantic_ot import (
    ENCODER,
    ENCODER_FILES,
    REVISION,
    embed,
    is_heldout,
)
from aspd.multimodel.discovery.semantic_ot_v2 import SPLIT_SEED, lexical_key, shingles
from aspd.multimodel.discovery.supernode_reference import load_native_reference
from aspd.multimodel.posthoc.artifacts import sha256_file
from aspd.multimodel.posthoc.artifacts import write_json as write_exclusive_json

MODELS = ("gemma", "smol", "granite")
SPLITS = ("discovery", "heldout")
CHECKPOINT = "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969"


def write_json(path, value):
    with TemporaryDirectory(dir=path.parent) as stage:
        file = Path(stage) / path.name
        write_exclusive_json(file, value)
        file.replace(path)


def load(path):
    return json.loads(path.read_text())


def choose_cohort(components, coverage, count, seed, repeat_gates=0):
    """Round robin across hashed gate IDs, beta order within each gate."""
    cohort, excluded = [], {}
    for model in MODELS:
        model_components = [(i, c) for i, c in enumerate(components) if c["model"] == model]
        by_gate = defaultdict(list)
        for i, component in model_components:
            if coverage.get(component["id"], {}).get("discovery", 0) >= 20:
                by_gate[component["gate"]].append((i, component))
        for group in by_gate.values():
            group.sort(key=lambda x: (-x[1]["beta"], x[1]["id"]))
        gate_order = sorted(
            by_gate, key=lambda g: hashlib.sha256(f"{seed}/{model}/{g}".encode()).digest()
        )
        selected = [
            by_gate[gate][j]
            for j in range(max(map(len, by_gate.values()), default=0))
            for gate in gate_order
            if j < len(by_gate[gate])
        ][:count]
        if not selected:
            raise ValueError(f"Missing discovery cohort for {model}")
        if repeat_gates:
            # Keep a deterministic subset of gates with two representatives so the same-gate
            # baseline has support while the rest of the cohort stays gate-balanced.
            repeated = [by_gate[gate][:2] for gate in gate_order if len(by_gate[gate]) >= 2][
                :repeat_gates
            ]
            reserved = [item for pair in repeated for item in pair]
            reserved_ids = {i for i, _ in reserved}
            selected = (reserved + [item for item in selected if item[0] not in reserved_ids])[
                :count
            ]
        cohort.extend(selected)
        eligible = sum(map(len, by_gate.values()))
        excluded[model] = {
            "available": len(model_components),
            "selected": len(selected),
            "insufficientDiscovery": len(model_components) - eligible,
            "eligibleOutsideBudget": eligible - len(selected),
        }
    return cohort, excluded


def pool_blocks(root, contract, contract_hash):
    directories = sorted(root.glob("pool-[0-9]*"))
    if len(directories) not in contract["waves"]:
        raise ValueError("Expected a fully measured predefined wave")
    for pool, directory in enumerate(directories, start=contract["firstPool"]):
        info = verified_json(directory, "meta.json")
        if info["contractSha256"] != contract_hash or info["pool"] != pool:
            raise ValueError(f"Pool provenance mismatch at {directory}")
        with gzip.open(directory / "anchors.json.gz", "rt") as handle:
            metadata = json.load(handle)
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            block = arrays["coefficients"]
            gates = arrays["gates"]
        count = contract["backend"]["pool_anchors"]
        if (
            block.shape != (count, len(contract["components"]))
            or gates.shape != (count, len(contract["gateColumns"]))
            or len(metadata["anchors"]) != count
            or info["anchors"] != count
        ):
            raise ValueError(f"Incomplete pool at {directory}")
        yield pool, metadata, block, gates


def harvest(root, contract, contract_hash, cohort, goal):
    columns = [i for i, _ in cohort]
    gate_columns = [contract["gateColumns"].index(c["gate"]) for _, c in cohort]
    winners = [defaultdict(dict) for _ in cohort]
    documents, witnesses = {}, {}
    initial_document = None
    for pool, metadata, block, gates in pool_blocks(root, contract, contract_hash):
        docs, anchors = metadata["documents"], metadata["anchors"]
        if initial_document is None:
            initial_document = docs[0]["id"]
        document_ids = [doc["id"] for doc in docs]
        for doc in docs:
            if doc["id"] == initial_document:
                continue
            if documents.setdefault(doc["id"], doc) != doc:
                raise ValueError("A document identity was reused for different text")
        selected = block[:, columns]
        selected_gates = gates[:, gate_columns]
        if (
            not np.isfinite(selected).all()
            or not np.isfinite(selected_gates).all()
            or np.any(selected_gates < 0)
            or np.any(selected[selected_gates == 0] != 0)
        ):
            raise ValueError(f"Native support mismatch in pool {pool}")
        nonzero_rows, nonzero_columns = np.nonzero(selected)
        for i, j in zip(nonzero_rows, nonzero_columns, strict=True):
            aid, doc_index, boundary, native = anchors[i]
            document = document_ids[doc_index]
            if document == initial_document:
                continue
            coefficient = float(selected[i, j])
            candidate = (-abs(coefficient), aid)
            previous = winners[j][document].get("strong")
            if previous is None or candidate < previous[0]:
                winners[j][document]["strong"] = (candidate, coefficient, pool, boundary, native)
        # A deterministic within-document measured-zero example is eligible as contrast.
        zero_row = selected == 0
        for doc_index in range(len(docs)):
            doc = document_ids[doc_index]
            if doc == initial_document:
                continue
            rows = np.asarray([i for i, item in enumerate(anchors) if item[1] == doc_index])
            if not len(rows):
                continue
            options = zero_row[rows]
            for j in np.flatnonzero(options.any(0)):
                i = rows[np.flatnonzero(options[:, j])[-1]]
                aid, _, boundary, native = anchors[i]
                previous = winners[j][doc].get("contrast")
                if previous is None or aid < previous[0][1]:
                    winners[j][doc]["contrast"] = ((0, aid), 0.0, pool, boundary, native)
    evidence, counts = {}, {}
    for j, (_, component) in enumerate(cohort):
        key = component["id"]
        evidence[key] = {}
        counts[key] = {}
        for split in SPLITS:
            entries = [
                (doc, data["strong"])
                for doc, data in winners[j].items()
                if "strong" in data and is_heldout(doc, SPLIT_SEED) == (split == "heldout")
            ]
            entries.sort(key=lambda item: item[1][0])
            kept, signatures = [], []
            for doc, record in entries:
                _, _, _, boundary, _ = record
                signature = shingles(documents[doc]["text"][:boundary])
                if not signature or any(
                    len(signature & prior) >= 0.85 * len(signature | prior) for prior in signatures
                ):
                    continue
                signatures.append(signature)
                kept.append((doc, record))
                if len(kept) == goal:
                    break
            counts[key][split] = {
                "distinctNonzeroDocuments": len(entries),
                "exportedNonzero": len(kept),
            }
            for doc, record in kept:
                add_witness(witnesses, doc, documents[doc], record)
            evidence[key][split] = [example(doc, value, component) for doc, value in kept]
            contrasts = [
                (d, winners[j][d]["contrast"]) for d, _ in kept if "contrast" in winners[j][d]
            ]
            evidence[key][split + "Contrast"] = []
            for doc, record in contrasts[:2]:
                add_witness(witnesses, doc, documents[doc], record)
                evidence[key][split + "Contrast"].append(example(doc, record, component))
    return evidence, counts, witnesses, len(documents)


def add_witness(witnesses, doc, source, record):
    (_, aid), _, pool, boundary, native = record
    witness = {
        "anchor": aid,
        "document": doc,
        "source": source["source"],
        "pool": pool,
        "boundary": boundary,
        "native": native,
        "prefix": source["text"][:boundary],
    }
    if witnesses.setdefault(aid, witness) != witness:
        raise ValueError("Conflicting anchor evidence")


def example(doc, record, component):
    (_, aid), coefficient, _, _, _ = record
    return {
        "anchor": aid,
        "document": doc,
        "coefficient": coefficient,
        "strength": abs(coefficient) * component["uNorm"] / component["sigma"],
    }


def heatmaps(root, contract, contract_hash, cohort, examples, witnesses):
    """Export actual anchor-level scalar series for the top and contrast documents."""
    requests = defaultdict(list)
    for col, component in cohort:
        key = component["id"]
        for split, items in examples[key].items():
            for item in items[: (2 if split.endswith("Contrast") else 3)]:
                requests[item["document"]].append((col, component, item))
    series = defaultdict(list)
    for _, metadata, block, _ in pool_blocks(root, contract, contract_hash):
        docs = metadata["documents"]
        for i, (aid, j, boundary, native) in enumerate(metadata["anchors"]):
            doc = docs[j]["id"]
            if doc in requests:
                for col, component, item in requests[doc]:
                    if boundary <= witnesses[item["anchor"]]["boundary"]:
                        pos, token, start, end = native[component["model"]]
                        value = float(block[i, col])
                        series[component["id"], item["anchor"]].append(
                            [
                                aid,
                                start,
                                end,
                                pos,
                                token,
                                value,
                                abs(value) * component["uNorm"] / component["sigma"],
                            ]
                        )
    for values in requests.values():
        for _, component, item in values:
            item["heatmap"] = sorted(series[component["id"], item["anchor"]], key=lambda x: x[2])


def partition_record(model, ids, groups, distance):
    result = []
    for group in groups:
        members = [ids[i] for i in group]
        name = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:12]
        means = distance[np.ix_(group, group)].sum(1) / max(1, len(group) - 1)
        result.append(
            {
                "id": f"{model}-{name}",
                "members": members,
                "medoid": members[int(means.argmin())],
                "outlier": members[int(means.argmax())],
                "withinDistance": sn.cost_summary([group], distance)["mean"],
            }
        )
    return result


def same_lexical_unit(witnesses, left, right):
    return lexical_key(witnesses[left]["prefix"]) == lexical_key(witnesses[right]["prefix"])


def link_models(models, indexes, vectors, neighbors, witnesses, maximum):
    matches = []
    context_keys = sorted(witnesses)
    for a, b in combinations(MODELS, 2):
        left, right = models[a]["clusters"], models[b]["clusters"]
        merged = {
            cluster["id"]: [
                sn.mixture([indexes[split][key] for key in cluster["members"]])
                if all(key in indexes[split] for key in cluster["members"])
                else None
                for split in SPLITS
            ]
            for cluster in left + right
        }

        def centroid(cluster, distributions=merged):
            distribution = distributions[cluster["id"]][0]
            return np.average(vectors[distribution.anchors], axis=0, weights=distribution.mass)

        shortcut = cdist(
            np.stack([centroid(x) for x in left]), np.stack([centroid(x) for x in right])
        )
        pairs = {
            (i, int(j))
            for i, row in enumerate(shortcut)
            for j in np.argsort(row, kind="stable")[:neighbors]
        }
        pairs |= {
            (int(i), j)
            for j, col in enumerate(shortcut.T)
            for i in np.argsort(col, kind="stable")[:neighbors]
        }
        for i, j in sorted(pairs):
            row = {"left": left[i]["id"], "right": right[j]["id"]}
            for split_index, split in enumerate(SPLITS):
                left_dist = merged[row["left"]][split_index]
                right_dist = merged[row["right"]][split_index]
                value, plan = (
                    (None, None)
                    if left_dist is None or right_dist is None
                    else sn.compare(left_dist, right_dist, vectors)
                )
                row[split] = value
                if plan is not None and split == "heldout":
                    left_keys = [context_keys[i] for i in left_dist.anchors]
                    right_keys = [context_keys[i] for i in right_dist.anchors]
                    left_docs = np.array([witnesses[key]["document"] for key in left_keys])
                    right_docs = np.array([witnesses[key]["document"] for key in right_keys])
                    row["heldoutSameDocumentMass"] = float(
                        plan[left_docs[:, None] == right_docs].sum()
                    )
                    row["heldoutExactAnchorMass"] = float(
                        plan[np.array(left_keys)[:, None] == right_keys].sum()
                    )
                    row["heldoutSameTerminalMass"] = float(
                        sum(
                            plan[u, v]
                            for u, v in zip(*np.nonzero(plan), strict=True)
                            if same_lexical_unit(witnesses, left_keys[u], right_keys[v])
                        )
                    )
                    # This control retains full distributions but disallows same-document
                    # source/destination pairs. Infeasible couplings are reported explicitly.
                    excluded = left_docs[:, None] == right_docs
                    row["heldoutNoSameDocument"] = (
                        value
                        if not excluded.any()
                        else sn.cost_without_pairs(
                            cdist(vectors[left_dist.anchors], vectors[right_dist.anchors]),
                            left_dist.mass,
                            right_dist.mass,
                            excluded,
                        )
                    )
                    ranked = np.argsort(-plan.ravel(), kind="stable")[:8]
                    row["transported"] = [
                        {
                            "left": context_keys[left_dist.anchors[p // plan.shape[1]]],
                            "right": context_keys[right_dist.anchors[p % plan.shape[1]]],
                            "mass": float(plan.ravel()[p]),
                        }
                        for p in ranked
                        if plan.ravel()[p] > 0
                    ]
            row["accepted"] = row["discovery"] <= maximum
            matches.append(row)
    return sorted(matches, key=lambda row: (row["discovery"], row["left"], row["right"]))


def inspect_taylor(path, contract_hash, models, components):
    if path is None:
        return {"status": "unavailable", "witnesses": []}
    seal = load(path / "COMPLETE.json")
    if sha256_file(path / "graphs.json") != seal["sha256"]:
        raise ValueError("Graph archive seal mismatch")
    archive = load(path / "graphs.json")
    if archive["checkpoint"] != CHECKPOINT or archive["runId"] != contract_hash:
        raise ValueError("Graph archive does not match measured inputs")
    membership = {
        key: cluster
        for model in models.values()
        for cluster in model["clusters"]
        for key in cluster["members"]
    }
    scales = {c["id"]: c["uNorm"] / c["sigma"] for c in components}
    grouped = defaultdict(dict)
    matched = set()
    for graph in archive["graphs"].values():
        a = graph["component"]
        if a not in membership:
            continue
        anchor = archive["anchors"][graph["anchor"]]
        if anchor["pool"] != graph["pool"] or graph["model"] != a.split(".")[0]:
            raise ValueError("Taylor edge provenance mismatch")
        matched.add((graph["model"], graph["anchor"], a))
        for edge in graph["edges"]:
            b = edge["source"]
            if b in membership:
                grouped[graph["model"], graph["anchor"]][b, a] = edge["E"]
    rows = []
    for (model, anchor), measured in grouped.items():
        links = {}
        for b, a in measured:
            left, right = membership[b], membership[a]
            if left["id"] != right["id"]:
                links[left["id"], right["id"]] = (left, right)
        for (_, _), (left, right) in links.items():
            result = sn.aggregate_taylor(left["members"], right["members"], measured, scales)
            rows.append(
                {
                    "from": left["id"],
                    "to": right["id"],
                    "model": model,
                    "anchor": anchor,
                    "scope": archive["anchors"][anchor],
                    "measurement": "cached-top-k",
                    **result,
                }
            )
    return {
        "status": "archived-top-k",
        "archiveSha256": seal["sha256"],
        "matchedTargetMeasurements": len(matched),
        "witnesses": rows,
    }


def analyze(args):
    start = time.monotonic()
    contract = load(args.measurements / "contract.json")
    contract_hash = sha256_file(args.measurements / "contract.json")
    coverage = load(args.coverage)
    if contract["checkpoint"] != CHECKPOINT or coverage["contractSha256"] != contract_hash:
        raise ValueError("Checkpoint or coverage contract mismatch")
    if contract["backend"]["pool_anchors"] != 2048:
        raise ValueError("Unexpected natural BatchTopK pool size")
    if args.output.exists() and (args.output / "COMPLETE.json").exists():
        raise ValueError("Completed experiments are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    cohort, excluded = choose_cohort(
        contract["components"], coverage["components"], args.per_model, args.seed, args.repeat_gates
    )
    if sha256_file(Path(contract["backend"]["checkpoint"]) / "manifest.json") != CHECKPOINT:
        raise ValueError("Frozen checkpoint manifest differs from the measured wave")
    for name, digest in contract["references"].items():
        if sha256_file(Path(contract["reference"]) / name / "COMPLETE") != digest:
            raise ValueError(f"The Natural Validation reference changed: {name}")
    components = [c for _, c in cohort]
    signature = {
        "algorithm": 1,
        "contract": contract_hash,
        "cohort": [c["id"] for c in components],
        "coverage": sha256_file(args.coverage),
        "contexts": args.examples,
        "archive": sha256_file(args.graphs / "COMPLETE.json") if args.graphs else None,
        "reference": args.reference,
        "repeatGates": args.repeat_gates,
        "hiddenVariant": args.hidden_variant,
        "hiddenManifest": sha256_file(args.hidden_reference / "COMPLETE.json")
        if args.hidden_reference
        else None,
        "support": args.support,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "count": args.cluster_count,
        "cut": args.threshold,
        "discoveryNeighborQuantile": args.discovery_neighbor_quantile,
        "neighbors": args.neighbors,
        "crossMaximum": args.cross_maximum,
        "sourceSha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            Path(sn.__file__).name: sha256_file(Path(sn.__file__)),
        },
    }
    state = args.output / "contract.json"
    if state.exists() and load(state) != signature:
        raise ValueError("Use a new output directory for a different experiment")
    if not state.exists():
        write_json(state, signature)
    stages = {}
    if args.reuse is not None:
        old = load(args.reuse / "contract.json")
        seal = load(args.reuse / "COMPLETE.json")
        for key in (
            "contract",
            "cohort",
            "coverage",
            "contexts",
            "reference",
            "support",
            "epsilon",
        ):
            if old.get(key) != signature[key]:
                raise ValueError(f"Cached experiment has different {key}")
        if (
            old.get("hiddenManifest") != signature["hiddenManifest"]
            or old.get("hiddenVariant") != signature["hiddenVariant"]
        ):
            raise ValueError("Native within-model reference changed")
        for name in (
            "evidence.json",
            "reference.npy",
            *(f"{m}-{s}.npy" for m in MODELS for s in SPLITS),
        ):
            path = args.reuse / name
            if sha256_file(path) != seal["sha256"][name]:
                raise ValueError(f"Cached artifact failed verification: {name}")
            target = args.output / name
            if not target.exists():
                with TemporaryDirectory(dir=args.output) as stage:
                    pending = Path(stage) / target.name
                    shutil.copyfile(path, pending)
                    pending.replace(target)

    extraction = args.output / "evidence.json"
    if extraction.exists():
        rows = load(extraction)
        examples, counts, witnesses, documents = (
            rows[k] for k in ("examples", "counts", "witnesses", "documents")
        )
    else:
        examples, counts, witnesses, documents = harvest(
            args.measurements,
            contract,
            contract_hash,
            cohort,
            args.examples,
        )
        heatmaps(args.measurements, contract, contract_hash, cohort, examples, witnesses)
        write_json(
            extraction,
            {
                "examples": examples,
                "counts": counts,
                "witnesses": witnesses,
                "documents": documents,
            },
        )
    stages["extraction"] = time.monotonic() - start
    print("Measured supports and heatmaps cached", stages["extraction"], flush=True)
    ids = sorted(witnesses)
    row_by_id = {aid: i for i, aid in enumerate(ids)}
    indexes = {split: {} for split in SPLITS}
    for key, splits in examples.items():
        for split in SPLITS:
            items = splits[split][: args.support]
            if len(items) >= args.support:
                indexes[split][key] = sn.Distribution(
                    np.asarray([row_by_id[item["anchor"]] for item in items]),
                    np.asarray([item["strength"] for item in items]),
                )
    vectors_path = args.output / "reference.npy"
    if not vectors_path.exists():
        for name, expected in ENCODER_FILES.items():
            if sha256_file(args.encoder / name) != expected:
                raise ValueError(f"Unverified encoder file {name}")
        vectors = embed(
            [witnesses[aid]["prefix"] for aid in ids],
            args.encoder,
            threads=args.threads,
            max_tokens=40 if args.reference == "local" else 128,
        )
        write_array(vectors_path, vectors)
    vectors = np.load(vectors_path, mmap_mode="r")
    stages["reference"] = time.monotonic() - start - stages["extraction"]
    print("Reference space", vectors.shape, "seconds", stages["reference"], flush=True)
    rng = np.random.default_rng(args.seed)
    models = {}
    for model in MODELS:
        keys = [
            c["id"] for c in components if c["model"] == model and c["id"] in indexes["discovery"]
        ]
        if not keys:
            raise ValueError(f"No adequate semantic distributions for {model}")
        adequate = [
            key
            for key in keys
            if counts[key]["heldout"]["exportedNonzero"] >= max(10, args.support)
        ]
        excluded[model]["insufficientMeasuredHeldout"] = sorted(set(keys) - set(adequate))
        keys = adequate
        if not keys:
            raise ValueError(f"{model}: no component meets the held-out coverage requirement")
        within_vectors = vectors
        within_indexes = indexes
        native_reference = None
        if args.hidden_reference is not None:
            needed = sorted(
                {
                    int(anchor)
                    for split in SPLITS
                    for key in keys
                    for anchor in indexes[split][key].anchors
                }
            )
            position = {anchor: i for i, anchor in enumerate(needed)}
            within_vectors, native_reference = load_native_reference(
                args.hidden_reference,
                model,
                [ids[i] for i in needed],
                contract_hash,
                args.hidden_variant,
            )
            within_indexes = {
                split: {
                    key: sn.Distribution(
                        np.array([position[i] for i in indexes[split][key].anchors]),
                        indexes[split][key].mass,
                    )
                    for key in keys
                }
                for split in SPLITS
            }
        matrices = {}
        for split in SPLITS:
            path = args.output / f"{model}-{split}.npy"
            if not path.exists():
                write_array(
                    path,
                    sn.pairwise(
                        [within_indexes[split][key] for key in keys],
                        within_vectors,
                        epsilon=args.epsilon,
                    ),
                )
            matrices[split] = np.load(path)
        d, held = matrices["discovery"], matrices["heldout"]
        threshold = args.threshold
        if args.discovery_neighbor_quantile is not None:
            nearest = np.where(np.eye(len(d), dtype=bool), np.inf, d).min(1)
            threshold = float(np.quantile(nearest, args.discovery_neighbor_quantile))
        tree, groups = sn.tree_and_partition(d, count=args.cluster_count, threshold=threshold)
        _, held_groups = sn.tree_and_partition(held, count=len(groups))
        centers = np.stack(
            [
                np.average(
                    within_vectors[within_indexes["discovery"][key].anchors],
                    axis=0,
                    weights=indexes["discovery"][key].mass,
                )
                for key in keys
            ]
        )
        _, mean_groups = sn.tree_and_partition(cdist(centers, centers), count=len(groups))
        by_gate = defaultdict(list)
        gate_map = {c["id"]: c["gate"] for c in components}
        for i, key in enumerate(keys):
            by_gate[gate_map[key]].append(i)
        null = []
        lengths = np.cumsum([len(group) for group in groups])[:-1]
        for _ in range(200):
            value = sn.cost_summary(np.split(rng.permutation(len(d)), lengths), held)["mean"]
            if value is not None:
                null.append(value)
        models[model] = {
            "nativeReference": native_reference,
            "componentOrder": keys,
            "clusters": partition_record(model, keys, groups, d),
            "tree": tree.tolist(),
            "distanceCut": threshold,
            "distances": d.tolist(),
            "heldoutDistances": held.tolist(),
            "evaluation": {
                "discovery": sn.cost_summary(groups, d),
                "heldout": sn.cost_summary(groups, held),
                "heldoutCoassignmentJaccard": sn.coassignment(groups, held_groups),
                "meanEmbedding": sn.cost_summary(mean_groups, held),
                "sharedGate": sn.cost_summary(list(by_gate.values()), held),
                "random": (
                    {
                        "mean": float(np.mean(null)),
                        "quantiles": np.quantile(null, [0.025, 0.975]).tolist(),
                        "draws": len(null),
                    }
                    if null
                    else None
                ),
            },
        }
        print("Partition measured", model, len(keys), len(groups), flush=True)
    stages["partition"] = time.monotonic() - start - sum(stages.values())
    retained = [key for model in models.values() for key in model["componentOrder"]]
    if np.mean([counts[key]["heldout"]["exportedNonzero"] for key in retained]) < 20:
        raise ValueError("Retained cohort does not meet the mean held-out coverage requirement")
    matches = link_models(models, indexes, vectors, args.neighbors, witnesses, args.cross_maximum)
    diagnostics = (
        transport_diagnostics(models, indexes, vectors, witnesses, ids, args.support)
        if args.hidden_reference is None
        else []
    )
    taylor = inspect_taylor(args.graphs, contract_hash, models, components)
    stages["crossModelAndTaylor"] = time.monotonic() - start - sum(stages.values())
    output = {
        "schemaVersion": 1,
        "dataset": "natural-validation",
        "checkpoint": CHECKPOINT,
        "runId": sha256_file(state),
        "measurementContract": contract_hash,
        "measurementProvenance": {
            k: contract[k] for k in ("backend", "corpus", "references", "git_sha")
        },
        "settings": signature,
        "cohort": excluded,
        "documents": documents,
        "models": models,
        "matches": matches,
        "components": [c for c in components if c["id"] in models[c["model"]]["componentOrder"]],
        "counts": counts,
        "diagnostics": diagnostics,
        "taylor": taylor,
        "exactAblation": {"status": "unmeasured"},
        "reference": {
            "id": ENCODER,
            "revision": REVISION,
            "dimensions": vectors.shape[1],
            "scope": args.reference,
            "normalization": "L2",
            "groundCost": "Euclidean",
            "clustering": "average linkage",
        },
        "execution": {
            "secondsByStage": stages,
            "totalSeconds": time.monotonic() - start,
            "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpuCount": 0,
            "gitHead": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    }
    write_json(args.output / "catalog.json", output)
    files = {p.name: sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    write_json(args.output / "COMPLETE.json", {"runId": output["runId"], "sha256": files})
    print(json.dumps(output["execution"]), flush=True)


def write_array(path, value):
    with TemporaryDirectory(dir=path.parent) as stage:
        pending = Path(stage) / path.name
        np.save(pending, value, allow_pickle=False)
        pending.replace(path)


def transport_diagnostics(models, indexes, vectors, witnesses, ids, support):
    selected = []
    for model in MODELS:
        keys = models[model]["componentOrder"]
        ranked = sorted(
            combinations(range(len(keys)), 2),
            key=lambda pair: models[model]["distances"][pair[0]][pair[1]],
        )
        take = np.linspace(0, len(ranked) - 1, min(12, len(ranked)), dtype=int)
        for index in take:
            i, j = ranked[index]
            left_dist, right_dist = (indexes["heldout"][keys[k]] for k in (i, j))
            distance, plan = sn.compare(left_dist, right_dist, vectors)
            reverse, _ = sn.compare(right_dist, left_dist, vectors)
            smooth, _ = sn.compare(left_dist, right_dist, vectors, epsilon=0.1)
            cost = cdist(vectors[left_dist.anchors], vectors[right_dist.anchors])
            count = max(1, support // 2)
            small, _ = sn.compare(
                sn.Distribution(left_dist.anchors[:count], left_dist.mass[:count]),
                sn.Distribution(right_dist.anchors[:count], right_dist.mass[:count]),
                vectors,
            )
            same_terminal = sum(
                plan[a, b]
                for a, b in zip(*np.nonzero(plan), strict=True)
                if same_lexical_unit(
                    witnesses, ids[left_dist.anchors[a]], ids[right_dist.anchors[b]]
                )
            )
            selected.append(
                {
                    "left": keys[i],
                    "right": keys[j],
                    "exact": distance,
                    "sinkhorn": smooth,
                    "reverseExact": reverse,
                    "halfSupport": small,
                    "sameTerminalMass": same_terminal,
                    "maxMarginalError": float(
                        max(
                            np.max(abs(plan.sum(1) - left_dist.mass)),
                            np.max(abs(plan.sum(0) - right_dist.mass)),
                        )
                    ),
                    "groundCostMean": float(cost.mean()),
                }
            )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("measurements", "coverage", "encoder", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--graphs", type=Path)
    parser.add_argument(
        "--reuse", type=Path, help="Reuse sealed outcome-independent extraction and distances"
    )
    parser.add_argument(
        "--per-model", type=int, default=100, choices=range(2, 301), metavar="2..300"
    )
    parser.add_argument("--support", type=int, default=8)
    parser.add_argument("--repeat-gates", type=int, default=0)
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=52736)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--cross-maximum", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reference", choices=("prefix", "local"), default="local")
    parser.add_argument("--hidden-reference", type=Path)
    parser.add_argument("--hidden-variant", choices=("all-hidden", "selected-hidden"), default=None)
    parser.add_argument("--epsilon", type=float)
    cut = parser.add_mutually_exclusive_group()
    cut.add_argument("--cluster-count", type=int)
    cut.add_argument("--threshold", type=float)
    cut.add_argument("--discovery-neighbor-quantile", type=float)
    args = parser.parse_args()
    if (
        args.cluster_count is None
        and args.threshold is None
        and args.discovery_neighbor_quantile is None
    ):
        args.discovery_neighbor_quantile = 0.35
    if (
        args.discovery_neighbor_quantile is not None
        and not 0 <= args.discovery_neighbor_quantile <= 1
    ):
        parser.error("Discovery nearest-neighbor quantile must be in [0, 1]")
    if not (
        2 <= args.support <= args.examples
        and args.neighbors > 0
        and args.threads > 0
        and 0 <= args.repeat_gates <= args.per_model // 2
    ):
        parser.error("Need 2 <= support <= examples and positive neighbors and threads")
    if (args.hidden_reference is None) != (args.hidden_variant is None):
        parser.error("Provide both --hidden-reference and --hidden-variant")
    analyze(args)


if __name__ == "__main__":
    main()

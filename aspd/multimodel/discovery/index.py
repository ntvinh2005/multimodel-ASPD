"""Verify measured artifacts and store one shared mmap index of exact anchor graphs."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

MODELS = ("gemma", "smol", "granite")
START = (9, 8, 16)
MATRICES = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")
C = 8192
M = 56
EDGE = np.dtype([("source", "<u4"), ("target", "<u4"), ("E", "<f4"), ("w", "<f4")])
NODE = np.dtype([("id", "<u4"), ("coefficient", "<f4"), ("normalized", "<f4")])
CHECKPOINT = "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path, expected=None):
    blob = path.read_bytes()
    if expected is not None and hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"Recorded hash mismatch: {path}")
    return json.loads(gzip.decompress(blob) if path.suffix == ".gz" else blob)


def encode(name):
    prefix, feature = name.split("/")
    model, layer, matrix = prefix.split(".")
    n = MODELS.index(model)
    local = int(layer.removeprefix("L")) - START[n]
    feature = int(feature)
    if not 0 <= local < 8 or not 0 <= feature < C:
        raise ValueError(f"Component outside pinned checkpoint: {name}")
    return (n * M + local * 7 + MATRICES.index(matrix)) * C + feature


def decode(code):
    matrix, feature = divmod(int(code), C)
    model, local = divmod(matrix, M)
    layer, matrix = divmod(local, 7)
    return f"{MODELS[model]}.L{START[model] + layer:02}.{MATRICES[matrix]}/{feature}"


def model_of(code):
    return int(code) // (C * M)


def gate_of(code):
    matrix, feature = divmod(int(code), C)
    return f"{'PRE' if matrix % 7 < 3 else 'MID'}:{feature}"


def group_split(dataset, family, groups, seed):
    groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{dataset}:{family}:{group}".encode()).digest(),
    )
    cut = max(1, len(groups) // 3) if len(groups) >= 2 else 0
    return set(groups[:cut])


def meta_for_window(row, example, dataset, split):
    text = example["prompt"]
    boundary = row["character_boundary"]
    roles = row.get("roles_ending", [])
    query = row.get("final_query", False)
    return {
        "dataset": dataset,
        "split": split,
        "anchor": row["anchor_id"],
        "pool": row["pool_id"],
        "example": row["example_id"],
        "group": row["group_id"],
        "family": row.get("family", ""),
        "condition": row["condition"],
        "text": text,
        "boundary": boundary,
        "positions": row["model_positions"],
        "role": "final-query" if query else ",".join(sorted(roles)),
        "active": {
            f"{site.upper()}:{key}": value
            for site in ("pre", "mid")
            for key, value in zip(row[site]["active_gate_ids"], row[site]["values"], strict=True)
        },
    }


def token_scopes(root, seed, limit=None):
    index = load(root / "index.json")
    if index["checkpointSha256"] != CHECKPOINT:
        raise ValueError("Wrong checkpoint in token workbench")
    graphs = index["graphs"]
    for dataset, groups in sorted(index["datasets"].items()):
        inventory = []
        for group_id, ref in sorted(groups.items()):
            group = load(root / ref["path"], ref["sha256"])
            if (group["checkpointSha256"], group["dataset"], group["groupId"]) != (
                CHECKPOINT,
                dataset,
                group_id,
            ):
                raise ValueError(f"Unexpected token group: {group_id}")
            for example in group["examples"]:
                for row in example["positions"]:
                    if row["anchor_id"] in graphs:
                        inventory.append((row, example))
        by_family = {}
        for row, _ in inventory:
            by_family.setdefault(row.get("family", ""), set()).add(row["group_id"])
        reserved = set().union(*(group_split(dataset, f, g, seed) for f, g in by_family.items()))
        if limit and len(inventory) > limit:
            # Sample complete measured groups so matched contrasts remain available.
            group_ids = sorted({row["group_id"] for row, _ in inventory})
            chosen = set([group for group in group_ids if group not in reserved][:2])
            chosen.update([group for group in group_ids if group in reserved][:1])
            inventory = [(row, example) for row, example in inventory if row["group_id"] in chosen]
        for row, example in inventory:
            meta = meta_for_window(
                row, example, dataset, "heldout" if row["group_id"] in reserved else "discovery"
            )
            ref = graphs[meta["anchor"]]
            asset = load(root / ref["path"], ref["sha256"])
            expected = (CHECKPOINT, meta["anchor"], meta["example"], meta["pool"])
            if expected != tuple(
                asset[key] for key in ("checkpointSha256", "anchorId", "exampleId", "poolId")
            ):
                raise ValueError("Token graph does not match archived example/pool/anchor")
            yield from token_graphs(asset, {**meta, "artifact": ref["sha256"]})


def token_graphs(asset, meta):
    edges, nodes = {}, {}
    for component, (e, norm, parents) in asset["componentTargets"].items():
        target = encode(component)
        sigma = asset["outputScale"][component.rsplit("/", 1)[0]]
        nodes[target] = (e, abs(e) * norm / sigma)
        for parent, effect in parents:
            edges[encode(parent), target] = (effect, abs(effect) * norm / sigma)
    for gate, graph in asset["official"].items():
        if graph["anchorId"] != meta["anchor"] or not np.isclose(
            graph["activation"], meta["active"].get(gate, 0.0)
        ):
            raise ValueError("Official graph does not match natural support")
    official = [
        target
        for graph in asset["official"].values()
        for record in graph["models"].values()
        for target in record["targets"]
    ]
    for record in [*official, *asset.get("supplementalTargets", {}).values()]:
        target = encode(record["component"]["id"])
        scale = record.get("outputScale")
        contribution = record.get("contributionNorm")
        nodes.setdefault(
            target,
            (
                record["coefficient"],
                contribution / scale if scale and contribution is not None else np.nan,
            ),
        )
        for parent in record["parents"]:
            key = encode(parent["component"]["id"]), target
            existing = edges.get(key)
            pair = parent["E"], parent["w"]
            if existing is not None and not np.allclose(existing, pair, rtol=1e-5, atol=1e-10):
                raise ValueError("Conflicting same-anchor edge measurements")
            edges[key] = pair
    partition = [[] for _ in MODELS]
    for (source, target), (signed, weight) in edges.items():
        n = model_of(target)
        if model_of(source) != n:
            raise ValueError("Unexpected cross-model edge")
        partition[n].append((source, target, signed, weight))
    coefficients = [[] for _ in MODELS]
    for node, values in nodes.items():
        coefficients[model_of(node)].append((node, *values))
    for n, model in enumerate(MODELS):
        edge = np.array(partition[n], dtype=EDGE)
        scalar = np.array(coefficients[n], dtype=NODE)
        if len(edge) or len(scalar):
            yield {**meta, "model": model, "complete": asset["completeComponents"]}, edge, scalar


def natural_scopes(root, limit=None, witness_anchors=()):
    profile = root / "profiles"
    complete = load(profile / "COMPLETE")
    manifest = load(profile / "manifest.json", complete["sha256"]["manifest.json"])
    if complete["status"] != "COMPLETE" or manifest["rule"] != {"k_par": 10, "tau": 0}:
        raise ValueError("Unknown Natural Validation parent rule or incomplete archive")
    edge_archive = load(root / "edges" / "COMPLETE")
    edge_manifest = load(root / "edges" / "manifest.json", edge_archive["sha256"]["manifest.json"])
    if edge_manifest["config_fingerprint"] != manifest["config_fingerprint"]:
        raise ValueError("Natural Validation edge/profiles configuration mismatch")
    shard = root / "edges-shard-000-of-008"
    shard_marker = load(shard / "COMPLETE")
    backend = load(shard / "manifest.json", shard_marker["sha256"]["manifest.json"])["backend"]
    if (
        backend["split"] != "validation"
        or sha(Path(backend["checkpoint"]) / "manifest.json") != CHECKPOINT
    ):
        raise ValueError("Natural Validation checkpoint/split mismatch")
    vocab = load(profile / "parents_vocab.json", complete["sha256"]["parents_vocab.json"])
    encoded = np.array([encode(key) for key in vocab["components"]], dtype=np.uint32)
    file = profile / "parents_per_anchor.safetensors"
    digest = complete["sha256"][file.name]
    if sha(file) != digest:
        raise ValueError("Natural Validation parent tensor changed")
    with safe_open(file, framework="numpy") as data:
        anchor = data.get_tensor("anchor")
        targets = data.get_tensor("downstream")
        parents = data.get_tensor("parent")
        score = data.get_tensor("E")
        weight = data.get_tensor("w")
        if len(anchor) != manifest["parent_rows"]:
            raise ValueError("Natural Validation parent row count differs")
        if limit:
            sample = np.linspace(
                0, len(vocab["anchor_ids"]) - 1, min(limit, len(vocab["anchor_ids"])), dtype=int
            )
            known = [i for i, value in enumerate(vocab["anchor_ids"]) if value in witness_anchors]
            sample = np.union1d(sample, known)
            selected = np.flatnonzero(np.isin(anchor, sample))
        else:
            selected = np.arange(len(anchor), dtype=np.int32)
        targets = encoded[targets[selected]]
        parents = encoded[parents[selected]]
        model = targets // (C * M)
        if np.any(parents // (C * M) != model):
            raise ValueError("Unexpected cross-model edge in Natural Validation")
        key = anchor[selected] * len(MODELS) + model
        order = np.argsort(key, stable=True)
        keys = key[order]
        cuts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
            n, m = divmod(int(keys[lo]), len(MODELS))
            indices = order[lo:hi]
            values = np.empty(hi - lo, dtype=EDGE)
            values["source"], values["target"] = parents[indices], targets[indices]
            values["E"], values["w"] = score[selected[indices]], weight[selected[indices]]
            meta = {
                "dataset": "natural-validation",
                "split": "discovery-only",
                "anchor": vocab["anchor_ids"][n],
                "model": MODELS[m],
                "pool": None,
                "example": None,
                "group": None,
                "complete": False,
                "artifact": digest,
            }
            yield meta, values, np.empty(0, dtype=NODE)


def heldout_natural_scopes(root, pilot):
    graph_index = load(root / "index.json")
    source = load(pilot)
    if graph_index["checkpointSha256"] != CHECKPOINT or sha(pilot) != graph_index["pilotSha256"]:
        raise ValueError("Natural test provenance differs")
    points = {}
    for window in source["examples"].values():
        for position, point in window["points"].items():
            if point["anchor"] in graph_index["graphs"]:
                points[point["anchor"]] = (window, int(position), point)
    for anchor, ref in sorted(graph_index["graphs"].items()):
        window, pos, point = points[anchor]
        asset = load(root / ref["path"], ref["sha256"])
        document, pool = window["documentId"], str(point["pool"])
        if (asset["anchorId"], asset["exampleId"], asset["poolId"], asset["checkpointSha256"]) != (
            anchor,
            document,
            pool,
            CHECKPOINT,
        ):
            raise ValueError("Natural test graph differs from archived token")
        tokens = [token.replace("▁", " ") for token in window["tokens"]]
        meta = {
            "dataset": "natural-test-pilot",
            "split": "exploratory-test",
            "anchor": anchor,
            "pool": pool,
            "example": document,
            "group": document,
            "positions": point["positions"],
            "text": "".join(tokens),
            "boundary": len("".join(tokens[: pos + 1])),
            "active": point["gates"],
            "artifact": ref["sha256"],
        }
        yield from token_graphs(asset, meta)


def write_index(root, sources):
    scopes = []
    counts = Counter()
    edge_offset = node_offset = 0
    seen = set()
    with (root / "edges.bin").open("xb") as edges, (root / "nodes.bin").open("xb") as nodes:
        for source in sources:
            for meta, edge, node in source:
                identity = meta["dataset"], meta["model"], meta["anchor"]
                if identity in seen:
                    raise ValueError(f"Duplicate graph scope: {identity}")
                seen.add(identity)
                if not np.all(np.isfinite(edge["E"]) & np.isfinite(edge["w"]) & (edge["w"] >= 0)):
                    raise ValueError(f"Invalid graph numeric fields: {identity}")
                edge.tofile(edges)
                node.tofile(nodes)
                scopes.append(
                    {
                        **meta,
                        "edgeStart": edge_offset,
                        "edgeCount": len(edge),
                        "nodeStart": node_offset,
                        "nodeCount": len(node),
                    }
                )
                edge_offset += len(edge)
                node_offset += len(node)
                counts[f"{meta['dataset']}:{meta['split']}"] += 1
    (root / "scopes.json").write_text(json.dumps(scopes, separators=(",", ":"), allow_nan=False))
    return {
        "edges": edge_offset,
        "coefficientRows": node_offset,
        "scopes": len(scopes),
        "bySource": dict(counts),
    }


class Index:
    def __init__(self, root):
        self.scopes = load(root / "scopes.json")
        self.edges = (
            np.memmap(root / "edges.bin", dtype=EDGE, mode="r")
            if (root / "edges.bin").stat().st_size
            else np.empty(0, dtype=EDGE)
        )
        self.nodes = (
            np.memmap(root / "nodes.bin", dtype=NODE, mode="r")
            if (root / "nodes.bin").stat().st_size
            else np.empty(0, dtype=NODE)
        )
        self.edge_starts = np.array([row["edgeStart"] for row in self.scopes])

    def graph(self, scope_id):
        row = self.scopes[scope_id]
        edge, node = row["edgeStart"], row["nodeStart"]
        return self.edges[edge : edge + row["edgeCount"]], self.nodes[
            node : node + row["nodeCount"]
        ]

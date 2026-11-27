"""Component-level transport over measured Natural Validation causal prefixes."""

from __future__ import annotations

import hashlib
import re
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import coo_matrix

from aspd.multimodel.discovery.index import C, M, decode, gate_of, model_of
from aspd.multimodel.discovery.natural_validation import measurements

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_FILES = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
}


def is_heldout(document: str, seed: int):
    digest = hashlib.sha256(f"{seed}:{document}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 4 == 0


def normalized(weights):
    values = np.asarray(weights, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("OT weights must have finite positive total mass")
    return values / values.sum()


@lru_cache
def marginals(n: int, m: int):
    cells = np.arange(n * m)
    rows = np.r_[np.repeat(np.arange(n), m), n + np.tile(np.arange(m), n)]
    return coo_matrix((np.ones(2 * n * m), (rows, np.tile(cells, 2))), shape=(n + m, n * m)).tocsr()


def transport(left_z, right_z, left_mass, right_mass):
    """Exact balanced transport with cosine ground cost and a sparse constraint matrix."""
    left, right = normalized(left_mass), normalized(right_mass)
    cost = np.clip(1 - left_z @ right_z.T, 0, 2)
    if cost.shape != (len(left), len(right)) or not np.isfinite(cost).all():
        raise ValueError("Embedding and mass shapes differ")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unrecognized options detected.*", OptimizeWarning)
        result = linprog(
            cost.ravel(),
            A_eq=marginals(*cost.shape),
            b_eq=np.r_[left, right],
            bounds=(0, None),
            method="highs",
            options={"threads": 1},
        )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun), result.x.reshape(cost.shape), cost


def without_terminal_unit(prefix):
    """Remove one trailing lexical or punctuation unit from the existing causal prefix."""
    return re.sub(r"[^\w\s]+\s*$|\w+\s*$", "", prefix).rstrip()


def embed(prefixes, snapshot: Path, *, threads: int, max_tokens: int = 128):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    tokenizer.truncation_side = "left"
    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    ).eval()
    parts = []
    with torch.inference_mode():
        for first in range(0, len(prefixes), 24):
            batch = tokenizer(
                prefixes[first : first + 24],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            parts.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    return np.concatenate(parts)


def build_supports(archive, rows, seed, max_per_model, structural=(), structural_cap=0):
    """Build sparse positive-mass supports, retaining comparison coverage separately."""
    discovery = np.array([not is_heldout(row["document"], seed) for row in rows])
    repeated = {row["prefixSha256"] for i, row in enumerate(rows) if discovery[i]}
    usable = np.array(
        [
            len(row["prefix"].split()) >= 12
            and (discovery[i] or row["prefixSha256"] not in repeated)
            for i, row in enumerate(rows)
        ]
    )
    available = np.zeros(len(rows), dtype=np.uint8)
    for ids, anchors, _, _ in measurements(archive, rows):
        np.bitwise_or.at(available, anchors, (1 << (ids // (M * C))).astype(np.uint8))
    usable &= available == 7
    buckets = defaultdict(lambda: defaultdict(dict))
    coverage = defaultdict(set)
    total = Counter()
    zeros = Counter()
    for ids, anchors, coefficients, mass in measurements(archive, rows):
        keep = usable[anchors]
        for code, anchor, coefficient, weight in zip(
            ids[keep],
            anchors[keep],
            coefficients[keep],
            mass[keep],
            strict=True,
        ):
            model = model_of(code)
            split = "discovery" if discovery[anchor] else "heldout"
            coverage[model, split].add(int(anchor))
            total[model, split] += 1
            if weight == 0:
                zeros[model, split] += 1
                continue
            target = buckets[int(code)][split]
            if int(anchor) in target:
                raise ValueError("Duplicate component and exact anchor in measured support")
            target[int(anchor)] = (float(coefficient), float(weight))
    selected = {}
    eligible = Counter()
    for model in range(3):
        model_rows = [
            (code, by_split)
            for code, by_split in buckets.items()
            if model_of(code) == model
            and len(by_split["discovery"]) >= 6
            and len({rows[i]["document"] for i in by_split["discovery"]}) >= 4
        ]
        eligible[model] = len(model_rows)
        # Stable non-semantic draw from eligible exact component IDs.
        model_rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row[0]}".encode()).digest())
        chosen = model_rows[:max_per_model]
        selected.update(chosen)
        extras = [item for item in model_rows if item[0] in structural and item[0] not in selected]
        selected.update(extras[:structural_cap])
    summary = [
        dict(
            model=model,
            split=split,
            usableMeasuredAnchors=len(ids),
            coefficientRows=total[model, split],
            measuredZeros=zeros[model, split],
        )
        for (model, split), ids in sorted(coverage.items())
    ]
    return selected, summary, dict(eligible)


def centroid(points, mass):
    center = normalized(mass) @ points
    length = np.linalg.norm(center)
    return center / length if length else center


def prefilter(codes, centers, neighbors):
    pairs = set()
    similarities = centers @ centers.T
    for i, code in enumerate(codes):
        for model in range(3):
            same_model = [j for j, other in enumerate(codes) if i != j and model_of(other) == model]
            ordered = sorted(same_model, key=lambda j: (-similarities[i, j], codes[j]))
            pairs.update(tuple(sorted((i, j))) for j in ordered[:neighbors])
            pairs.update(
                tuple(sorted((i, j))) for j in same_model if gate_of(code) == gate_of(codes[j])
            )
    return sorted(pairs)


def control_signature(support, rows, model):
    counts = Counter()
    for i, (_, mass) in support.items():
        row = rows[i]
        p = row["prefix"].rstrip()
        token = row["tokenIds"][model]
        for key in (
            ("source", row["source"]),
            ("token", token),
            ("position", row["positions"][model] // 32),
            ("punctuation", bool(p and not p[-1].isalnum())),
        ):
            counts[key] += mass
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def comparison(left, right, embeddings, cached):
    if len(left) < 3 or len(right) < 3:
        return None, []
    left_ids, right_ids = list(left), list(right)
    x = np.array([embeddings[cached[i]] for i in left_ids])
    y = np.array([embeddings[cached[i]] for i in right_ids])
    a = normalized([left[i][1] for i in left_ids])
    b = normalized([right[i][1] for i in right_ids])
    dist, plan, cost = transport(x, y, a, b)
    lookup = dict(zip(left_ids, a, strict=True))
    overlap = float(sum(min(v, lookup.get(i, 0)) for i, v in zip(right_ids, b, strict=True)))
    pooled = float(np.clip(1 - centroid(x, a) @ centroid(y, b), 0, 2))
    cells = list(zip(*np.where(plan > 1e-7), strict=True))
    top = sorted(cells, key=lambda cell: (-plan[cell], cost[cell]))[:2]
    worst = sorted(cells, key=lambda cell: (-cost[cell], -plan[cell]))[:1]
    matches = [
        dict(
            left=left_ids[i],
            right=right_ids[j],
            mass=float(plan[i, j]),
            cost=float(cost[i, j]),
            kind=kind,
        )
        for kind, cell_list in (("match", top), ("counterexample", worst))
        for i, j in cell_list
    ]
    return {"ot": dist, "centroid": pooled, "anchorOverlap": overlap}, matches


def cluster_pairs(n, pairs, threshold):
    """Complete-link groups over tested distances; unknown distances cannot join groups."""
    distances = {(i, j): d for i, j, d in pairs}
    groups = {i: {i} for i in range(n)}
    parents = list(range(n))
    for i, j, d in sorted(pairs, key=lambda row: (row[2], row[0], row[1])):
        left, right = parents[i], parents[j]
        if left == right or d > threshold:
            continue
        if not all(
            distances.get(tuple(sorted(pair)), np.inf) <= threshold
            for pair in ((a, b) for a in groups[left] for b in groups[right])
        ):
            continue
        groups[left].update(groups.pop(right))
        for k in groups[left]:
            parents[k] = left
    return [sorted(values) for values in groups.values() if len(values) > 1]


def select_control(i, j, codes, supports, signatures):
    target = supports[codes[j]]["discovery"]
    identity = decode(codes[j]).split("/")[0].split(".")
    options = []
    for k, code in enumerate(codes):
        if (
            k in (i, j)
            or model_of(code) != model_of(codes[j])
            or gate_of(code) == gate_of(codes[j])
        ):
            continue
        cand = supports[code]["discovery"]
        if not 0.67 <= len(cand) / len(target) <= 1.5:
            continue
        # Compare corpus, terminal token, punctuation and position histograms.
        left, right = signatures[j], signatures[k]
        discrepancy = (
            sum(abs(left.get(key, 0) - right.get(key, 0)) for key in left.keys() | right.keys()) / 2
        )
        key = decode(code).split("/")[0].split(".")
        options.append(
            (discrepancy, key[2] != identity[2], abs(int(key[1][1:]) - int(identity[1][1:])), k)
        )
    return min(options, default=None)


def attach_graphs(index, scopes, rows, codes, support, *, limit=3):
    # Read the immutable graph index only for exact recovered Natural Validation anchors.
    result = []
    for code in codes:
        model = decode(code).split(".")[0]
        candidates = sorted(
            support[code]["discovery"].items(), key=lambda row: (-row[1][1], row[0])
        )
        for anchor, _ in candidates:
            scope = scopes.get((rows[anchor]["anchor"], model))
            if scope is None:
                continue
            edge, _ = index.graph(scope)
            direct = edge[(edge["source"] == code) | (edge["target"] == code)]
            if not len(direct):
                continue
            saved = sorted(direct.tolist(), key=lambda item: (-item[3], item[0], item[1]))[:12]
            result.append(
                dict(
                    anchor=anchor,
                    component=decode(code),
                    model=model,
                    artifact=index.scopes[scope]["artifact"],
                    edges=[
                        dict(source=decode(a), target=decode(b), E=e, w=w) for a, b, e, w in saved
                    ],
                )
            )
            break
        if len(result) == limit:
            break
    return result


def summary_rows(support):
    return {
        split: dict(
            measuredNonzero=len(part),
            normalizedMass=sum(v[1] for v in part.values()),
            positiveMass=sum(v[1] for v in part.values() if v[0] > 0),
            negativeMass=sum(v[1] for v in part.values() if v[0] < 0),
        )
        for split, part in support.items()
    }

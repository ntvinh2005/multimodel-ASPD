"""Matched, document-disjoint comparison for fresh Natural Validation measurements."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aspd.multimodel.discovery.semantic_ot import (
    ENCODER_FILES,
    centroid,
    is_heldout,
    transport,
    without_terminal_unit,
)

MIN_HELDOUT = 10
MEAN_HELDOUT = 20
DISCOVERY_TARGET = 20
SUPPORT_LIMIT = 24
VARIANTS = ("prefix", "fullPrefix", "local", "anchor", "combined")
SPLIT_SEED = 41793


@dataclass(frozen=True)
class Anchor:
    id: str
    document: str
    source: str
    pool: int
    boundary: int
    prefix: str
    native: dict[str, list[int]]

    @property
    def split(self) -> str:
        return "heldout" if is_heldout(self.document, SPLIT_SEED) else "discovery"


def units(text: str):
    return re.findall(r"\w+|[^\w\s]", text.casefold())[-64:]


def lexical_key(prefix: str):
    tokens = units(prefix)
    return tokens[-1] if tokens else ""


def shingles(prefix: str):
    strings = units(prefix)
    width = min(4, len(strings))
    return frozenset(tuple(strings[i : i + width]) for i in range(len(strings) - width + 1))


def diverse(rows, magnitude, indices, *, forbidden=()):
    """At most one observed anchor per document; reject near-identical causal suffixes."""
    documents = set()
    signatures = []
    by_shingle = defaultdict(set)

    def add(index):
        start = len(signatures)
        signature = shingles(rows[index].prefix)
        signatures.append(signature)
        for item in signature:
            by_shingle[item].add(start)

    for i in forbidden:
        add(i)
    keep = []
    for i in sorted(indices, key=lambda j: (-abs(magnitude[j]), rows[j].id)):
        doc = rows[i].document
        if doc in documents:
            continue
        signature = shingles(rows[i].prefix)
        possible = set().union(*(by_shingle[item] for item in signature))
        if not signature or any(
            len(signature & signatures[n]) >= 0.85 * len(signature | signatures[n])
            for n in possible
        ):
            continue
        documents.add(doc)
        add(i)
        keep.append(int(i))
    return keep


def coverage(components):
    """Every endpoint and the mean over unique retained components must pass."""
    counts = [item["coverage"]["heldout"] for item in components.values()]
    if not counts or min(counts) < MIN_HELDOUT or np.mean(counts) < MEAN_HELDOUT:
        raise ValueError("No publishable catalog meets distinct held-out context thresholds")
    return {
        "components": len(counts),
        "minimumHeldout": min(counts),
        "meanHeldout": float(np.mean(counts)),
        "discoveryAtTarget": sum(
            item["coverage"]["discovery"] >= DISCOVERY_TARGET for item in components.values()
        ),
    }


def select_for_ot(rows, ids, *, salt):
    # Selection is frozen and independent of held-out outcomes. The browser retains every example.
    return sorted(ids, key=lambda i: hashlib.sha256(f"{salt}/{rows[i].id}".encode()).digest())[
        :SUPPORT_LIMIT
    ]


def matched_controls(rows, model, targets, eligible):
    """Same-component zero observations, with lexical/source/position matching tiers."""
    bins = defaultdict(list)
    for i in eligible:
        row = rows[i]
        pos = row.native[model][0] // 16
        token = lexical_key(row.prefix)
        native_id = row.native[model][1]
        bins[row.source, native_id, pos].append(i)
        bins[row.source, token, pos].append(i)
        bins[row.source, token].append(i)
        bins[row.source, pos].append(i)
    used = {rows[i].document for i in targets}
    chosen = []
    tiers = []
    for i in targets:
        row = rows[i]
        pos = row.native[model][0] // 16
        token = lexical_key(row.prefix)
        for tier, key in (
            ("source-native-token-position", (row.source, row.native[model][1], pos)),
            ("source-token-position", (row.source, token, pos)),
            ("source-token", (row.source, token)),
            ("source-position", (row.source, pos)),
        ):
            available = [j for j in bins[key] if rows[j].document not in used]
            if available:
                j = min(
                    available,
                    key=lambda n: (
                        abs(len(rows[n].prefix) - len(row.prefix)),
                        rows[n].id,
                    ),
                )
                used.add(rows[j].document)
                chosen.append(j)
                tiers.append(tier)
                break
        else:
            return [], []
    return chosen, tiers


def embed_variants(prefixes, snapshot: Path, threads: int):
    """No tokens beyond the annotated boundary enter the common embedding model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from aspd.multimodel.posthoc.artifacts import sha256_file

    for name, expected in ENCODER_FILES.items():
        if sha256_file(snapshot / name) != expected:
            raise ValueError(f"Invalid embedding snapshot file {name}")
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    tokenizer.truncation_side = "left"
    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    ).eval()
    normalize = torch.nn.functional.normalize

    def encode(text, limit, original_pooling=False):
        batch = tokenizer(
            text,
            return_tensors="pt",
            max_length=limit,
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
        )
        special = batch.pop("special_tokens_mask").bool()
        hidden = model(**batch).last_hidden_state
        native = batch["attention_mask"].bool() & ~special
        attended = batch["attention_mask"].bool()
        pooling = (
            attended
            if original_pooling
            else torch.where(native.any(1, keepdim=True), native, attended)
        )
        mean = (hidden * pooling.unsqueeze(-1)).sum(1) / pooling.sum(1, keepdim=True)
        indices = torch.arange(hidden.size(1)).expand_as(native)
        last = torch.where(native, indices, 0).max(1).values
        end = hidden[torch.arange(len(text)), last]
        return normalize(mean, dim=1).numpy(), normalize(end, dim=1).numpy()

    parts = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(prefixes), 24):
            text = prefixes[start : start + 24]
            prefix, anchor = encode(text, 128, original_pooling=True)
            full, _ = encode(text, int(model.config.max_position_embeddings))
            local, _ = encode(text, 40)
            no_terminal, _ = encode([without_terminal_unit(x) for x in text], 40)
            for variant, value in (
                ("prefix", prefix),
                ("fullPrefix", full),
                ("local", local),
                ("anchor", anchor),
                ("combined", np.concatenate((local, anchor), axis=1) / np.sqrt(2)),
                ("withoutTerminal", no_terminal),
            ):
                parts[variant].append(value)
    return {name: np.concatenate(values) for name, values in parts.items()}


def comparison(a, b, embedding, magnitudes, left, right, *, mass_b=None):
    x, y = embedding[a], embedding[b]
    px = np.abs(magnitudes[a, left])
    py = np.abs(magnitudes[b, right]) if mass_b is None else mass_b
    value, plan, cost = transport(x, y, px, py)
    mean_distance = float(np.clip(1 - centroid(x, px) @ centroid(y, py), 0, 2))
    overlap = len(set(a) & set(b)) / len(set(a) | set(b))
    return (
        {
            "ot": value,
            "centroid": mean_distance,
            "anchorOverlap": overlap,
            "distinctContextMass": float(plan[~np.equal.outer(a, b)].sum()),
        },
        plan,
        cost,
    )


def bootstrap_advantage(rows, left, right, left_zero, right_zero, embedding, px, py, rng, draws):
    """Estimate the smaller directional advantage with common document weights."""
    bags = (left, right, left_zero, right_zero)
    docs = sorted({rows[i].document for bag in bags for i in bag})
    doc_index = {document: i for i, document in enumerate(docs)}
    li, ri, lzi, rzi = [np.array([doc_index[rows[i].document] for i in bag]) for bag in bags]
    a, b, az, bz = [embedding[bag] for bag in bags]
    out = []
    # Bayesian document bootstrap: positive multipliers preserve the matched support.
    for _ in range(draws):
        draw = rng.exponential(size=len(docs))
        lw, rw = px * draw[li], py * draw[ri]
        score = transport(a, b, lw, rw)[0]
        distance_a = transport(b, az, rw, px * draw[lzi])[0]
        distance_b = transport(a, bz, lw, py * draw[rzi])[0]
        out.append(min(distance_a, distance_b) - score)
    return [float(q) for q in np.quantile(out, [0.025, 0.975])]

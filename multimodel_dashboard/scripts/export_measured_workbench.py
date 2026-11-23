#!/usr/bin/env python3
"""Export a bounded, measured ASPD v3.1 Workbench bundle.

The raw post-hoc run is tens of gigabytes and cannot be loaded by a browser.  This
exporter reads the immutable edge shards with NumPy memory maps and emits a small JSON
subset.  The scientific quantities are not recomputed with new settings:

* representatives come from ``representatives/representatives.json``;
* parent sets use the run's frozen ``K_par=10`` and strict ``tau=0`` rule;
* signed ``E`` and component importance values are copied from the raw tensors;
* ``w = |E| ||u_a|| / sigma_a`` and parent distributions follow the formulation.

Only the visualization is bounded.  Gate and anchor selection rules are written into
the output manifest so this file cannot be mistaken for a dictionary-wide estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DTYPES = {
    "BOOL": np.dtype("?"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
}
MATRIX_LABEL = {
    "q": "Q",
    "k": "K",
    "v": "V",
    "o": "O",
    "mlp_gate": "gate",
    "mlp_up": "up",
    "mlp_down": "down",
}
MODEL_SPECS = {
    "gemma": {
        "label": "Gemma 2 2B IT",
        "layers": list(range(9, 17)),
    },
    "smol": {
        "label": "SmolLM2 1.7B Instruct",
        "layers": list(range(8, 16)),
    },
    "granite": {
        "label": "Granite 3.3 2B Instruct",
        "layers": list(range(16, 24)),
    },
}
DATASET_REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"
QUICK_CONTEXT_PROTOCOL_SHA256 = "5ea04ee18e76f9fbf71c6816f7a31b052550e7f1a940a369d5dced5429338285"
QUICK_CONTEXT_DISPLAY_MODEL_KEY = "gemma"
COMPONENT_RE = re.compile(
    r"^(?P<model>[^.]+)\.L(?P<layer>\d+)\.(?P<matrix>[^/]+)/(?P<feature>\d+)$"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SafeTensorFile:
    """Minimal read-only safetensors reader backed by NumPy memory maps."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(header_size))
        self.data_start = 8 + header_size

    def tensor(self, name: str) -> np.ndarray:
        spec = self.header[name]
        dtype = DTYPES.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported dtype {spec['dtype']} for {name}")
        start, stop = spec["data_offsets"]
        shape = tuple(spec["shape"])
        expected = math.prod(shape) * dtype.itemsize
        if stop - start != expected:
            raise ValueError(f"malformed tensor byte range for {name}")
        return np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=self.data_start + start,
            shape=shape,
        )


class ProfileStore:
    """Verified read-only adapter for the completed parent-profile stage."""

    def __init__(self, root: Path):
        self.root = root / "profiles"
        complete = read_json(self.root / "COMPLETE")
        required = (
            "manifest.json",
            "parents_vocab.json",
            "parents_per_anchor.safetensors",
            "distributions_vocab.json",
            "parent_distributions.safetensors",
        )
        for name in required:
            actual = sha256_file(self.root / name)
            if actual != complete["sha256"][name]:
                raise ValueError(f"profiles artifact failed SHA-256 verification: {name}")
        self.complete_hash = sha256_file(self.root / "COMPLETE")
        self.source_hashes = {name: complete["sha256"][name] for name in required}
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["runtime"]["git_sha"] != "1348bf9587122c396e2b741cc8ba7e7394a13253":
            raise ValueError("unsupported parent-profile source revision")
        if self.manifest["rule"] != {"k_par": 10, "tau": 0.0}:
            raise ValueError("parent-profile rule differs from the frozen Workbench contract")

        parent_vocab = read_json(self.root / "parents_vocab.json")
        self.parent_anchor_ids = parent_vocab["anchor_ids"]
        self.parent_components = parent_vocab["components"]
        self.parent_anchor_index = {key: index for index, key in enumerate(self.parent_anchor_ids)}
        self.parent_component_index = {
            key: index for index, key in enumerate(self.parent_components)
        }
        self.parents = SafeTensorFile(self.root / "parents_per_anchor.safetensors")

        distribution_vocab = read_json(self.root / "distributions_vocab.json")
        self.distribution_anchor_index = {
            key: index for index, key in enumerate(distribution_vocab["anchor_ids"])
        }
        self.distribution_gate_index = {
            key: index for index, key in enumerate(distribution_vocab["gates"])
        }
        self.distribution_gates = distribution_vocab["gates"]
        self.distribution_model_index = {
            key: index for index, key in enumerate(distribution_vocab["models"])
        }
        self.distributions = SafeTensorFile(self.root / "parent_distributions.safetensors")

    def parent_rows(
        self, requested_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return stored Pa rows for the requested (anchor, downstream component) pairs."""
        component_count = len(self.parent_components)
        code_to_pair: dict[int, tuple[str, str]] = {}
        for anchor_id, downstream in requested_pairs:
            try:
                code = (
                    self.parent_anchor_index[anchor_id] * component_count
                    + self.parent_component_index[downstream]
                )
            except KeyError as error:
                raise ValueError(f"profiles artifact lacks requested parent row: {error}") from error
            code_to_pair[code] = (anchor_id, downstream)
        requested_codes = np.asarray(sorted(code_to_pair), dtype=np.int64)
        result = {pair: [] for pair in code_to_pair.values()}

        anchor = self.parents.tensor("anchor")
        downstream = self.parents.tensor("downstream")
        parent = self.parents.tensor("parent")
        rank = self.parents.tensor("rank")
        score = self.parents.tensor("E")
        weight = self.parents.tensor("w")
        chunk = 1_000_000
        for start in range(0, len(anchor), chunk):
            stop = min(start + chunk, len(anchor))
            codes = anchor[start:stop].astype(np.int64) * component_count + downstream[
                start:stop
            ].astype(np.int64)
            positions = np.flatnonzero(np.isin(codes, requested_codes)) + start
            for position in positions:
                pair = code_to_pair[
                    int(anchor[position]) * component_count + int(downstream[position])
                ]
                parent_component = self.parent_components[int(parent[position])]
                result[pair].append(
                    {
                        "component": parent_component,
                        "parentGate": component_gate(parent_component),
                        "rank": int(rank[position]) + 1,
                        "score": float(score[position]),
                        "weight": float(weight[position]),
                    }
                )
        for rows in result.values():
            rows.sort(key=lambda row: row["rank"])
        return result

    def distribution(self, gate: str, model: str, anchor_id: str) -> dict[str, Any]:
        """Read stored formulation S and pi for one (gate, model, anchor) row."""
        try:
            gate_index = self.distribution_gate_index[gate]
            model_index = self.distribution_model_index[model]
            anchor_index = self.distribution_anchor_index[anchor_id]
        except KeyError as error:
            raise ValueError(f"profiles artifact lacks requested distribution: {error}") from error
        header_gate = self.distributions.tensor("header_gate")
        header_model = self.distributions.tensor("header_model")
        header_anchor = self.distributions.tensor("header_anchor")
        matches = np.flatnonzero(
            (header_gate == gate_index)
            & (header_model == model_index)
            & (header_anchor == anchor_index)
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one profile distribution for {(gate, model, anchor_id)}, got {len(matches)}"
            )
        row = int(matches[0])
        entry_row = self.distributions.tensor("entry_row")
        start = int(np.searchsorted(entry_row, row, side="left"))
        stop = int(np.searchsorted(entry_row, row, side="right"))
        entry_gate = self.distributions.tensor("entry_xi")[start:stop]
        entry_score = self.distributions.tensor("entry_S")[start:stop]
        entry_pi = self.distributions.tensor("entry_pi")[start:stop]
        valid = bool(self.distributions.tensor("header_valid")[row])
        values = {
            gate_key(*_split_gate(self.distribution_gates[int(xi)])): float(pi)
            for xi, pi in zip(entry_gate, entry_pi)  # noqa: B905 - lengths share one tensor row
        } if valid else None
        scores = {
            gate_key(*_split_gate(self.distribution_gates[int(xi)])): float(value)
            for xi, value in zip(entry_gate, entry_score)  # noqa: B905 - same row length
        }
        return {
            "status": "measured" if valid else "undefined",
            "totalStrength": float(self.distributions.tensor("header_M")[row]),
            "values": values,
            "scores": scores,
            "representatives": int(self.distributions.tensor("header_representatives")[row]),
        }


def parse_component(key: str) -> tuple[str, int, str, int]:
    match = COMPONENT_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"malformed component key {key!r}")
    return (
        match.group("model"),
        int(match.group("layer")),
        match.group("matrix"),
        int(match.group("feature")),
    )


def gate_key(site: str, feature: int) -> str:
    return f"{site.upper()}:{feature}"


def component_gate(key: str) -> str:
    _, _, matrix, feature = parse_component(key)
    site = "pre" if matrix in {"q", "k", "v"} else "mid"
    return gate_key(site, feature)


def gate_sort_key(key: str) -> tuple[int, int]:
    site, feature = key.split("/", 1)
    return (0 if site == "pre" else 1, int(feature))


def component_sort_key(key: str) -> tuple[str, int, str, int]:
    return parse_component(key)


def canonicalize_numbers(value: Any) -> Any:
    """Make JSON bytes stable across Python/NumPy summation implementations."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in measured export")
        rounded = round(value, 12)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, list):
        return [canonicalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize_numbers(item) for key, item in value.items()}
    return value


def choose_gates(
    representatives: dict[str, dict[str, list[str]]],
    selection: dict[str, Any],
    per_site: int,
) -> list[str]:
    selected = set(selection["gates"])
    eligible: dict[str, list[str]] = {"pre": [], "mid": []}
    for gate, by_model in representatives.items():
        site, _ = gate.split("/", 1)
        if (
            gate in selected
            and selection["anchors"].get(gate)
            and all(by_model.get(model) for model in MODEL_SPECS)
        ):
            eligible[site].append(gate)
    result: list[str] = []
    for site in ("pre", "mid"):
        ordered = sorted(eligible[site], key=gate_sort_key)
        if len(ordered) < per_site:
            raise ValueError(f"only {len(ordered)} eligible {site} gates; need {per_site}")
        result.extend(ordered[:per_site])
    return result


class EdgeShard:
    def __init__(self, directory: Path):
        self.directory = directory
        self.meta = read_json(directory / "edge_records.json")
        self.selection = read_json(directory / "selection.json")
        self.tensors = SafeTensorFile(directory / "edges.safetensors")
        self.anchor_lookup = {key: i for i, key in enumerate(self.meta["anchor_ids"])}
        self.downstream_lookup = {
            key: i for i, key in enumerate(self.meta["downstream_keys"])
        }
        self.record_anchor = self.tensors.tensor("record_anchor")
        self.record_downstream = self.tensors.tensor("record_downstream")
        self.record_start = self.tensors.tensor("record_start")
        self.record_stop = self.tensors.tensor("record_stop")
        self.score = self.tensors.tensor("score")
        self.upstream_feature = self.tensors.tensor("upstream_feature")
        self.upstream_matrix = self.tensors.tensor("upstream_matrix")
        self._row_cache: dict[tuple[int, int], int] | None = None

    def _rows(self) -> dict[tuple[int, int], int]:
        if self._row_cache is None:
            if len(self.record_anchor) != len(self.record_downstream):
                raise ValueError("edge record anchor/downstream arrays have different lengths")
            self._row_cache = {
                (int(anchor), int(downstream)): row
                for row, (anchor, downstream) in enumerate(
                    zip(self.record_anchor, self.record_downstream)  # noqa: B905
                )
            }
        return self._row_cache

    def parents(self, anchor_id: str, downstream: str, k_par: int, tau: float) -> list[dict[str, Any]]:
        anchor_index = self.anchor_lookup.get(anchor_id)
        downstream_index = self.downstream_lookup.get(downstream)
        if anchor_index is None or downstream_index is None:
            return []
        row = self._rows().get((anchor_index, downstream_index))
        if row is None:
            return []
        start, stop = int(self.record_start[row]), int(self.record_stop[row])
        scores = self.score[start:stop]
        features = self.upstream_feature[start:stop]
        matrices = self.upstream_matrix[start:stop]
        if not (len(scores) == len(features) == len(matrices)):
            raise ValueError("edge score/component arrays have different lengths")
        records = []
        matrix_keys = self.meta["matrix_keys"]
        for score, feature, matrix_index in zip(scores, features, matrices):  # noqa: B905
            key = f"{matrix_keys[int(matrix_index)]}/{int(feature)}"
            records.append({"component": key, "score": float(score)})
        records.sort(key=lambda item: (-abs(item["score"]), component_sort_key(item["component"])))
        return [item for item in records[:k_par] if abs(item["score"]) > tau]


def importance_value(table: SafeTensorFile, component: str, metric: str) -> float | None:
    model, layer, matrix, feature = parse_component(component)
    values = table.tensor(f"{model}.L{layer:02d}.{matrix}::{metric}")
    value = float(values[feature])
    return value if math.isfinite(value) else None


def component_record(
    key: str,
    importance: SafeTensorFile,
    representative_set: set[str],
) -> dict[str, Any]:
    model, layer, matrix, _ = parse_component(key)
    return {
        "id": key,
        "modelId": model,
        "gateId": component_gate(key),
        "layer": layer,
        "matrix": MATRIX_LABEL[matrix],
        "beta": importance_value(importance, key, "beta"),
        "rho": importance_value(importance, key, "rho"),
        "representative": key in representative_set,
    }


def load_activation_examples(
    context_root: Path,
    frontend_gates: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the natural-firing examples needed by the browser bundle."""
    manifest_path = context_root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["checkpoint_step"] != 61036 or not manifest["natural_gate_only"]:
        raise ValueError("activation-context extraction is not the final natural-gate run")
    if manifest["selected_split"] != "test":
        raise ValueError("activation-context extraction must use the frozen test split")
    if manifest.get("protocol_sha256") != QUICK_CONTEXT_PROTOCOL_SHA256:
        raise ValueError("activation-context extraction does not use the frozen Gemma display protocol")

    source = manifest["source_model_provenance"][QUICK_CONTEXT_DISPLAY_MODEL_KEY]
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer_snapshot = snapshot_download(
        repo_id=source["model_id"],
        revision=source["tokenizer_revision"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)

    requested = {
        site.lower(): {int(gate.split(":")[1]) for gate in frontend_gates if gate.startswith(site)}
        for site in ("PRE", "MID")
    }
    rows: list[dict[str, Any]] = []
    context_hashes: dict[str, str] = {}
    for site in ("pre", "mid"):
        path = context_root / f"{site}_feature_contexts.jsonl"
        actual_hash = sha256_file(path)
        expected_hash = manifest["site_reports"][site]["feature_contexts_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"{site} feature contexts failed SHA-256 verification")
        context_hashes[site] = actual_hash
        remaining = set(requested[site])
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                feature = int(row["feature"])
                if feature not in remaining:
                    continue
                gate_id = gate_key(site, feature)
                examples = []
                for context in row["contexts"][:4]:
                    token_ids = [int(value) for value in context["token_ids"]]
                    activations = [float(value) for value in context["activations"]]
                    firings = [bool(value) for value in context["firings"]]
                    if (
                        not activations
                        or len(activations) > 65
                        or len(activations) != len(firings)
                        or len(token_ids) != len(activations)
                    ):
                        raise ValueError(f"unexpected natural-context window for {gate_id}")
                    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
                    if isinstance(token_pieces, str) or len(token_pieces) != len(token_ids):
                        raise ValueError(f"tokenizer pieces lost position alignment for {gate_id}")
                    token_texts = [
                        tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        for token_id in token_ids
                    ]
                    center_candidates = [
                        position
                        for position, (activation, firing) in enumerate(
                            zip(activations, firings)  # noqa: B905 - lengths checked above
                        )
                        if firing
                        and math.isclose(
                            activation,
                            float(context["activation"]),
                            rel_tol=1e-6,
                            abs_tol=1e-8,
                        )
                    ]
                    if len(center_candidates) != 1:
                        raise ValueError(f"saved center firing is inconsistent for {gate_id}")
                    examples.append(
                        {
                            "activation": float(context["activation"]),
                            "activations": activations,
                            "anchorId": str(context["anchor_id"]),
                            "anchorIndex": int(context["anchor_index"]),
                            "centerPosition": center_candidates[0],
                            "context": str(context["context"]),
                            "documentId": str(context["document_id"]),
                            "firings": firings,
                            "sourceGroup": str(context["source_group"]),
                            "tokenIds": token_ids,
                            "tokenPieces": [str(value) for value in token_pieces],
                            "tokenTexts": token_texts,
                        }
                    )
                rows.append(
                    {
                        "gateId": gate_id,
                        "site": site.upper(),
                        "featureIndex": feature,
                        "examples": examples,
                    }
                )
                remaining.remove(feature)
                if not remaining:
                    break
        if remaining:
            raise ValueError(f"activation contexts missing features for {site}: {sorted(remaining)}")

    provenance = {
        "artifactRoot": str(context_root),
        "checkpointStep": manifest["checkpoint_step"],
        "checkpointManifestSha256": manifest["checkpoint_manifest_sha256"],
        "evaluationGitSha": manifest["evaluation_git_sha"],
        "experimentFamily": manifest["experiment_family"],
        "extractionManifestSha256": sha256_file(manifest_path),
        "featureContextSha256": context_hashes,
        "naturalGateOnly": True,
        "selectedSplit": manifest["selected_split"],
        "splitUnit": manifest["split_unit"],
        "windowTokens": 65,
        "relationToGraph": "independent-natural-firing-extraction; examples are not graph anchors",
        "displayTokenizer": {
            "modelKey": QUICK_CONTEXT_DISPLAY_MODEL_KEY,
            "modelId": source["model_id"],
            "revision": source["tokenizer_revision"],
            "protocolSha256": manifest["protocol_sha256"],
            "axis": "saved token_ids, activations, and firings are position-aligned on the Gemma display-tokenizer axis",
        },
    }
    return sorted(rows, key=lambda row: (row["site"], row["featureIndex"])), provenance


def build_bundle(
    root: Path,
    activation_context_root: Path,
    gates_per_site: int,
    display_parent_gates: int,
) -> dict[str, Any]:
    importance_manifest = read_json(root / "importance/manifest.json")
    representatives_payload = read_json(root / "representatives/representatives.json")
    representatives = representatives_payload["representatives"]
    selection = read_json(root / "edges/selection.json")
    chosen_gates = choose_gates(representatives, selection, gates_per_site)

    shard_listing = read_json(root / "edges/shards.json")["shards"]
    shard_for_gate: dict[str, str] = {}
    shard_complete_hashes: dict[str, str] = {}
    for item in shard_listing:
        stage = item["stage"]
        shard_selection = read_json(root / stage / "selection.json")
        for gate in chosen_gates:
            if gate in shard_selection["gates"]:
                shard_for_gate[gate] = stage
        shard_complete_hashes[stage] = item["complete_sha256"]
    missing_shards = set(chosen_gates) - set(shard_for_gate)
    if missing_shards:
        raise ValueError(f"chosen gates missing edge shards: {sorted(missing_shards)}")

    importance = SafeTensorFile(root / "importance/importance.safetensors")
    profiles = ProfileStore(root)
    requested_parent_rows = [
        (selection["anchors"][gate][0], downstream)
        for gate in chosen_gates
        for model in MODEL_SPECS
        for downstream in representatives[gate][model]
    ]
    stored_parents = profiles.parent_rows(requested_parent_rows)
    k_par, tau = 10, 0.0
    representative_set = {
        component
        for gate_models in representatives.values()
        for components in gate_models.values()
        for component in components
    }

    all_components: set[str] = set()
    edges: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    parent_distributions: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []

    for gate in chosen_gates:
        anchor_id = selection["anchors"][gate][0]
        model_parents: dict[str, dict[str, list[dict[str, Any]]]] = {}
        model_scores: dict[str, dict[str, float]] = {}

        for model in MODEL_SPECS:
            downstream_components = representatives[gate][model]
            parents_by_downstream: dict[str, list[dict[str, Any]]] = {}
            for downstream in downstream_components:
                parents = stored_parents[(anchor_id, downstream)]
                parents_by_downstream[downstream] = parents
                all_components.add(downstream)
            model_parents[model] = parents_by_downstream
            distribution = profiles.distribution(gate, model, anchor_id)
            if distribution["representatives"] != len(downstream_components):
                raise ValueError(
                    f"profile representative count mismatch for {(gate, model, anchor_id)}"
                )
            model_scores[model] = distribution["values"] or {}
            parent_distributions.append(
                {
                    "gateId": gate_key(*_split_gate(gate)),
                    "modelId": model,
                    "anchorId": anchor_id,
                    "status": distribution["status"],
                    "totalStrength": distribution["totalStrength"],
                    "values": distribution["values"],
                }
            )

        pooled: dict[str, float] = defaultdict(float)
        valid_models = 0
        for distribution_values in model_scores.values():
            if not distribution_values:
                continue
            valid_models += 1
            for parent_gate, value in distribution_values.items():
                pooled[parent_gate] += value
        if valid_models:
            for parent_gate in list(pooled):
                pooled[parent_gate] /= valid_models
        displayed_parent_gates = [
            key
            for key, _ in sorted(pooled.items(), key=lambda item: (-item[1], item[0]))[
                :display_parent_gates
            ]
        ]
        displayed_set = set(displayed_parent_gates)

        for model, parents_by_downstream in model_parents.items():
            for downstream, parents in parents_by_downstream.items():
                for parent in parents:
                    if parent["parentGate"] not in displayed_set:
                        continue
                    source = parent["component"]
                    all_components.add(source)
                    edges.append(
                        {
                            "id": f"{anchor_id}|{source}>{downstream}",
                            "modelId": model,
                            "anchorId": anchor_id,
                            "source": source,
                            "target": downstream,
                            "signedEffect": parent["score"],
                            "normalizedMagnitude": parent["weight"],
                            "upstreamCoefficient": None,
                            "gradient": None,
                            "normalization": "w=|E|*||u_target||2/sigma_target",
                            "evidence": "measured-taylor",
                            "parentRank": parent["rank"],
                        }
                    )

        frontend_gate = gate_key(*_split_gate(gate))
        anchors.append(
            {
                "id": anchor_id,
                "label": f"{frontend_gate} · natural anchor · {anchor_id[:10]}",
                "text": (
                    "Canonical text was not retained in the immutable edge artifact. "
                    f"Measured aligned anchor id: {anchor_id}"
                ),
                "start": 0,
                "end": 0,
                "gateActivations": {frontend_gate: None},
                "firingGateIds": [frontend_gate],
                "modelTokens": {model: [] for model in MODEL_SPECS},
            }
        )
        scenarios.append(
            {
                "gateId": frontend_gate,
                "rawGateId": gate,
                "anchorId": anchor_id,
                "displayParentGateIds": displayed_parent_gates,
                "edgeShard": shard_for_gate[gate],
            }
        )

    component_rows = [
        component_record(key, importance, representative_set)
        for key in sorted(all_components, key=component_sort_key)
    ]
    frontend_gates = sorted(
        {row["gateId"] for row in component_rows},
        key=lambda key: (0 if key.startswith("PRE:") else 1, int(key.split(":")[1])),
    )
    activation_examples, activation_provenance = load_activation_examples(
        activation_context_root, frontend_gates
    )
    first = scenarios[0]
    default_gate_ids = [first["gateId"], *first["displayParentGateIds"]]
    default_gate_ids = list(dict.fromkeys(default_gate_ids))

    models = []
    backend_models = importance_manifest["backend"]["models"]
    for model, spec in MODEL_SPECS.items():
        source = backend_models[model]
        checkpoint = f"{source['model_id']}@{source['revision']}"
        tokenizer = f"{source['model_id']}@{source['tokenizer_revision']}"
        models.append(
            {
                "id": model,
                "label": spec["label"],
                "checkpoint": checkpoint,
                "tokenizer": tokenizer,
                "selectedLayers": spec["layers"],
            }
        )

    source_hashes = {
        "importanceComplete": sha256_file(root / "importance/COMPLETE"),
        "representativesComplete": sha256_file(root / "representatives/COMPLETE"),
        "edgesComplete": sha256_file(root / "edges/COMPLETE"),
        "profilesComplete": profiles.complete_hash,
        "profileArtifacts": profiles.source_hashes,
        "posthocConfig": importance_manifest["config_sha256"],
        "datasetManifest": importance_manifest["backend"]["dataset_manifest_sha256"],
        "selectedShardComplete": {
            stage: shard_complete_hashes[stage]
            for stage in sorted(set(shard_for_gate.values()))
        },
    }
    payload = {
        "manifest": {
            "schemaVersion": "1.0",
            "datasetId": "aspd-v3.1-500m-measured-workbench-v1",
            "label": "Measured bounded graph from ASPD v3.1 500M post-hoc validation",
            "synthetic": False,
            "sourceRevision": profiles.manifest["runtime"]["git_sha"],
            "trainingBranch": "aspd-v3.1-main500m-checkpoint-timeout-fix",
            "trainingConfig": "configs/multimodel/aspd_v3/main500m_preparation_v1.yaml",
            "specification": {
                "status": "local-pdf",
                "path": "docs/Universal_Mechanism.pdf",
                "sha256": "a963c3616a498ca1b299fad6f033a94a2ca27a7ac8029de6dc5c7e67ebf31f27",
            },
            "dataset": {
                "id": "monology/pile-uncopyrighted",
                "revision": DATASET_REVISION,
                "splitPolicy": "validation; natural gates; edge pools disjoint after importance pools",
            },
            "screening": {
                "metric": "beta",
                "kRep": 3,
                "tauRep": 0.07,
                "method": "frozen post-hoc v1 representative screening",
            },
            "edgeScoring": {
                "method": "measured signed first-order Taylor attribution; frozen natural gates",
                "normalization": "w=|E|*||u_target||2/sigma_target",
            },
            "display": {
                "defaultGateIds": default_gate_ids,
                "defaultSpotlight": first["gateId"],
                "defaultAnchorId": first["anchorId"],
                "defaultEdgeThreshold": 0.0,
            },
            "export": {
                "artifactRoot": str(root),
                "checkpointStep": importance_manifest["backend"]["checkpoint_step"],
                "selectionRule": (
                    f"lowest feature-index eligible gates, {gates_per_site} per PRE/MID; "
                    "first stored seeded-random natural anchor per gate"
                ),
                "graphBound": (
                    f"official K_par={k_par}, tau={tau}; retain graph edges from the top "
                    f"{display_parent_gates} parent gates by mean pi across models"
                ),
                "selectedScenarios": scenarios,
                "sourceHashes": source_hashes,
                "parentBackend": {
                    "stage": "profiles",
                    "status": profiles.manifest["status"],
                    "sourceRevision": profiles.manifest["runtime"]["git_sha"],
                    "slurmJobId": profiles.manifest["runtime"]["slurm_job_id"],
                    "parentRows": profiles.manifest["parent_rows"],
                    "distributionSource": "profiles/parent_distributions.safetensors",
                    "parentSetSource": "profiles/parents_per_anchor.safetensors",
                    "rule": profiles.manifest["rule"],
                },
                "activationExamples": activation_provenance,
                "verification": "uses immutable COMPLETE hash chain, including parent profiles; selected tensors read-only",
                "floatPrecision": "12 decimal places for cross-runtime deterministic JSON",
                "limitations": [
                    "canonical context text and continuous g^s were not stored in the edge artifact",
                    "displayed natural-firing contexts come from a separate test-split extraction and are not the graph anchors",
                    "four saved contexts per gate support semantic inspection, not a validated feature interpretation",
                    "graph edges are first-order Taylor measurements, not general causal validation",
                    "parent sets and S/pi are formulation objects; profile means over anchors are descriptive additions",
                    "the bounded bundle is a deterministic visualization subset, not a population estimate",
                ],
            },
        },
        "models": models,
        "gates": [
            {
                "id": key,
                "stage": key.split(":")[0],
                "featureIndex": int(key.split(":")[1]),
                "label": key,
            }
            for key in frontend_gates
        ],
        "anchors": anchors,
        "components": component_rows,
        "edges": sorted(edges, key=lambda row: row["id"]),
        "parentDistributions": sorted(
            parent_distributions,
            key=lambda row: (row["anchorId"], row["gateId"], row["modelId"]),
        ),
        "activationExamples": activation_examples,
    }
    canonical = canonicalize_numbers(payload)
    validate_bundle(canonical)
    return canonical


def _split_gate(gate: str) -> tuple[str, int]:
    site, feature = gate.split("/", 1)
    return site, int(feature)


def validate_bundle(payload: dict[str, Any]) -> None:
    if payload["manifest"]["synthetic"] is not False:
        raise ValueError("measured bundle must set synthetic=false")
    component_by_id = {row["id"]: row for row in payload["components"]}
    anchor_ids = {row["id"] for row in payload["anchors"]}
    if len(component_by_id) != len(payload["components"]):
        raise ValueError("duplicate component ids")
    edge_ids: set[str] = set()
    for edge in payload["edges"]:
        if edge["id"] in edge_ids:
            raise ValueError(f"duplicate edge id {edge['id']}")
        edge_ids.add(edge["id"])
        source, target = component_by_id[edge["source"]], component_by_id[edge["target"]]
        if source["modelId"] != target["modelId"] or edge["modelId"] != source["modelId"]:
            raise ValueError(f"cross-model edge {edge['id']}")
        if edge["anchorId"] not in anchor_ids:
            raise ValueError(f"unknown anchor on edge {edge['id']}")
        if edge["evidence"] != "measured-taylor":
            raise ValueError(f"unexpected evidence label on {edge['id']}")
    if not payload["edges"]:
        raise ValueError("bundle contains no measured edges")
    gate_ids = {row["id"] for row in payload["gates"]}
    example_gate_ids = {row["gateId"] for row in payload["activationExamples"]}
    if example_gate_ids != gate_ids:
        raise ValueError("activation examples must cover every exported gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--activation-context-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates-per-site", type=int, default=2)
    parser.add_argument("--display-parent-gates", type=int, default=6)
    args = parser.parse_args()
    bundle = build_bundle(
        args.artifact_root.resolve(),
        args.activation_context_root.resolve(),
        args.gates_per_site,
        args.display_parent_gates,
    )
    encoded = (json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "gates": len(bundle["gates"]),
                "anchors": len(bundle["anchors"]),
                "components": len(bundle["components"]),
                "edges": len(bundle["edges"]),
                "parent_distributions": len(bundle["parentDistributions"]),
                "activation_example_gates": len(bundle["activationExamples"]),
                "activation_examples": sum(len(row["examples"]) for row in bundle["activationExamples"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

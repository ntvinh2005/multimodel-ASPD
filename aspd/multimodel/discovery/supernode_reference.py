"""Load anchor-indexed native reference states without mixing model coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aspd.multimodel.posthoc.artifacts import sha256_file


def load_native_reference(directory: Path, model, anchor_ids, measurement_contract, variant):
    """Read sealed [layer, anchor, hidden] states and return raw concatenation."""
    manifest_file = directory / "COMPLETE.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest["measurementContract"] != measurement_contract:
        raise ValueError("Hidden-state and contribution contracts differ")
    record = manifest["models"][model]
    if record["location"] != "decoder-block-output":
        raise ValueError("Native hidden location is not a decoder-block output")
    path = directory / f"{model}.npz"
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Unverified native reference for {model}")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["anchors"].tolist()
        lookup = {key: i for i, key in enumerate(ids)}
        layers = archive["layers"].tolist()
        if len(set(ids)) != len(ids) or len(set(layers)) != len(layers) or sorted(layers) != layers:
            raise ValueError("Expected unique anchors and ordered native layers")
        if list(record["allLayers"]) != layers:
            raise ValueError("Hidden-state shard does not cover the manifest layer list")
        selected = layers if variant == "all-hidden" else record["selectedLayers"]
        if variant == "selected-hidden" and len(selected) != 8:
            raise ValueError("Expected the eight approved ASPD layers")
        positions = [lookup[key] for key in anchor_ids]
        hidden = archive["hidden"]
        if hidden.ndim != 3 or hidden.shape[:2] != (len(layers), len(ids)):
            raise ValueError("Hidden-state shape differs from its index")
        vectors = np.concatenate(
            [
                np.asarray(hidden[layers.index(layer), positions], dtype=np.float32)
                for layer in selected
            ],
            axis=1,
        )
        if not np.isfinite(vectors).all():
            raise ValueError("Invalid native hidden state")
    return vectors, {
        "layers": selected,
        "location": record["location"],
        "dimensions": vectors.shape[1],
        "normalization": "raw concatenation",
        "manifestSha256": sha256_file(manifest_file),
    }

"""Recover exact Natural Validation prefixes and sparse downstream coefficients."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from aspd.multimodel.discovery.index import CHECKPOINT, decode, encode, load, sha

MODEL_ORDER = ("gemma", "smol", "granite")
DATASET = "monology/pile-uncopyrighted"
REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"


def sealed(root: Path, name: str):
    complete = load(root / "COMPLETE")
    if complete["status"] != "COMPLETE":
        raise ValueError(f"Incomplete Natural Validation archive: {root}")
    return load(root / name, complete["sha256"][name])


def inventory(root: Path):
    shards = sorted(root.glob("edges-shard-*-of-*"))
    expected = sealed(root / "edges", "manifest.json")
    if len(shards) != expected["shards"]:
        raise ValueError("Edge shard inventory differs from the merged archive")
    anchors = set()
    pools = set()
    for shard in shards:
        manifest = sealed(shard, "manifest.json")
        if manifest["config_fingerprint"] != expected["config_fingerprint"]:
            raise ValueError(f"Configuration mismatch: {shard}")
        if manifest["backend"]["split"] != "validation":
            raise ValueError("Natural Validation is the only supported source")
        selection = sealed(shard, "selection.json")
        start = manifest["first_pool_index"]
        pools.update(range(start, start + selection["pools_used"]))
        metadata = sealed(shard, "edge_records.json")
        anchors.update(metadata["anchor_ids"])
    if not pools or not anchors:
        raise ValueError("No Natural Validation measurements")
    return shards, anchors, pools


class FirstPileShard:
    """Replay the leading bytes of the pinned train/00 stream in its original order."""

    def __init__(self, path: Path):
        self.path = path
        self.position = 0

    def state_dict(self):
        return {"row": self.position}

    def load_state_dict(self, state):
        self.position = state["row"]

    def __iter__(self):
        import zstandard

        start = self.position
        with self.path.open("rb") as source:
            stream = zstandard.ZstdDecompressor().stream_reader(source, read_across_frames=True)
            with io.BufferedReader(stream) as reader:
                for i, line in enumerate(reader):
                    if i < start:
                        continue
                    if not line.endswith(b"\n"):
                        break
                    self.position = i + 1
                    yield json.loads(line)


def recover(archive: Path, pile: Path, tokenizer_cache: Path, training_config: Path, output):
    """Recover anchor IDs and pool membership without loading source model weights."""
    import yaml
    from transformers import AutoTokenizer

    from aspd.multimodel.v3.data import HashSplitPolicy
    from aspd.multimodel.v3.diagnostic_stream import DiagnosticAlignedAnchorStream

    shards, wanted, pools = inventory(archive)
    manifest = sealed(shards[0], "manifest.json")
    backend = manifest["backend"]
    if sha(Path(backend["checkpoint"]) / "manifest.json") != CHECKPOINT:
        raise ValueError("Checkpoint manifest mismatch")
    if sha(training_config) != backend["config_file_sha256"]:
        raise ValueError("Original training configuration hash mismatch")
    config = yaml.safe_load(training_config.read_text())["data"]
    if (config["dataset_id"], config["dataset_revision"]) != (DATASET, REVISION):
        raise ValueError("Unexpected Natural Validation corpus")
    revisions = {key: backend["models"][key]["tokenizer_revision"] for key in MODEL_ORDER}
    paths = {
        key: tokenizer_cache
        / ("models--" + backend["models"][key]["model_id"].replace("/", "--"))
        / "snapshots"
        / revisions[key]
        for key in MODEL_ORDER
    }
    tokenizers = {
        key: AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        for key, path in paths.items()
    }
    stream = DiagnosticAlignedAnchorStream(
        dataset=FirstPileShard(pile),
        dataset_id=DATASET,
        dataset_revision=REVISION,
        tokenizers=tokenizers,
        tokenizer_revisions=revisions,
        selected_split="validation",
        dedup_path=output / "replay.sqlite",
        sequence_length=config["sequence_length"],
        split_policy=HashSplitPolicy(
            modulus=config["split_modulus"],
            seed=config["split_seed"],
            validation_buckets=frozenset(config["validation_buckets"]),
            test_buckets=frozenset(config["test_buckets"]),
        ),
        model_order=MODEL_ORDER,
    )
    records = 0
    matched = set()
    with (output / "anchors.jsonl").open("x") as handle:
        try:
            for pool_index in range(max(pools) + 1):
                pool = stream.next_pool(pool_anchors=backend["pool_anchors"])
                if pool_index not in pools:
                    continue
                row = 0
                for document in pool.source_observations:
                    text = document["canonical_text"]
                    for boundary in document["anchor_character_boundaries"]:
                        anchor = pool.anchor_ids[row]
                        positions = {
                            key: pool.models[key].anchor_token_indices[row] for key in MODEL_ORDER
                        }
                        if anchor in wanted:
                            prefix = text[:boundary]
                            prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()
                            document_id, group = document["document_id"], document["source_group"]
                            expected = hashlib.sha256(
                                f"{group}\0{document_id}\0{boundary}\0{prefix_hash}".encode()
                            ).hexdigest()
                            if expected != anchor or anchor in matched:
                                raise ValueError(f"Repeated or mismatched anchor {anchor}")
                            matched.add(anchor)
                            ids = {}
                            for key in MODEL_ORDER:
                                model = pool.models[key]
                                ids[key] = model.input_ids[model.anchor_document_indices[row]][
                                    positions[key]
                                ]
                            record = dict(
                                anchor=anchor,
                                dataset="natural-validation",
                                document=document_id,
                                source=group,
                                pool=pool_index,
                                boundary=boundary,
                                positions=positions,
                                tokenIds=ids,
                                prefix=prefix,
                                prefixSha256=prefix_hash,
                            )
                            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                            records += 1
                        row += 1
                if row != len(pool.anchor_ids):
                    raise ValueError("Source observation order differs from the exact pool")
                print(f"validation pool {pool_index}: {records}/{len(wanted)} anchors", flush=True)
        finally:
            stream.close()
    if matched != wanted:
        raise ValueError(f"Exact replay missed {len(wanted - matched)} measured anchors")
    return {
        "checkpoint": CHECKPOINT,
        "dataset": DATASET,
        "revision": REVISION,
        "stream": "train/00.jsonl.zst",
        "split": "validation",
        "pools": sorted(pools),
        "recoveredAnchors": records,
        "documentsRead": stream.documents_read,
        "inputPrefixSha256": sha(pile),
        "anchorsSha256": sha(output / "anchors.jsonl"),
        "configSha256": sha(training_config),
        "models": backend["models"],
        "tokenizerHashes": {key: sha(path / "tokenizer.json") for key, path in paths.items()},
        "edgeCompleteSha256": sha(archive / "edges" / "COMPLETE"),
    }


def measurements(archive: Path, anchor_rows: list[dict]):
    """Yield component-level e and |eu|/sigma from sealed scalar columns."""
    shards, _, _ = inventory(archive)
    sigma = sealed(archive / "importance", "importance_meta.json")
    by_anchor = {row["anchor"]: i for i, row in enumerate(anchor_rows)}
    for shard in shards:
        meta = sealed(shard, "edge_records.json")
        norms = sealed(shard, "selection.json")["downstream_write_norms"]
        component = np.array([encode(key) for key in meta["downstream_keys"]], dtype=np.int32)
        scales = np.array(
            [norms[decode(key)] / sigma[decode(key).split("/")[0]]["sigma"] for key in component]
        )
        if not np.all(np.isfinite(scales) & (scales >= 0)):
            raise ValueError(f"Invalid archived rank-one scales: {shard}")
        # Only read the three scalar columns. The original edge files are memory-mapped.
        with safe_open(shard / "edges.safetensors", framework="numpy") as data:
            a = data.get_tensor("record_anchor")
            d = data.get_tensor("record_downstream")
            e = data.get_tensor("record_coefficient")
            if not (len(a) == len(d) == len(e) == meta["records"] and np.isfinite(e).all()):
                raise ValueError(f"Invalid recorded coefficients: {shard}")
            lookup = np.array([by_anchor.get(key, -1) for key in meta["anchor_ids"]])
            ix = lookup[a]
            keep = ix >= 0
            yield component[d[keep]], ix[keep], e[keep], np.abs(e[keep]) * scales[d[keep]]

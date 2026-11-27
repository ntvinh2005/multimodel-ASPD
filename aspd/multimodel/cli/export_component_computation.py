"""Read NVX-025 component scalars and factors from the original frozen artifacts."""

import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader

from aspd.multimodel.posthoc.computation import affine_up_removal, largest_terms
from aspd.multimodel.posthoc.ids import ComponentId

MODELS = ("gemma", "smol", "granite")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def checked_file(root, name, *, hash_contents=True):
    marker = read(root / "COMPLETE")
    digest = marker["sha256"][name]
    path = root / name
    if marker["status"] != "COMPLETE" or (hash_contents and sha(path) != digest):
        raise ValueError(f"Archive integrity check failed: {path}")
    return path, digest


def factor(reader, metadata, name, feature):
    meta = metadata.state_dict_metadata[name]
    chunks = meta.chunks
    if len(chunks) != 1 or tuple(chunks[0].offsets) != (0, 0) or chunks[0].sizes != meta.size:
        raise ValueError(f"Factor must be in a single contiguous DCP tensor: {name}")
    storage = next(row for key, row in metadata.storage_data.items() if key.fqn == name)
    path = reader / storage.relative_path
    if storage.offset or storage.length != path.stat().st_size:
        raise ValueError(f"Cannot mmap a packed DCP object: {name}")
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if data.shape != meta.size or data.dtype != torch.float32:
        raise ValueError(f"Unexpected factor tensor at {name}")
    return data[feature].clone(), {"file": storage.relative_path, "key": name}


def finite(value):
    return value if math.isfinite(value) else None


def export(args):
    checkpoint_sha = sha(args.checkpoint / "manifest.json")
    checkpoint = read(args.checkpoint / "manifest.json")
    evidence = read(args.evidence)
    if (read(args.checkpoint / "COMPLETE")["manifest_sha256"] != checkpoint_sha
            or checkpoint_sha != evidence["provenance"]["checkpoint_manifest_sha256"]
            or checkpoint["step"] != 61036):
        raise ValueError("Frozen checkpoint identity differs from candidate evidence")
    case = next(row for row in evidence["candidates"] if row["id"] == "NVX-025")
    extension = next(row for row in read(args.extensions)["raw_checked_intra_gate_witnesses"]
                     if row["candidate"] == case["id"])
    anchors = set(case["fixed_common"])
    if anchors != set(extension["anchor_ids"]):
        raise ValueError("The archived extension and original motif have different coverage")
    ids = sorted({ComponentId.from_key(key) for row in extension["models"].values()
                  for key in [row["source"], row["target"], *row["original_nodes"]]})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "factors").mkdir()
    validation = args.validation
    manifest_path, _ = checked_file(validation / "importance", "manifest.json")
    backend = read(manifest_path)["backend"]
    if (backend["config_file_sha256"] != checkpoint["config_file_sha256"]
            or backend["dataset_manifest_sha256"] != checkpoint["dataset_manifest_sha256"]
            or set(backend["models"]) != set(MODELS)):
        raise ValueError("Importance provenance disagrees with the checkpoint")
    meta_path, _ = checked_file(validation / "importance", "importance_meta.json")
    scale = read(meta_path)
    importance, importance_hash = checked_file(validation / "importance", "importance.safetensors")
    rep_path, _ = checked_file(validation / "representatives", "representatives.json")
    rep = read(rep_path)
    shard_catalog, catalog_hash = checked_file(validation / "edges", "shards.json")
    if catalog_hash != evidence["provenance"]["artifact_hashes"]["edges"]["shards.json"]:
        raise ValueError("Raw shard catalog differs from candidate provenance")
    shard_digests = {row["stage"]: row["complete_sha256"] for row in read(shard_catalog)["shards"]}
    static = {}
    native_configs = {}
    with safe_open(importance, framework="pt") as imp:
        for rank, model in enumerate(MODELS):
            native_config = None
            if args.hf_cache:
                pinned = backend["models"][model]
                config_path = args.hf_cache / ("models--" + pinned["model_id"].replace("/", "--")) / "snapshots" / pinned["revision"] / "config.json"
                native_config = read(config_path)
                native_configs[model] = sha(config_path)
            location = args.checkpoint / f"rank-{rank:05d}"
            sidecar = read(location / "sidecar.json")
            if sidecar["model_key"] != model or sidecar["step"] != 61036:
                raise ValueError("Rank ownership differs from the checkpoint")
            path = location / "dcp"
            metadata = FileSystemReader(path).read_metadata()
            for i, component in enumerate(ids):
                if component.model != model:
                    continue
                fqn = "model.matrix_bank.components." + component.matrix.key.replace(".", "__dot__")
                u, pos_u = factor(path, metadata, fqn + ".U", component.feature)
                v, pos_v = factor(path, metadata, fqn + ".V", component.feature)
                filename = f"factors/{i:02d}.f32"
                binary = args.output / filename
                binary.write_bytes(torch.cat([u, v]).numpy().astype("<f4", copy=False).tobytes())
                matrix = component.matrix.key
                values = {key: finite(float(imp.get_slice(f"{matrix}::{key}")[component.feature:component.feature + 1][0]))
                          for key in ("beta", "rho", "fire_count")}
                entry = {
                    "uNorm": float(u.double().norm()), "vNorm": float(v.double().norm()),
                    "dOut": u.numel(), "dIn": v.numel(), "readWeights": largest_terms(v), "writeWeights": largest_terms(u),
                    "sigma": scale[matrix]["sigma"], **values,
                    "representative": component.key in rep["representatives"].get(component.gate.key, {}).get(model, []),
                    "factors": {"path": filename, "sha256": sha(binary), "dtype": "float32-le", "layout": "u,v"},
                    "checkpointSlices": {"rank": rank, "u": pos_u, "v": pos_v},
                }
                if native_config:
                    entry["nativeActivation"] = native_config.get("hidden_activation") or native_config.get("hidden_act")
                    matrix_name = component.matrix.matrix
                    if matrix_name in {"q", "k", "v", "o"}:
                        query_heads = native_config["num_attention_heads"]
                        count = native_config["num_key_value_heads"] if matrix_name in {"k", "v"} else query_heads
                        width = native_config.get("head_dim", native_config["hidden_size"] // query_heads)
                        factor_vector = v if matrix_name == "o" else u
                        if factor_vector.numel() != count * width:
                            raise ValueError(f"Native head layout disagrees for {component.key}")
                        entry["heads"] = {"count": count, "width": width, "queryHeads": query_heads,
                                          "vector": "v" if matrix_name == "o" else "u",
                                          "norms": factor_vector.reshape(count, width).double().norm(dim=1).tolist()}
                static[component.key] = entry
    cases = {anchor: {"anchor": anchor, "rows": {}, "edges": [], "local": []} for anchor in sorted(anchors)}
    notebooks = {}
    with ExitStack() as stack:
        def archive(shard):
            if shard not in notebooks:
                directory = validation / shard
                index_path, _ = checked_file(directory, "edge_records.json")
                manifest_path, _ = checked_file(directory, "manifest.json")
                source, _ = checked_file(directory, "edges.safetensors", hash_contents=False)
                if read(manifest_path)["backend"] != backend or sha(directory / "COMPLETE") != shard_digests[shard]:
                    raise ValueError(f"Edge provenance disagrees in {shard}")
                notebooks[shard] = (read(index_path), stack.enter_context(safe_open(source, framework="pt")))
            return notebooks[shard]

        for model in MODELS:
            original = case["models"][model]
            pairs = list(zip(original["nodes"][::2], original["nodes"][1::2], strict=True))
            up = extension["models"][model]["source"]
            for item in original["fixed_ranked_edges"]:
                anchor = item["anchor"]
                if anchor not in cases:
                    continue
                result = cases[anchor]
                for j, ((source, target), raw) in enumerate(zip(pairs, item["raw"], strict=True)):
                    index, tensors = archive(raw["shard"])
                    record, offset = raw["record_index"], raw["raw_flat_index"]
                    field = lambda key, at: tensors.get_slice(key)[at:at + 1][0].item()
                    target_key = index["downstream_keys"][field("record_downstream", record)]
                    recorded_anchor = index["anchor_ids"][field("record_anchor", record)]
                    if recorded_anchor != anchor or target_key != target:
                        raise ValueError("Stored graph endpoint or anchor changed")
                    if (field("score", offset) != item["E"][j]
                            or index["matrix_keys"][field("upstream_matrix", offset)] + "/" + str(field("upstream_feature", offset)) != source):
                        raise ValueError("Raw edge value or identity changed")
                    coefficient = field("record_coefficient", record)
                    if coefficient != raw["coefficient"]:
                        raise ValueError("Raw coefficient differs from the published archive")
                    result["rows"][target] = {"coefficient": coefficient, "reading": coefficient, "support": True,
                                               "raw": {"shard": raw["shard"], "record": record}}
                    result["edges"].append({"source": source, "target": target, "E": item["E"][j], "w": item["w"][j]})
                    if j != 0:
                        continue
                    start, stop = (field("record_" + field_name, record) for field_name in ("start", "stop"))
                    component = ComponentId.from_key(up)
                    matrix = index["matrix_keys"].index(component.matrix.key)
                    matches = ((tensors.get_slice("upstream_matrix")[start:stop] == matrix)
                               & (tensors.get_slice("upstream_feature")[start:stop] == component.feature)).nonzero().flatten()
                    if len(matches) != 1:
                        raise ValueError("No exact Up extension in the frozen raw row")
                    effect = field("score", start + int(matches[0]))
                    stored = extension["models"][model]
                    if anchor == extension["witness_anchor"] and effect != stored["E"]:
                        raise ValueError("The Up extension witness differs from the raw archive")
                    weight = abs(effect) * static[target]["uNorm"] / static[target]["sigma"]
                    result["edges"].append({"source": up, "target": target, "E": effect, "w": weight})
                    result["local"].append({"source": up, "target": target, "nativeReplay": False,
                                            **affine_up_removal(component, ComponentId.from_key(target), coefficient, effect)})
    if any(len(row["local"]) != 3 for row in cases.values()):
        raise ValueError("Expected all three same-layer Up effects at each archived anchor")
    artifact = {
        "schemaVersion": 1, "candidate": case["id"], "checkpoint": checkpoint_sha,
        "provenance": {"evidenceSha256": sha(args.evidence), "extensionsSha256": sha(args.extensions),
                       "importanceSha256": importance_hash, "models": backend["models"], "nativeConfigSha256": native_configs,
                       "rawShardSha256": {name: read(validation / name / "COMPLETE")["sha256"]["edges.safetensors"]
                                          for name in notebooks}},
        "static": static, "anchors": cases,
    }
    with (args.output / "index.json").open("x") as handle:
        json.dump(artifact, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "validation", "evidence", "extensions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, help="Local Hugging Face hub cache containing the pinned config.json files")
    args = parser.parse_args()
    destination = args.output
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent, prefix=".component-computation-") as stage:
        args.output = Path(stage) / "output"
        export(args)
        args.output.rename(destination)


if __name__ == "__main__":
    main()

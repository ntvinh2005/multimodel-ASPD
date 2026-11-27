"""Collect native ASPD component coefficients on untouched Natural Validation pools."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import resource
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from aspd.multimodel.posthoc.artifacts import (
    require_complete,
    runtime_provenance,
    sha256_file,
    write_json,
)
from aspd.multimodel.posthoc.ids import ComponentId, GateId
from aspd.multimodel.posthoc.importance import ImportanceTable
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend

MODELS = ("gemma", "smol", "granite")
CHECKPOINT = "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969"
FIRST_POOL = 75
WAVES = (4, 64, 128, 192)
SEED = 28491
REFERENCE_GATES = ("mid/90", "mid/4856", "mid/1117", "mid/7417", "pre/1529")


def cohort(archive: Path):
    importance = require_complete(archive, "importance")
    reps = require_complete(archive, "representatives")
    table = ImportanceTable.load(importance)
    candidates = json.loads((reps / "representatives.json").read_text())["representatives"]
    rng = random.Random(SEED)
    selected = set(REFERENCE_GATES)
    strata = {}
    for site in ("pre", "mid"):
        counts = table.gate_fire_counts(site)
        eligible = sorted(
            (int(counts[g.feature]), key)
            for key, models in candidates.items()
            if (g := GateId.from_key(key)).site == site and all(models[m] for m in MODELS)
        )
        for quartile in range(4):
            pool = eligible[quartile * len(eligible) // 4 : (quartile + 1) * len(eligible) // 4]
            picked = rng.sample(pool, min(24, len(pool)))
            strata[f"{site}/{quartile}"] = [{"gate": key, "referenceFires": n} for n, key in picked]
            selected.update(key for _, key in picked)
    components = sorted(
        {
            ComponentId.from_key(key)
            for gate in selected
            for keys in candidates[gate].values()
            for key in keys
        }
    )
    rows = []
    for comp in components:
        stats = table[comp.matrix]
        if not math.isfinite(stats.sigma) or stats.sigma <= 0:
            raise ValueError(f"Invalid normalization scale for {comp.key}")
        rows.append(
            {
                "id": comp.key,
                "model": comp.model,
                "gate": comp.gate.key,
                "sigma": stats.sigma,
                "beta": float(stats.beta[comp.feature]),
                "referenceFires": int(stats.fire_count[comp.feature]),
            }
        )
    return rows, strata


def gated_coefficients(x, factors, support):
    """Apply the frozen binary support, including measured zeros for inactive cells."""
    readings = x.float() @ factors.float().T
    if readings.shape != support.shape:
        raise ValueError("Coefficient and gate axes differ")
    if not bool(torch.isfinite(readings).all()):
        raise ValueError("Nonfinite native component reading")
    return torch.where(support, readings, 0.0)


def token_offsets(tokenizer, text, expected, sequence_length):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=sequence_length,
    )
    if tuple(encoded["input_ids"]) != expected:
        raise ValueError("Offsets were obtained from different native tokens")
    return encoded["offset_mapping"]


def anchor_rows(aligned, tokenizers, sequence_length):
    anchors = []
    documents = []
    for index, obs in enumerate(aligned.source_observations):
        documents.append(
            {
                "id": obs["document_id"],
                "source": obs["source_group"],
                "text": obs["canonical_text"],
            }
        )
        offsets = {
            key: token_offsets(
                tokenizers[key], obs["canonical_text"], model.input_ids[index], sequence_length
            )
            for key, model in aligned.models.items()
        }
        for boundary in obs["anchor_character_boundaries"]:
            row = len(anchors)
            native = {}
            for key, model in aligned.models.items():
                if model.anchor_document_indices[row] != index:
                    raise ValueError("Aligned document and model gather map differ")
                position = model.anchor_token_indices[row]
                ids = model.input_ids[index]
                start, stop = offsets[key][position]
                if stop != boundary or not 0 <= start < stop:
                    raise ValueError("Native token offset does not end at the aligned boundary")
                native[key] = [position, ids[position], start, stop]
            if not 0 < boundary <= len(obs["canonical_text"]):
                raise ValueError("Invalid canonical boundary")
            anchors.append([aligned.anchor_ids[row], index, boundary, native])
    if len(anchors) != aligned.anchor_count:
        raise ValueError("Canonical anchor inventory differs from pool")
    return {"documents": documents, "anchors": anchors}


def measure(backend, aligned, components, gates):
    encoded = backend.encode_aligned_pool(aligned)
    binary = encoded.pool
    support = torch.stack([binary.gates[g.site][:, g.feature] for g in gates], dim=1)
    continuous = torch.stack(
        [encoded.continuous[g.site][:, g.feature].float() for g in gates], dim=1
    )
    if not torch.equal(support, continuous > 0):
        raise ValueError("Binary and continuous gate support disagree")
    gate_columns = {gate: i for i, gate in enumerate(gates)}
    by_matrix = defaultdict(list)
    for index, comp in enumerate(components):
        by_matrix[comp.matrix].append((index, comp.feature, gate_columns[comp.gate]))
    coefficients = np.empty((aligned.anchor_count, len(components)), dtype=np.float32)
    for model in MODELS:
        factors = backend.factors(model)
        specs = [spec for spec in backend.hook_specs(model) if spec.matrix in by_matrix]
        inputs = binary.models[model].to(backend.device)
        captured = capture_matrix_io(backend.source_model(model), inputs, specs)
        for matrix, (x, _) in captured.items():
            columns, features, gate_indices = zip(*by_matrix[matrix], strict=True)
            values = gated_coefficients(
                x, factors[matrix][1][list(features)], support[:, gate_indices].to(backend.device)
            )
            coefficients[:, columns] = values.cpu().numpy()
        del captured
    return coefficients, continuous.numpy()


def verified_json(root: Path, name: str, suffix: str = "COMPLETE"):
    seal = json.loads((root / suffix).read_text())
    for filename, expected in seal["sha256"].items():
        if sha256_file(root / filename) != expected:
            raise ValueError(f"Invalid artifact hash: {root / filename}")
    if name not in seal["sha256"]:
        raise ValueError(f"Missing sealed artifact {root / name}")
    return json.loads((root / name).read_text())


def require_benchmark(output, pools):
    benchmark = [
        verified_json(output / f"pool-{i:06}", "meta.json")
        for i in range(FIRST_POOL, FIRST_POOL + WAVES[0])
    ]
    startup = max(0, benchmark[0]["wallSecondsSinceStart"] - benchmark[0]["elapsedSeconds"])
    seconds = startup + pools * max(item["elapsedSeconds"] for item in benchmark)
    data = sum(
        file.stat().st_size
        for i in range(FIRST_POOL, FIRST_POOL + WAVES[0])
        for file in (output / f"pool-{i:06}").iterdir()
    )
    projected_bytes = data / WAVES[0] * pools
    if seconds > 6 * 3600 or projected_bytes > 3 * 2**30:
        raise ValueError(
            f"Pilot projection exceeds bounds: {seconds:.0f}s, {projected_bytes:.0f} bytes"
        )
    print(json.dumps({"projectedSeconds": seconds, "projectedBytes": projected_bytes}), flush=True)


def collect(args):
    started = time.monotonic()
    options = yaml.safe_load(args.config.read_text())["backend"]["options"]
    if (options["pool_anchors"], options["split"], options["source_dtype"]) != (
        2048,
        "validation",
        "bfloat16",
    ):
        raise ValueError("Natural Validation protocol differs from approved checkpoint evaluation")
    if sha256_file(Path(options["checkpoint"]) / "manifest.json") != CHECKPOINT:
        raise ValueError("Unexpected checkpoint manifest")
    revision = runtime_provenance()
    runtime = {key: revision[key] for key in ("git_sha", "git_tracked_changes", "torch", "cuda")}
    runtime["collectorSha256"] = sha256_file(Path(__file__))
    runtime["references"] = {
        name: sha256_file(args.reference / name / "COMPLETE")
        for name in ("importance", "representatives")
    }

    rows, strata = cohort(args.reference)
    components = [ComponentId.from_key(row["id"]) for row in rows]
    gates = sorted({comp.gate for comp in components})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.pools > WAVES[0]:
        require_benchmark(args.output, args.pools)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix="semantic-ot-stream-") as tmp:
        backend = V31CheckpointBackend(options, work_dir=Path(tmp))
        try:
            if backend.model_keys != MODELS or backend.n_features != 8192:
                raise ValueError("Unexpected model set or dictionary size")
            if int(backend.config.architecture.batch_topk_k) != 32:
                raise ValueError("Unexpected BatchTopK budget")
            expected_layers = {
                "gemma": range(9, 17),
                "smol": range(8, 16),
                "granite": range(16, 24),
            }
            for key, layers in expected_layers.items():
                if {m.layer for m in backend.factors(key)} != set(layers) or len(
                    backend.factors(key)
                ) != 56:
                    raise ValueError(f"Unexpected native matrix scope for {key}")
            for row, comp in zip(rows, components, strict=True):
                row["uNorm"] = float(
                    torch.linalg.vector_norm(
                        backend.factors(comp.model)[comp.matrix][0][comp.feature]
                    )
                )
            if any(not math.isfinite(row["uNorm"]) or row["uNorm"] <= 0 for row in rows):
                raise ValueError("Cohort contains a component with unusable native write norm")
            contract = {
                "checkpoint": CHECKPOINT,
                "configSha256": sha256_file(args.config),
                "reference": str(args.reference.resolve()),
                "firstPool": FIRST_POOL,
                "waves": WAVES,
                "cohortSeed": SEED,
                "strata": strata,
                "components": rows,
                "gateColumns": [gate.key for gate in gates],
                **runtime,
                "backend": backend.provenance(),
                "corpus": {
                    "id": backend.config.data.dataset_id,
                    "revision": backend.config.data.dataset_revision,
                },
            }
            contract = json.loads(json.dumps(contract, allow_nan=False))
            contract_path = args.output / "contract.json"
            if contract_path.exists():
                if json.loads(contract_path.read_text()) != contract:
                    raise ValueError("Cannot resume with a different extraction contract")
            else:
                write_json(contract_path, contract)
            contract_sha = sha256_file(contract_path)
            for ordinal, aligned in enumerate(
                backend.aligned_pools(args.pools, skip=FIRST_POOL), start=FIRST_POOL
            ):
                final = args.output / f"pool-{ordinal:06}"
                if final.exists():
                    prior = verified_json(final, "meta.json")
                    if prior["contractSha256"] != contract_sha or prior["pool"] != ordinal:
                        raise ValueError("Pool seal belongs to a different extraction contract")
                    continue
                elapsed = time.monotonic()
                if backend.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(backend.device)
                coefficients, continuous = measure(backend, aligned, components, gates)
                with tempfile.TemporaryDirectory(
                    prefix=f".pool-{ordinal:06}-", dir=args.output
                ) as target:
                    stage = Path(target)
                    with gzip.open(stage / "anchors.json.gz", "wt", encoding="utf-8") as handle:
                        json.dump(
                            anchor_rows(
                                aligned,
                                backend.tokenizers,
                                int(backend.config.data.sequence_length),
                            ),
                            handle,
                            ensure_ascii=False,
                        )
                    np.savez_compressed(
                        stage / "values.npz", coefficients=coefficients, gates=continuous
                    )
                    meta = {
                        "pool": ordinal,
                        "contractSha256": contract_sha,
                        "anchors": aligned.anchor_count,
                        "elapsedSeconds": time.monotonic() - elapsed,
                        "wallSecondsSinceStart": time.monotonic() - started,
                        "gpuPeakReservedBytes": torch.cuda.max_memory_reserved(backend.device)
                        if backend.device.type == "cuda"
                        else None,
                        "cpuPeakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        * 1024,
                        "slurmJob": revision["slurm_job_id"],
                    }
                    write_json(stage / "meta.json", meta)
                    write_json(
                        stage / "COMPLETE",
                        {
                            "sha256": {path.name: sha256_file(path) for path in stage.iterdir()},
                        },
                    )
                    stage.rename(final)
                print(json.dumps(meta), flush=True)
        finally:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pools", type=int, required=True, choices=WAVES)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()

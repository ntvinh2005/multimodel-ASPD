"""Replay a bounded set of archived Natural Validation anchors under their frozen test gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint import FileSystemReader

from aspd.multimodel.cli.evaluate_aspd_v3_negation import read_contract
from aspd.multimodel.cli.export_component_computation import factor
from aspd.multimodel.cli.export_natural_token_pilot import (
    MODELS,
    SITES,
    anchor_measurement,
    read,
    sha,
    verify,
)
from aspd.multimodel.posthoc.backend import MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.computation import component_read_write, local_mlp
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io
from aspd.multimodel.posthoc.v31_backend import _pad_id, load_frozen_config
from aspd.multimodel.v3.alignment import align_canonical_text
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.registry import build_model_registry_2b, resolve_submodule


def replay_rows(pilot, native, anchors, tokenizers):
    locations = {}
    for window in pilot["examples"].values():
        for point in window["points"].values():
            value = (window["documentId"], point)
            if locations.setdefault(point["anchor"], value) != value:
                raise ValueError("Duplicated anchor has conflicting measurements")
    results = []
    alignments = {}
    for anchor in dict.fromkeys(anchors):
        document, point = locations[anchor]
        source = native["documents"][document]
        text = source["canonicalPrefix"]
        end = point["characterBoundary"]
        if end is None or not 0 < end <= len(text):
            raise ValueError(f"Unverified canonical boundary: {anchor}")
        if document not in alignments:
            alignments[document] = align_canonical_text(
                text,
                document_id=document,
                source_group=source["source"],
                tokenizers=tokenizers,
                model_order=MODELS,
                max_length=pilot["provenance"]["maxLength"],
            )
        alignment = alignments[document]
        aligned = next((item for item in alignment.anchors if item.anchor_id == anchor), None)
        positions = (
            {model: position.token_index for model, position in aligned.positions.items()}
            if aligned
            else None
        )
        if aligned is None or aligned.character_boundary != end or positions != point["positions"]:
            raise ValueError(f"Canonical anchor or token positions differ for {anchor}")
        for model, tokens in source["inputIds"].items():
            if tuple(tokens) != alignment.tokenized[model].input_ids:
                raise ValueError(f"Stored token IDs differ for {anchor} · {model}")
        results.append((document, point))
    return results


def model_inputs(native, rows, model, pad_id, device):
    ids_by_document = {}
    for document, point in rows:
        end = point["positions"][model] + 1
        tokens = native["documents"][document]["inputIds"][model][:end]
        if len(tokens) > len(ids_by_document.get(document, [])):
            ids_by_document[document] = tokens
    lookup = {doc: i for i, doc in enumerate(ids_by_document)}
    size = max(map(len, ids_by_document.values()))
    ids = torch.full((len(lookup), size), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(ids_by_document.values()):
        ids[i, : len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, : len(tokens)] = 1
    return ModelInputs(
        ids,
        mask,
        torch.tensor([lookup[doc] for doc, _ in rows], device=device),
        torch.tensor([point["positions"][model] for _, point in rows], device=device),
    )


def load_test_archive(args):
    pilot, native = read(args.pilot), read(args.replay_inputs)
    if pilot["schemaVersion"] != 1 or pilot["split"] != "test" or native["schemaVersion"] != 1:
        raise ValueError("Unsupported held-out artifact")
    verify(args.pilot, native["pilotSha256"])
    archive_path = args.archive / "experiment_manifest.json"
    verify(archive_path, pilot["provenance"]["archiveManifestSha256"])
    archive = read(archive_path)
    _, reference = read_contract(args.config)
    checkpoint = Path(reference["checkpoint"])
    expected_hash = reference["checkpoint_manifest_sha256"]
    verify(checkpoint / "manifest.json", expected_hash)
    if (
        read(checkpoint / "COMPLETE")["manifest_sha256"] != expected_hash
        or pilot["checkpoint"] != expected_hash
        or archive["checkpoint_manifest_sha256"] != expected_hash
        or archive["selected_split"] != "test"
        or pilot["dataset"] != archive["experiment_family"]
        or not archive["natural_gate_only"]
    ):
        raise ValueError("Checkpoint or held-out provenance differs")
    config = load_frozen_config(
        reference["training_config"],
        checkpoint,
        skip_training_launch_evidence=reference["skip_training_launch_evidence"],
    )
    registry = build_model_registry_2b(config)
    for name in MODELS:
        layout = registry[name]
        recorded = archive["source_model_provenance"][name]
        if (layout.model_id, layout.revision, layout.tokenizer_revision) != (
            recorded["model_id"],
            recorded["revision"],
            recorded["tokenizer_revision"],
        ):
            raise ValueError(f"Tokenizer or model revision differs for {name}")
    tokenizers = {
        model: load_v3_tokenizer(registry[model], local_files_only=True) for model in MODELS
    }
    return pilot, native, archive, reference, registry, tokenizers


def verify_test_support(archive_root, archive, rows):
    with np.load(verify(archive_root / "anchors.npz", archive["anchors_sha256"])) as original:
        original_rows = {key: row for row, key in enumerate(original["anchor_ids"])}
        documents = original["document_indices"]
    source_docs = read(verify(archive_root / "documents.json", archive["documents_sha256"]))
    matrices = {}
    for site in SITES:
        report = archive["site_reports"][site]
        with np.load(
            verify(archive_root / f"{site}_natural_firings.npz", report["natural_firings_sha256"])
        ) as source:
            matrices[site] = {key: source[key] for key in ("indptr", "indices", "values")}
    for doc, point in rows:
        original_row = original_rows[point["anchor"]]
        if (
            point["pool"] != original_row // 2048
            or source_docs[int(documents[original_row])]["document_id"] != doc
            or point["gates"] != anchor_measurement(original_row, matrices)
        ):
            raise ValueError("Pilot point differs from the original frozen test support")


def replay(args):
    pilot, native, archive, reference, registry, tokenizers = load_test_archive(args)
    checkpoint = Path(reference["checkpoint"])
    expected_hash = reference["checkpoint_manifest_sha256"]
    windows = [pilot["examples"][key] for key in args.window]
    anchors = list(
        dict.fromkeys(
            [
                *args.anchor,
                *(
                    point["anchor"]
                    for window in windows
                    for point in window["points"].values()
                    if point["characterBoundary"] is not None
                ),
            ]
        )
    )
    if not 1 <= len(anchors) <= 512:
        raise ValueError("Select between 1 and 512 aligned anchors")
    rows = replay_rows(pilot, native, anchors, tokenizers)
    if len({document for document, _ in rows}) > 4:
        raise ValueError("Select at most four distinct documents per replay")
    verify_test_support(args.archive, archive, rows)
    components = [ComponentId.from_key(key) for key in dict.fromkeys(args.component)]
    model = components[0].model
    if any(component.model != model for component in components):
        raise ValueError("Replay one model per export")
    pairs = []
    for pair in args.local_pair:
        source, target = map(ComponentId.from_key, pair.split("=", maxsplit=1))
        if (
            source not in components
            or target not in components
            or source.matrix.matrix != "mlp_up"
            or target.matrix != MatrixId(model, source.matrix.layer, "mlp_down")
        ):
            raise ValueError(
                "Local replay requires selected Up and Down components in the same model and layer"
            )
        pairs.append((source, target))
    report = {
        "anchors": len(rows),
        "components": len(components),
        "model": model,
        "checkpoint": expected_hash,
        "maxNativeTokens": max(point["positions"][model] + 1 for _, point in rows),
    }
    print(json.dumps(report))
    if not args.execute:
        return

    model_layout = registry[model]
    frozen = load_frozen_v3_model(
        model_layout, args.device, local_files_only=True
    ).forward_model.eval()
    inputs = model_inputs(
        native, rows, model, _pad_id(tokenizers[model]), torch.device(args.device)
    )
    requested = {component.matrix for component in components}
    requested.update(MatrixId(model, source.matrix.layer, "mlp_gate") for source, _ in pairs)
    specs = {}
    for target in model_layout.targets():
        matrix = (target.layer_index, target.logical_name)
        name = next((item for item in requested if (item.layer, item.matrix) == matrix), None)
        if name is None:
            continue
        path, _ = resolve_submodule(frozen, target.physical_module_candidates)
        sliced = target.output_slice
        specs[name] = MatrixHookSpec(
            name, path, (sliced.start, sliced.stop) if sliced else None, target.d_in, target.d_out
        )
    if set(specs) != requested:
        raise ValueError("Replay matrices differ from the frozen model layout")
    captured = capture_matrix_io(frozen, inputs, specs.values())
    checkpoint_rank = checkpoint / f"rank-{MODELS.index(model):05d}"
    if read(checkpoint_rank / "sidecar.json")["model_key"] != model:
        raise ValueError("Wrong checkpoint owner")
    parts = checkpoint_rank / "dcp"
    metadata = FileSystemReader(parts).read_metadata()
    scale_path = (
        Path(reference["components"]["reference_posthoc_root"]) / "importance/importance_meta.json"
    )
    scales = read(scale_path)
    output_rows = {
        point["anchor"]: {
            "pool": point["pool"],
            "position": point["positions"][model],
            "components": {},
        }
        for _, point in rows
    }
    factors = {}
    for component in components:
        key = "model.matrix_bank.components." + component.matrix.key.replace(".", "__dot__")
        u, v = (
            factor(parts, metadata, key + "." + field, component.feature)[0].to(args.device)
            for field in ("U", "V")
        )
        factors[component] = u, v
        sigma = scales[component.matrix.key]["sigma"]
        gate_key = f"{component.gate.site.upper()}:{component.feature}"
        for i, (_, point) in enumerate(rows):
            summary = component_read_write(
                captured[component.matrix][0][i], u, v, gate_key in point["gates"], sigma
            )
            output_rows[point["anchor"]]["components"][component.key] = {
                key: summary[key] for key in ("reading", "coefficient", "magnitude", "normalized")
            }
    for source, target in pairs:
        gate = MatrixId(model, source.matrix.layer, "mlp_gate")
        activation = frozen.get_submodule(specs[gate].module_path.rsplit(".", 1)[0]).act_fn
        target_gate = f"{target.gate.site.upper()}:{target.feature}"
        for i, (_, point) in enumerate(rows):
            out = output_rows[point["anchor"]]
            coefficient = out["components"][source.key]["coefficient"]
            local = local_mlp(
                captured[gate][1][i],
                captured[source.matrix][1][i],
                activation,
                coefficient * factors[source][0],
                "mlp_up",
                factors[target][1],
                target_gate in point["gates"],
                checked_native_down_input=captured[target.matrix][0][i],
            )
            out.setdefault("local", []).append(
                {"source": source.key, "target": target.key, **local}
            )
    with args.output.open("x") as handle:
        json.dump(
            {
                "schemaVersion": 1,
                "checkpoint": expected_hash,
                "dataset": pilot["dataset"],
                "split": pilot["split"],
                "archiveManifestSha256": pilot["provenance"]["archiveManifestSha256"],
                "pilotSha256": sha(args.pilot),
                "nativeInputsSha256": sha(args.replay_inputs),
                "configSha256": sha(args.config),
                "modelRevision": model_layout.revision,
                "tokenizerRevision": model_layout.tokenizer_revision,
                "matrixScalesSha256": sha(scale_path),
                "rows": output_rows,
            },
            handle,
            allow_nan=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "replay-inputs", "archive", "config", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--component", action="append", required=True)
    parser.add_argument("--local-pair", action="append", default=[], metavar="UP=DOWN")
    parser.add_argument("--anchor", action="append", default=[])
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Include every aligned token of this example center",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not (args.anchor or args.window) or len(set(args.component)) > 12 or args.output.exists():
        parser.error("Select anchors or windows, up to 12 components, and a new artifact path")
    replay(args)


if __name__ == "__main__":
    main()

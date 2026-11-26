"""Recover exact Natural Validation token anchors and export frozen held-out gate values."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from aspd.multimodel.v3.alignment import align_canonical_text

MODELS = ("gemma", "smol", "granite")
SITES = ("pre", "mid")
CALENDAR_CONTRASTS = (
    ("day-first-control", "The meeting took place on 31 March.", "March"),
    ("change-day", "The meeting took place on 15 March.", "March"),
    ("change-month", "The meeting took place on 31 December.", "December"),
    ("month-only", "The meeting took place in March.", "March"),
    ("month-first", "The meeting took place on March 31.", "March"),
    ("verb", "They march every morning.", "march"),
)


def contrast_manifest(tokenizers, manifest, max_length):
    rows = []
    for key, text, word in CALENDAR_CONTRASTS:
        example = align_canonical_text(
            text,
            document_id=hashlib.sha256(text.encode()).hexdigest(),
            source_group="nvx025-controlled",
            tokenizers=tokenizers,
            model_order=MODELS,
            max_length=max_length,
        )
        boundary = text.index(word) + len(word)
        target = next(
            (anchor for anchor in example.anchors if anchor.character_boundary == boundary), None
        )
        if target is None:
            raise ValueError(f"Controlled target has no exact shared boundary: {key}")
        rows.append(
            {
                "id": key,
                "text": text,
                "target": word,
                "targetAnchor": target.anchor_id,
                "positions": {
                    model: position.token_index for model, position in target.positions.items()
                },
                "anchorCount": len(example.anchors),
                "measured": False,
            }
        )
    return {
        "schemaVersion": 1,
        "checkpoint": manifest["checkpoint_manifest_sha256"],
        "tokenizers": manifest["source_model_provenance"],
        "status": "tokenization-verified",
        "examples": rows,
    }


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Archive hash does not match: {path}")
    return path


def recover_native_prefix(windows, bos_id, limit):
    """Propose a Gemma prefix. Each resulting anchor still requires an exact archive hash match."""
    windows = sorted(set(windows), key=lambda row: (-len(row), row))
    prefix = next((row for row in windows if row[0] == bos_id), ())
    if not prefix:
        return prefix
    if any(
        row[: min(len(row), len(prefix))] != prefix[: len(row)]
        for row in windows
        if row[0] == bos_id
    ):
        return ()
    while len(prefix) < limit:
        overlap = 0
        extensions = set()
        for row in windows:
            for width in range(min(len(row) - 1, len(prefix)), max(7, overlap - 1), -1):
                if prefix[-width:] == row[:width]:
                    if width > overlap:
                        extensions.clear()
                        overlap = width
                    extensions.add(row[width:])
                    break
        if len(extensions) != 1:
            return prefix
        prefix = (prefix + next(iter(extensions)))[:limit]
    return prefix


def anchor_measurement(row, matrices):
    result = {}
    for site, matrix in matrices.items():
        start, end = matrix["indptr"][row : row + 2]
        for feature, value in zip(
            matrix["indices"][start:end], matrix["values"][start:end], strict=True
        ):
            if not np.isfinite(value) or value <= 0 or not 0 <= feature < 8192:
                raise ValueError("Invalid natural firing")
            key = f"{site.upper()}:{feature}"
            if key in result:
                raise ValueError(f"Duplicate gate: {key}")
            result[key] = float(value)
    return result


def aligned_window(ids, center, alignment, prefix, archive_rows):
    """Join native indices only when a unique window placement has its expected archived center."""
    anchors = {
        item.positions["gemma"].token_index: item
        for item in alignment.anchors
        if item.anchor_id in archive_rows
    }
    offsets = [
        i
        for i in range(len(prefix) - len(ids) + 1)
        if prefix[i : i + len(ids)] == ids
        and any(anchors[j].anchor_id == center for j in range(i, i + len(ids)) if j in anchors)
    ]
    if len(offsets) != 1:
        return {}
    start = offsets[0]
    return {j - start: anchors[j] for j in range(start, start + len(ids)) if j in anchors}


def export(args):
    root = args.archive
    manifest_path = root / "experiment_manifest.json"
    complete = read(root / "COMPLETE")
    verify(manifest_path, complete["manifest_sha256"])
    manifest = read(manifest_path)
    evidence = read(args.evidence)[args.candidate]
    if (
        complete["status"],
        manifest["selected_split"],
        manifest["natural_gate_only"],
        manifest["checkpoint_manifest_sha256"],
        tuple(manifest["model_order"]),
    ) != ("COMPLETE", "test", True, args.checkpoint, MODELS):
        raise ValueError("Unexpected held-out archive provenance")
    examples = {
        row["anchor"]: (gate, row) for gate, rows in evidence["examples"].items() for row in rows
    }
    with np.load(verify(root / "anchors.npz", manifest["anchors_sha256"])) as source:
        all_ids, document_indices = source["anchor_ids"], source["document_indices"]
    if (
        len(set(all_ids)) != len(all_ids)
        or len(all_ids) != manifest["accepted_evaluation_anchors"]
        or len(all_ids) % 2048
    ):
        raise ValueError("Unexpected anchor inventory")
    if set(all_ids).intersection(row["anchor"] for row in evidence["snapshots"]):
        raise ValueError("Unexpected intersection between validation witnesses and test extraction")
    documents = read(verify(root / "documents.json", manifest["documents_sha256"]))
    locations = {anchor: i for i, anchor in enumerate(all_ids)}
    document_ids = {
        documents[int(document_indices[locations[anchor]])]["document_id"] for anchor in examples
    }
    matrices = {}
    windows = defaultdict(set)
    rows = {}
    for site in SITES:
        report = manifest["site_reports"][site]
        with np.load(
            verify(root / f"{site}_natural_firings.npz", report["natural_firings_sha256"])
        ) as source:
            matrices[site] = {key: source[key] for key in ("indptr", "indices", "values")}
        with verify(
            root / f"{site}_feature_contexts.jsonl", report["feature_contexts_sha256"]
        ).open() as handle:
            for line in handle:
                feature = json.loads(line)
                gate = f"{site.upper()}:{feature['feature']}"
                for row in feature["contexts"]:
                    if row["document_id"] in document_ids:
                        windows[row["document_id"]].add(tuple(row["token_ids"]))
                    anchor = row["anchor_id"]
                    if anchor in examples and gate == examples[anchor][0]:
                        rows[anchor] = row
    tokenizers = {
        model: AutoTokenizer.from_pretrained(
            record["model_id"],
            revision=record["tokenizer_revision"],
            local_files_only=True,
            use_fast=True,
        )
        for model in MODELS
        for record in (manifest["source_model_provenance"][model],)
    }
    if args.contrasts and args.candidate != "NVX-025":
        raise ValueError("The controlled calendar contrasts are specific to NVX-025")
    contrasts = contrast_manifest(tokenizers, manifest, args.max_length) if args.contrasts else None
    gemma = tokenizers["gemma"]
    prefixes = {}
    native = {}
    for document_id in document_ids:
        prefix = recover_native_prefix(windows[document_id], gemma.bos_token_id, args.max_length)
        if prefix:
            text = gemma.decode(
                prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            source = next(
                row["source_group"] for row in documents if row["document_id"] == document_id
            )
            alignment = align_canonical_text(
                text,
                document_id=document_id,
                source_group=source,
                tokenizers=tokenizers,
                model_order=MODELS,
                max_length=args.max_length,
            )
            if alignment.tokenized["gemma"].input_ids[: len(prefix)] == prefix:
                prefixes[document_id] = (prefix, alignment)
                native[document_id] = {
                    "canonicalPrefix": text,
                    "source": source,
                    "inputIds": {
                        model: tokenized.input_ids
                        for model, tokenized in alignment.tokenized.items()
                    },
                }

    exported = {}
    for center, (gate, witness) in examples.items():
        row = rows[center]
        index = locations[center]
        doc = documents[int(document_indices[index])]
        if (
            row["anchor_index"] != index
            or doc["document_id"] != row["document_id"]
            or doc["source_group"] != witness["source"]
        ):
            raise ValueError(f"Wrong test archive row or document: {center}")
        ids = tuple(row["token_ids"])
        recovered = prefixes.get(doc["document_id"])
        aligned = (
            aligned_window(ids, center, recovered[1], recovered[0], locations) if recovered else {}
        )
        if aligned:
            center_index = next(i for i, anchor in aligned.items() if anchor.anchor_id == center)
        else:
            # The candidate evidence independently displays the center token. Never guess
            # neighboring anchors or break ties between repeated equal-strength tokens.
            peak = max(row["activations"])
            possible = [
                i
                for i, value in enumerate(row["activations"])
                if value == peak and gemma.convert_ids_to_tokens(ids[i]) == witness["token"]
            ]
            if len(possible) != 1:
                raise ValueError(f"Ambiguous archived center: {center}")
            center_index = possible[0]
        points = {}
        for i in sorted({*aligned, center_index}):
            item = aligned.get(i)
            anchor = item.anchor_id if item else center
            offset = locations[anchor]
            if document_indices[offset] != document_indices[index]:
                raise ValueError(f"Foreign document for {anchor}")
            points[i] = {
                "anchor": anchor,
                "pool": offset // 2048,
                "characterBoundary": item.character_boundary if item else None,
                "positions": {
                    model: position.token_index for model, position in item.positions.items()
                }
                if item
                else {},
                "gates": anchor_measurement(offset, matrices),
                "graphCoverage": "unavailable",
            }
        if points[center_index]["gates"].get(gate) != row["activation"]:
            raise ValueError(f"Archive center firing differs from context record: {center}")
        exported[center] = {
            "source": doc["source_group"],
            "documentId": doc["document_id"],
            "center": center_index,
            "tokens": gemma.convert_ids_to_tokens(ids),
            "tokenIds": ids,
            "points": points,
        }
    payload = {
        "schemaVersion": 1,
        "candidate": args.candidate,
        "checkpoint": args.checkpoint,
        "dataset": manifest["experiment_family"],
        "split": "test",
        "provenance": {
            "archiveManifestSha256": sha(manifest_path),
            "candidateEvidenceSha256": sha(args.evidence),
            "poolSize": 2048,
            "tokenizers": manifest["source_model_provenance"],
            "sourceSha256": sha(Path(__file__)),
            "maxLength": args.max_length,
        },
        "examples": exported,
    }
    with args.output.open("x") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    if args.replay_inputs:
        with args.replay_inputs.open("x") as handle:
            json.dump(
                {"schemaVersion": 1, "pilotSha256": sha(args.output), "documents": native},
                handle,
                separators=(",", ":"),
                allow_nan=False,
            )
    if args.contrasts:
        with args.contrasts.open("x") as handle:
            json.dump(contrasts, handle, separators=(",", ":"), allow_nan=False)
    print(
        json.dumps(
            {
                "examples": len(exported),
                "supportedWindowPositions": sum(len(x["points"]) for x in exported.values()),
                "displayedWindowPositions": sum(len(x["tokens"]) for x in exported.values()),
                "uniqueAnchors": len(
                    {p["anchor"] for r in exported.values() for p in r["points"].values()}
                ),
                "recoveredDocumentPrefixes": len(prefixes),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "evidence", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument(
        "--replay-inputs", type=Path, help="Write a separate, hash-bound native replay sidecar"
    )
    parser.add_argument(
        "--contrasts", type=Path, help="Verify the six unevaluated calendar contrasts"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate", default="NVX-025")
    parser.add_argument("--max-length", type=int, default=512)
    export(parser.parse_args())


if __name__ == "__main__":
    main()

"""Freeze held-out documents and build a blinded offline human ASPD evaluation."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.cli.analyze_aspd_v3_mechanistic import (
    _document_counts,
    _feature_anchor_lists,
    _load_sparse,
    _stratified_features,
)
from aspd.multimodel.cli.prepare_aspd_v3_intruder_final import (
    _context_shape,
    _text_jaccard,
)
from aspd.multimodel.v3.human_interpretability import (
    balanced_positions,
    balanced_round_robin,
    canonical_json_bytes,
    context_views,
    freeze_excluded_documents,
    response_template,
    sha256_bytes,
    sha256_file,
    validate_public_blinding,
)
from aspd.multimodel.v3.mechanistic_validation import HumanInterpretabilityContract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-exclusions")
    freeze.add_argument("--source-manifest", type=Path, action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--control-dir", type=Path, required=True)
    build.add_argument("--zscore-dir", type=Path, required=True)
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--html-template", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any, *, protected: bool = False) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
    if protected:
        path.chmod(0o600)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, protected: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    if protected:
        path.chmod(0o600)


def _contexts(path: Path) -> dict[int, list[dict[str, Any]]]:
    return {int(row["feature"]): list(row["contexts"]) for row in _jsonl(path)}


def _choose_targets(
    contexts: list[dict[str, Any]], *, count: int, maximum_jaccard: float
) -> list[dict[str, Any]] | None:
    chosen: list[dict[str, Any]] = []
    documents: set[str] = set()
    for row in sorted(contexts, key=lambda value: (-float(value["activation"]), value["anchor_id"])):
        document = str(row["document_id"])
        text = str(row.get("canonical_context", ""))
        if not text or document in documents:
            continue
        if any(_text_jaccard(text, str(other["canonical_context"])) > maximum_jaccard for other in chosen):
            continue
        context_views(row, compact_characters_per_side=1)
        chosen.append(row)
        documents.add(document)
        if len(chosen) == count:
            return chosen
    return None


def _intruder_candidates(
    *,
    feature: int,
    stratum: str,
    sampled: list[dict[str, Any]],
    contexts: dict[int, list[dict[str, Any]]],
    targets: list[dict[str, Any]],
    firing: np.ndarray,
    maximum_jaccard: float,
    rng: np.random.Generator,
) -> list[tuple[tuple[Any, ...], int, dict[str, Any]]]:
    target_documents = {str(row["document_id"]) for row in targets}
    target_groups = {str(row.get("source_group", "unknown")) for row in targets}
    target_shapes = Counter(_context_shape(row) for row in targets)
    target_length = float(np.median([len(str(row["canonical_context"])) for row in targets]))
    active = set(int(value) for value in firing)
    donors = [
        row for row in sampled
        if int(row["feature"]) != feature and row["frequency_stratum"] == stratum
    ]
    rng.shuffle(donors)
    candidates: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for donor in donors:
        donor_feature = int(donor["feature"])
        for row in contexts.get(donor_feature, []):
            text = str(row.get("canonical_context", ""))
            if not text or str(row["document_id"]) in target_documents:
                continue
            if int(row["anchor_index"]) in active:
                continue
            if any(
                _text_jaccard(text, str(target["canonical_context"])) > maximum_jaccard
                for target in targets
            ):
                continue
            context_views(row, compact_characters_per_side=1)
            score = (
                int(str(row.get("source_group", "unknown")) not in target_groups),
                int(_context_shape(row) not in target_shapes),
                abs(len(text) - target_length),
                donor_feature,
                str(row["anchor_id"]),
            )
            candidates.append((score, donor_feature, row))
    return sorted(candidates, key=lambda value: value[0])


def _build_group(
    *,
    arm: str,
    site: str,
    directory: Path,
    contract: HumanInterpretabilityContract,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indptr, indices, _values, shape = _load_sparse(directory / f"{site}_natural_firings.npz")
    documents_rows = _json(directory / "documents.json")
    with np.load(directory / "anchors.npz") as artifact:
        document_indices = artifact["document_indices"].astype(np.int64, copy=False)
    document_ids = np.asarray([str(row["document_id"]) for row in documents_rows], dtype=object)
    anchor_documents = document_ids[document_indices]
    firing, counts = _feature_anchor_lists(indptr, indices, shape[1])
    document_counts = _document_counts(firing, anchor_documents)
    sampled, _strata, sampling = _stratified_features(
        counts,
        document_counts,
        minimum_fires=contract.minimum_feature_fires,
        minimum_documents=contract.minimum_unique_documents,
        sample_size=contract.features_per_arm_site,
        seed=contract.feature_sampling_seed + contract.arms.index(arm) * 100 + (site == "mid"),
    )
    retained = _contexts(directory / f"{site}_feature_contexts.jsonl")
    rng = np.random.default_rng(
        contract.intruder_selection_seed
        + contract.arms.index(arm) * 1_000
        + (100 if site == "mid" else 0)
    )
    parents: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for selected in sampled:
        feature = int(selected["feature"])
        targets = _choose_targets(
            retained.get(feature, []),
            count=contract.contexts_per_feature,
            maximum_jaccard=contract.maximum_normalized_text_jaccard,
        )
        if targets is None:
            exclusions["insufficient_distinct_valid_target_contexts"] += 1
            continue
        candidates = _intruder_candidates(
            feature=feature,
            stratum=str(selected["frequency_stratum"]),
            sampled=sampled,
            contexts=retained,
            targets=targets,
            firing=firing[feature],
            maximum_jaccard=contract.maximum_normalized_text_jaccard,
            rng=rng,
        )
        if not candidates:
            exclusions["no_matched_inactive_intruder"] += 1
            continue
        score, donor_feature, intruder = candidates[0]
        parent_uid = hashlib.sha256(f"{arm}\0{site}\0{feature}".encode()).hexdigest()[:24]
        parents.append(
            {
                "parent_uid": parent_uid,
                "arm": arm,
                "site": site,
                "feature": feature,
                "frequency_stratum": str(selected["frequency_stratum"]),
                "feature_fires": int(selected["fires"]),
                "feature_unique_documents": int(selected["unique_documents"]),
                "targets": targets,
                "intruder": intruder,
                "intruder_donor_feature": donor_feature,
                "intruder_match_score": list(score[:3]),
                "target_inactive_at_intruder": int(intruder["anchor_index"])
                not in set(int(value) for value in firing[feature]),
            }
        )
    return parents, {
        **sampling,
        "dictionary_capacity": shape[1],
        "features_firing": int(np.count_nonzero(counts)),
        "successfully_constructed": len(parents),
        "construction_exclusions": dict(exclusions),
    }


def _public_example(row: dict[str, Any], *, candidate_id: str, compact: int) -> dict[str, Any]:
    return {"candidate_id": candidate_id, **context_views(row, compact)}


def _audit_context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "activation": float(row["activation"]),
        "anchor_id": str(row["anchor_id"]),
        "anchor_index": int(row["anchor_index"]),
        "document_id": str(row["document_id"]),
        "source_group": str(row.get("source_group", "unknown")),
        "model_token_index": int(row["model_token_index"]),
        "canonical_character_boundary": int(row["canonical_character_boundary"]),
        "canonical_anchor_start": int(row["canonical_anchor_start"]),
        "canonical_anchor_end": int(row["canonical_anchor_end"]),
        "canonical_anchor_text": str(row["canonical_anchor_text"]),
    }


def _presentation(
    parent: dict[str, Any],
    *,
    intruder_position: int,
    compact_characters_per_side: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = list(parent["targets"])
    rng.shuffle(targets)
    displayed = list(targets)
    displayed.insert(intruder_position - 1, parent["intruder"])
    public = {
        "schema_version": 1,
        "phase": "scientific",
        "item_id": "PENDING",
        "batch": 0,
        "examples": [
            _public_example(
                row,
                candidate_id=str(index),
                compact=compact_characters_per_side,
            )
            for index, row in enumerate(displayed, start=1)
        ],
    }
    answer = {
        "schema_version": 1,
        "item_id": "PENDING",
        "phase": "scientific",
        "parent_uid": parent["parent_uid"],
        "repeat_of_parent_uid": parent.get("repeat_of_parent_uid"),
        "arm": parent["arm"],
        "site": parent["site"],
        "feature": parent["feature"],
        "frequency_stratum": parent["frequency_stratum"],
        "feature_fires": parent["feature_fires"],
        "feature_unique_documents": parent["feature_unique_documents"],
        "intruder_position": intruder_position,
        "intruder_donor_feature": parent["intruder_donor_feature"],
        "intruder_match_score": parent["intruder_match_score"],
        "target_inactive_at_intruder": parent["target_inactive_at_intruder"],
        "displayed_contexts": [
            {**_audit_context(row), "is_intruder": index == intruder_position}
            for index, row in enumerate(displayed, start=1)
        ],
    }
    return public, answer


def _practice_items() -> list[dict[str, Any]]:
    topics = [
        (["apple orchard", "pear harvest", "peach tree", "plum jam"], "diesel engine"),
        (["neural synapse", "cortical neuron", "axon terminal", "brain circuit"], "tax invoice"),
        (["def parse_json", "return payload", "raise ValueError", "python traceback"], "medieval castle"),
        (["The result may vary.", "Results were mixed.", "No clear effect.", "Evidence is limited."], "The outcome remains uncertain."),
        (["blue square", "red circle", "green triangle", "yellow star"], "purple diamond"),
        (["short sentence", "another phrase", "brief statement", "small clause"], "tiny remark"),
        (["quarterly revenue", "operating margin", "cash flow", "annual report"], "bird migration"),
        (["French verb", "Spanish noun", "Italian phrase", "Latin grammar"], "GPU kernel"),
        (["Monday meeting", "Tuesday call", "Friday review", "Sunday note"], "Wednesday memo"),
        (["alpha beta", "gamma delta", "epsilon zeta", "eta theta"], "iota kappa"),
    ]
    rows: list[dict[str, Any]] = []
    for index, (targets, intruder) in enumerate(topics, start=1):
        values = list(targets) + [intruder]
        examples = []
        for candidate, value in enumerate(values, start=1):
            anchor = value.split()[0]
            prefix, suffix = value.split(anchor, maxsplit=1)
            view = {
                "prefix": prefix,
                "anchor": anchor,
                "suffix": suffix,
                "truncated_left": False,
                "truncated_right": False,
            }
            examples.append({"candidate_id": str(candidate), "compact": view, "expanded": view})
        rows.append(
            {
                "schema_version": 1,
                "phase": "practice",
                "item_id": f"Practice {index:02d}",
                "batch": 0,
                "examples": examples,
            }
        )
    return rows


def _validate_extraction(directory: Path, contract: HumanInterpretabilityContract) -> dict[str, Any]:
    manifest = _json(directory / "experiment_manifest.json")
    if manifest.get("protocol_fingerprint") != contract.fingerprint():
        raise ValueError(f"protocol mismatch: {directory}")
    if manifest.get("experiment_family") != contract.experiment_family:
        raise ValueError(f"experiment family mismatch: {directory}")
    if not manifest.get("excluded_documents_disjoint") or not manifest.get("natural_gate_only"):
        raise ValueError(f"extraction is not natural and document-disjoint: {directory}")
    if int(manifest["forced_activations"]) != 0:
        raise ValueError(f"forced activations are present: {directory}")
    if int(manifest["accepted_evaluation_anchors"]) != contract.accepted_evaluation_anchors:
        raise ValueError(f"anchor budget mismatch: {directory}")
    return manifest


def _render_html(template: str, dataset: bytes, dataset_sha256: str, template_payload: bytes) -> str:
    replacements = {
        "__DATASET_B64__": base64.b64encode(dataset).decode(),
        "__DATASET_SHA256__": dataset_sha256,
        "__RESPONSE_TEMPLATE_B64__": base64.b64encode(template_payload).decode(),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"HTML template marker count changed: {marker}")
        template = template.replace(marker, value)
    return template


def _build(args: argparse.Namespace) -> None:
    contract = HumanInterpretabilityContract.load(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    control_manifest = _validate_extraction(args.control_dir, contract)
    zscore_manifest = _validate_extraction(args.zscore_dir, contract)
    if sha256_file(args.control_dir / "anchors.npz") != sha256_file(args.zscore_dir / "anchors.npz"):
        raise ValueError("control and z-score extractions do not share matched anchors")
    if control_manifest["source_model_provenance"] != zscore_manifest["source_model_provenance"]:
        raise ValueError("source-model provenance differs between arms")

    groups: dict[str, list[dict[str, Any]]] = {}
    coverage: dict[str, Any] = {}
    for arm, directory in (
        ("full_only_control", args.control_dir),
        ("per_anchor_zscore", args.zscore_dir),
    ):
        for site in ("pre", "mid"):
            key = f"{arm}.{site}"
            groups[key], coverage[key] = _build_group(
                arm=arm, site=site, directory=directory, contract=contract
            )

    rng = np.random.default_rng(contract.presentation_seed)
    for parents in groups.values():
        positions = balanced_positions(
            [str(parent["frequency_stratum"]) for parent in parents], rng=rng
        )
        for parent, position in zip(parents, positions, strict=True):
            parent["intruder_position"] = position

    all_presentations: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key, parents in groups.items():
        all_presentations[key] = [
            _presentation(
                parent,
                intruder_position=int(parent["intruder_position"]),
                compact_characters_per_side=contract.compact_context_characters_per_side,
                rng=rng,
            )
            for parent in parents
        ]
        repeat_count = contract.repeated_parent_items // len(groups)
        if len(parents) < repeat_count:
            raise ValueError(f"not enough parents for repeat sampling in {key}")
        repeated = rng.choice(len(parents), size=repeat_count, replace=False)
        for offset, parent_index in enumerate(repeated):
            source = dict(parents[int(parent_index)])
            source["repeat_of_parent_uid"] = source["parent_uid"]
            repeat_position = (int(source["intruder_position"]) + offset) % 5 + 1
            all_presentations[key].append(
                _presentation(
                    source,
                    intruder_position=repeat_position,
                    compact_characters_per_side=contract.compact_context_characters_per_side,
                    rng=rng,
                )
            )

    ordered = balanced_round_robin(all_presentations, rng=rng)
    public_scientific: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index, (public, answer) in enumerate(ordered, start=1):
        item_id = f"Feature Set {index:03d}"
        public["item_id"] = item_id
        public["batch"] = (index - 1) // contract.scientific_batch_size + 1
        answer["item_id"] = item_id
        answer["batch"] = public["batch"]
        public_scientific.append(public)
        answers.append(answer)

    practice = _practice_items()[: contract.practice_items]
    if len(practice) != contract.practice_items:
        raise ValueError("practice item count changed")
    public_items = practice + public_scientific
    validate_public_blinding(public_items)
    dataset = {
        "schema_version": 1,
        "title": "ASPD v3.1 Human Interpretability Evaluation",
        "instructions_version": "human_intruder_coherence_v1",
        "practice_item_count": len(practice),
        "scientific_unique_parent_count": sum(len(values) for values in groups.values()),
        "scientific_repeat_count": contract.repeated_parent_items,
        "scientific_presentation_count": len(public_scientific),
        "scientific_batch_size": contract.scientific_batch_size,
        "items": public_items,
    }
    dataset_bytes = canonical_json_bytes(dataset)
    dataset_sha256 = sha256_bytes(dataset_bytes)
    template = response_template(dataset_sha256, public_items)
    template_bytes = canonical_json_bytes(template)
    answer_key_hash = sha256_bytes(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in answers
        ).encode()
    )

    public_dir = args.output_dir / "public"
    protected_dir = args.output_dir / "protected"
    audit_dir = args.output_dir / "audit"
    for directory in (public_dir, protected_dir, audit_dir):
        directory.mkdir(parents=True)
    with (public_dir / "blinded_items.json").open("xb") as handle:
        handle.write(dataset_bytes)
    with (public_dir / "response_template.json").open("xb") as handle:
        handle.write(template_bytes)
    html = _render_html(
        args.html_template.read_text(encoding="utf-8"),
        dataset_bytes,
        dataset_sha256,
        template_bytes,
    )
    (public_dir / "evaluation.html").write_text(html, encoding="utf-8")
    (public_dir / "README.txt").write_text(
        "Open evaluation.html in a modern browser. Enter an evaluator ID, complete the practice "
        "items, then work through the scientific batches. Progress is saved locally in the browser. "
        "Use Export responses regularly. To resume elsewhere, use Import responses. Return only the "
        "exported response JSON to the grading operator; do not open the protected directory.\n",
        encoding="utf-8",
    )
    _write_jsonl(protected_dir / "answer_key.jsonl", answers, protected=True)
    _write_json(
        protected_dir / "answer_key_manifest.json",
        {
            "schema_version": 1,
            "answer_key_sha256": answer_key_hash,
            "dataset_sha256": dataset_sha256,
            "scientific_presentations": len(answers),
        },
        protected=True,
    )
    with (audit_dir / "sampling.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "parent_uid", "arm", "site", "feature", "frequency_stratum",
                "feature_fires", "feature_unique_documents",
            ),
        )
        writer.writeheader()
        for parents in groups.values():
            for parent in parents:
                writer.writerow({key: parent[key] for key in writer.fieldnames})
    validation_report = {
        "public_blinding": "PASS",
        "target_document_distinctness": "PASS",
        "intruder_document_disjointness": "PASS",
        "target_inactivity_at_intruder": (
            "PASS" if all(parent["target_inactive_at_intruder"] for values in groups.values() for parent in values) else "FAIL"
        ),
        "anchor_span_offsets": "PASS",
        "natural_gate_only": "PASS",
        "matched_anchors": "PASS",
        "source_weights_unchanged": (
            "PASS"
            if all(
                row["source_weights_unchanged"] == "PASS"
                for manifest in (control_manifest, zscore_manifest)
                for row in manifest["rank_reports"]
            )
            else "FAIL"
        ),
    }
    if "FAIL" in validation_report.values():
        raise ValueError(f"human package validation failed: {validation_report}")
    _write_json(audit_dir / "validation_report.json", validation_report)
    _write_json(
        audit_dir / "preparation_manifest.json",
        {
            "schema_version": 1,
            "status": "HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE",
            "git_sha": _git_sha(),
            "protocol_sha256": sha256_file(args.protocol),
            "protocol_fingerprint": contract.fingerprint(),
            "control_extraction_manifest_sha256": sha256_file(args.control_dir / "experiment_manifest.json"),
            "zscore_extraction_manifest_sha256": sha256_file(args.zscore_dir / "experiment_manifest.json"),
            "matched_anchor_sha256": sha256_file(args.control_dir / "anchors.npz"),
            "checkpoint_hashes": {
                "full_only_control": control_manifest["checkpoint_manifest_sha256"],
                "per_anchor_zscore": zscore_manifest["checkpoint_manifest_sha256"],
            },
            "source_model_provenance": control_manifest["source_model_provenance"],
            "dataset_sha256": dataset_sha256,
            "answer_key_sha256": answer_key_hash,
            "response_template_sha256": sha256_bytes(template_bytes),
            "html_sha256": sha256_file(public_dir / "evaluation.html"),
            "coverage": coverage,
            "scientific_unique_parent_count": sum(len(values) for values in groups.values()),
            "scientific_repeat_count": contract.repeated_parent_items,
            "scientific_presentation_count": len(public_scientific),
            "practice_item_count": len(practice),
            "validation": validation_report,
        },
    )
    print(json.dumps(_json(audit_dir / "preparation_manifest.json"), indent=2, sort_keys=True))


def main() -> None:
    args = _args()
    if args.command == "freeze-exclusions":
        report = freeze_excluded_documents(args.source_manifest, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    _build(args)


if __name__ == "__main__":
    main()

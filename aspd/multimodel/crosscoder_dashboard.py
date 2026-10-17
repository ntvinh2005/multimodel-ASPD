"""Read-only dashboard adapters for Crosscoder and cross-method research artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from aspd.multimodel.dashboard import AnalysisDashboardStore, activation_example_payload

STRICT_CLASSES = {"base-specific", "shared", "intermediate", "finetuned-specific"}
COMPARISON_STATUSES = {
    "semantic_overlap": (
        "Retained contexts support a similar qualitative interpretation; this does not establish "
        "latent identity or causal equivalence."
    ),
    "weak_semantic_overlap": (
        "Some retained contexts appear qualitatively related, but the evidence is inconsistent "
        "or incomplete."
    ),
    "low_support_hypothesis": (
        "A possible qualitative relationship is recorded, but at least one side has too little "
        "validation support for a strong comparison."
    ),
    "aspd_only_parameter_evidence": (
        "ASPD measures parameter-mechanism concentration without a strict-exclusive Crosscoder "
        "counterpart."
    ),
    "unresolved": "The current artifacts do not support a stable relationship interpretation.",
    "not_tested": "The proposed relationship has not been measured.",
}


def _read_object(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing file: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not parse {path.name}: {exc}"
    if not isinstance(value, dict):
        return {}, f"{path.name} must contain an object"
    return value, None


def _read_yaml_object(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing file: {path}"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, f"could not parse {path.name}: {exc}"
    if not isinstance(value, dict):
        return {}, f"{path.name} must contain an object"
    return value, None


def _display_group(group: str) -> str:
    return "shared-center" if group == "shared" else group


class CrosscoderDashboardStore:
    """Serve existing Crosscoder analysis artifacts without recomputing measurements."""

    def __init__(
        self,
        analysis_dir: str | Path,
        *,
        annotations_path: str | Path | None = None,
    ) -> None:
        self.analysis_dir = Path(analysis_dir).resolve()
        self.run_dir = self.analysis_dir.parent
        self.diagnostics: list[str] = []

        self.summary, error = _read_object(self.analysis_dir / "summary.json")
        if error:
            self.diagnostics.append(error)
        self.feature_artifact, error = _read_object(self.analysis_dir / "feature_metrics.json")
        if error:
            self.diagnostics.append(error)
        self.example_artifact, error = _read_object(
            self.analysis_dir / "top_activation_examples.json"
        )
        if error:
            self.diagnostics.append(error)
        self.local_annotations, error = _read_object(
            self.analysis_dir / "semantic_annotations.json"
        )
        if error:
            self.diagnostics.append(error)

        # Both are optional post-ranking artifacts. Their absence must not degrade the
        # canonical measured Crosscoder dashboard.
        self.context_backfill: dict[str, Any] = {}
        context_backfill_path = self.analysis_dir / "content_semantic_scan_contexts.json"
        if context_backfill_path.exists():
            self.context_backfill, error = _read_object(context_backfill_path)
            if error:
                self.diagnostics.append(error)
        self.triage_artifact: dict[str, Any] = {}
        triage_path = self.analysis_dir / "content_semantic_scan.json"
        if triage_path.exists():
            self.triage_artifact, error = _read_object(triage_path)
            if error:
                self.diagnostics.append(error)
        self.config, error = _read_yaml_object(self.run_dir / "config_resolved.yaml")
        if error:
            self.diagnostics.append(error)
        self.reference, error = _read_object(self.run_dir / "reference.json")
        if error:
            self.diagnostics.append(error)

        self.annotations_path = Path(annotations_path).resolve() if annotations_path else None
        self.research_annotations: dict[str, Any] = {}
        if self.annotations_path is not None:
            self.research_annotations, error = _read_object(self.annotations_path)
            if error:
                self.diagnostics.append(error)

        raw_features = self.feature_artifact.get("features", [])
        self.features: list[dict[str, Any]] = (
            [row for row in raw_features if isinstance(row, dict)]
            if isinstance(raw_features, list)
            else []
        )
        self.features_by_id = {
            int(row["feature_id"]): row
            for row in self.features
            if isinstance(row.get("feature_id"), int)
        }
        self.candidate_groups = self._candidate_groups()
        self.group_membership: dict[int, list[str]] = defaultdict(list)
        for group, feature_ids in self.candidate_groups.items():
            for feature_id in feature_ids:
                self.group_membership[feature_id].append(_display_group(group))
        self.examples, self.example_sources = self._example_map()
        self.annotations, self.annotation_metadata = self._annotation_index()
        self.semantic_triage = self._triage_index()
        self.completed_step = self._completed_step()
        self._validate()
        self.available = bool(self.summary and self.features_by_id)
        self._overview_payload = self._build_overview()

    def _candidate_groups(self) -> dict[str, list[int]]:
        raw = self.summary.get("candidate_groups")
        if not isinstance(raw, dict):
            raw = self.feature_artifact.get("metadata", {}).get("candidate_groups", {})
        return {
            str(group): [int(feature_id) for feature_id in ids if isinstance(feature_id, int)]
            for group, ids in raw.items()
            if isinstance(ids, list)
        } if isinstance(raw, dict) else {}

    def _example_map(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
        raw = self.example_artifact.get("features", {})
        if not isinstance(raw, dict):
            raw = {}
        examples = {
            str(feature_id): [example for example in examples if isinstance(example, dict)]
            for feature_id, examples in raw.items()
            if isinstance(examples, list)
        }
        sources = {feature_id: "crosscoder/top_activation_examples.json" for feature_id in examples}
        backfill = self.context_backfill.get("features", {})
        if isinstance(backfill, dict):
            for feature_id, feature_examples in backfill.items():
                if isinstance(feature_examples, list):
                    key = str(feature_id)
                    examples[key] = [
                        example for example in feature_examples if isinstance(example, dict)
                    ]
                    sources[key] = "crosscoder/content_semantic_scan_contexts.json"
        return examples, sources

    def _triage_index(self) -> dict[int, dict[str, Any]]:
        rows = self.triage_artifact.get("features", [])
        if not isinstance(rows, list):
            return {}
        return {
            int(row["feature_id"]): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("feature_id"), int)
        }

    def _annotation_index(self) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        indexed: dict[int, dict[str, Any]] = {}
        local = self.local_annotations.get("annotations", [])
        if isinstance(local, list):
            for row in local:
                if isinstance(row, dict) and isinstance(row.get("feature_id"), int):
                    indexed[int(row["feature_id"])] = row
        tracked = self.research_annotations.get("crosscoder", [])
        if isinstance(tracked, list):
            for row in tracked:
                if isinstance(row, dict) and isinstance(row.get("feature_id"), int):
                    indexed[int(row["feature_id"])] = row
        metadata = self.research_annotations.get("metadata", {})
        if not isinstance(metadata, dict) or not metadata:
            metadata = self.local_annotations.get("metadata", {})
        return indexed, metadata if isinstance(metadata, dict) else {}

    def _completed_step(self) -> int | None:
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return None
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            last = json.loads(lines[-1]) if lines else {}
        except (OSError, json.JSONDecodeError):
            return None
        step = last.get("step") if isinstance(last, dict) else None
        return int(step) if isinstance(step, int | float) else None

    def _validate(self) -> None:
        expected = self.summary.get("n_features")
        if isinstance(expected, int) and len(self.features_by_id) != expected:
            self.diagnostics.append(
                f"feature_metrics.json contains {len(self.features_by_id)} unique rows; expected {expected}"
            )
        for feature_id, row in self.features_by_id.items():
            classification = row.get("classification")
            if classification not in STRICT_CLASSES:
                self.diagnostics.append(
                    f"feature {feature_id} has invalid strict classification {classification!r}"
                )
        for group, feature_ids in self.candidate_groups.items():
            invalid = [feature_id for feature_id in feature_ids if feature_id not in self.features_by_id]
            if invalid:
                self.diagnostics.append(f"candidate group {group!r} has invalid IDs: {invalid}")

    def _metadata(self) -> dict[str, Any]:
        models = self.config.get("models", [])
        model_ids = [model.get("model_id") for model in models if isinstance(model, dict)]
        sites = []
        for model in models:
            grounding = model.get("grounding", {}) if isinstance(model, dict) else {}
            if isinstance(grounding, dict):
                module = grounding.get("module")
                capture = grounding.get("capture")
                sites.append(f"{module} ({capture})" if module and capture else module)
        crosscoder = self.config.get("crosscoder", {})
        data = self.config.get("data", {})
        return {
            "run_name": self.config.get("name"),
            "model_ids": model_ids,
            "sites": sites,
            "n_features": self.summary.get("n_features"),
            "training_steps": self.completed_step,
            "configured_max_steps": (
                crosscoder.get("max_steps") if isinstance(crosscoder, dict) else None
            ),
            "train_batch_topk": self.summary.get("k"),
            "validation_tokens": self.summary.get("tokens"),
            "sequence_length": data.get("sequence_length") if isinstance(data, dict) else None,
            "checkpoint": self.summary.get("checkpoint"),
            "config_path": str(self.run_dir / "config_resolved.yaml"),
            "analysis_dir": str(self.analysis_dir),
            "reference": self.reference,
        }

    def _histogram(self, bins: int = 50) -> list[dict[str, float | int]]:
        counts = [0] * bins
        for row in self.features_by_id.values():
            value = row.get("relative_decoder_norm")
            if isinstance(value, int | float):
                index = min(bins - 1, max(0, int(float(value) * bins)))
                counts[index] += 1
        return [
            {"start": index / bins, "end": (index + 1) / bins, "count": count}
            for index, count in enumerate(counts)
        ]

    def _build_overview(self) -> dict[str, Any]:
        counts = self.summary.get("classification_counts", {})
        strict_exclusive_absent = (
            isinstance(counts, dict)
            and counts.get("base-specific") == 0
            and counts.get("finetuned-specific") == 0
        )
        scores = [
            float(row["relative_decoder_norm"])
            for row in self.features_by_id.values()
            if isinstance(row.get("relative_decoder_norm"), int | float)
        ]
        dead_fraction = self.summary.get("dead_fraction")
        n_features = self.summary.get("n_features")
        dead_count = (
            round(float(dead_fraction) * int(n_features))
            if isinstance(dead_fraction, int | float) and isinstance(n_features, int)
            else None
        )
        return {
            "available": self.available,
            "diagnostics": self.diagnostics,
            "metadata": self._metadata(),
            "metrics": {
                "fve_total": self.summary.get("fve_total"),
                "fve_base": self.summary.get("fve_base"),
                "fve_finetuned": self.summary.get("fve_finetuned"),
                "validation_l0": self.summary.get("l0"),
                "dead_count": dead_count,
                "dead_fraction": dead_fraction,
                "relative_norm_min": min(scores) if scores else None,
                "relative_norm_max": max(scores) if scores else None,
            },
            "classification_counts": counts if isinstance(counts, dict) else {},
            "classification_fractions": self.summary.get("classification_fractions", {}),
            "thresholds": self.summary.get("thresholds", {}),
            "relative_decoder_norm_convention": self.summary.get(
                "relative_decoder_norm_convention"
            ),
            "relative_norm_histogram": self._histogram(),
            "strict_exclusive_absent": strict_exclusive_absent,
            "strict_result": (
                "No strict model-exclusive features under the published thresholds."
                if strict_exclusive_absent
                else "Strict model-exclusive features are present under the published thresholds."
            ),
            "candidate_selection": self.summary.get("candidate_selection"),
            "candidate_groups": {
                _display_group(group): ids for group, ids in self.candidate_groups.items()
            },
            "sources": {
                "summary": str(self.analysis_dir / "summary.json"),
                "features": str(self.analysis_dir / "feature_metrics.json"),
                "config": str(self.run_dir / "config_resolved.yaml"),
                "reference": str(self.run_dir / "reference.json"),
            },
        }

    def overview(self) -> dict[str, Any]:
        """Return the startup-derived overview without rescanning all 65k feature rows."""

        return self._overview_payload

    def _semantic_annotation(self, feature_id: int) -> dict[str, Any] | None:
        row = self.annotations.get(feature_id)
        if row is None:
            return None
        return {
            "status": row.get("annotation_status", "annotated"),
            "tentative_label": row.get("tentative_label"),
            "interpretation_confidence": row.get("confidence"),
            "annotation_source": row.get("annotation_source")
            or self.annotation_metadata.get("annotation_source"),
            "evidence_validation_sequence_indices": row.get(
                "evidence_validation_sequence_indices", []
            ),
            "warning": row.get("warning"),
            "notes": row.get("notes"),
            "model": self.annotation_metadata.get("model"),
            "model_version": self.annotation_metadata.get("model_version"),
            "limitations": self.annotation_metadata.get("limitations", []),
            "source": str(self.annotations_path) if self.annotations_path else None,
            "is_measured_ground_truth": False,
        }

    def _semantic_triage(self, feature_id: int) -> dict[str, Any] | None:
        row = self.semantic_triage.get(feature_id)
        if row is None:
            return None
        judge = row.get("judge")
        structural = row.get("deterministic_structural")
        return {
            "judge": judge if isinstance(judge, dict) else None,
            "deterministic_structural": structural if isinstance(structural, dict) else None,
            "context_support": row.get("context_support"),
            "source": str(self.analysis_dir / "content_semantic_scan.json"),
            "is_measured_ground_truth": False,
        }

    def feature_summary(self, feature_id: int) -> dict[str, Any]:
        try:
            measured = self.features_by_id[feature_id]
        except KeyError as exc:
            raise IndexError(feature_id) from exc
        return {
            **measured,
            "inspection_groups": self.group_membership.get(feature_id, []),
            "semantic_annotation": self._semantic_annotation(feature_id),
            "semantic_triage": self._semantic_triage(feature_id),
            "sources": {
                "measured": f"feature_metrics.json → features[{feature_id}]",
                "inspection_groups": "summary.json → candidate_groups",
                "semantic_annotation": (
                    str(self.annotations_path) if self.annotations_path else None
                ),
            },
        }

    def feature_detail(self, feature_id: int) -> dict[str, Any]:
        summary = self.feature_summary(feature_id)
        raw_examples = self.examples.get(str(feature_id))
        example_source = self.example_sources.get(
            str(feature_id), "crosscoder/top_activation_examples.json"
        )
        examples = (
            [
                activation_example_payload(
                    feature_id,
                    index,
                    example,
                    artifact_name=example_source,
                )
                for index, example in enumerate(raw_examples)
            ]
            if raw_examples is not None
            else None
        )
        if examples:
            message = "Exact measured Crosscoder activation contexts were retained."
        elif examples == []:
            message = "Context extraction ran, but no positive activation context was retained."
        else:
            message = "Contexts were not extracted for this feature; this is not evidence of zero firing."
        return {
            **summary,
            "examples": examples,
            "context": {
                "status": "extracted" if raw_examples is not None else "not_extracted",
                "saved_contexts": len(examples or []),
                "message": message,
                "source": example_source,
            },
        }

    def candidates(self) -> dict[str, Any]:
        ordered: list[int] = []
        for group in ("base-tail", "shared", "finetuned-tail"):
            ordered.extend(self.candidate_groups.get(group, []))
        ordered = list(dict.fromkeys(ordered))
        return {
            "available": self.available,
            "candidate_selection": self.summary.get("candidate_selection"),
            "groups": {
                _display_group(group): ids for group, ids in self.candidate_groups.items()
            },
            "items": [self.feature_summary(feature_id) for feature_id in ordered],
            "source": str(self.analysis_dir / "summary.json"),
        }

    def verification_summary(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "analysis_dir": str(self.analysis_dir),
            "features": len(self.features_by_id),
            "candidate_features": len(self.candidates()["items"]) if self.available else 0,
            "diagnostics": self.diagnostics,
        }


class ResearchComparisonStore:
    """Resolve qualitative relationship IDs against canonical measured method stores."""

    def __init__(
        self,
        comparison_path: str | Path,
        *,
        aspd_store: AnalysisDashboardStore,
        crosscoder_store: CrosscoderDashboardStore | None,
        annotations_path: str | Path | None = None,
    ) -> None:
        self.path = Path(comparison_path).resolve()
        self.aspd_store = aspd_store
        self.crosscoder_store = crosscoder_store
        self.annotations_path = Path(annotations_path).resolve() if annotations_path else None
        self.data, error = _read_object(self.path)
        self.diagnostics = [error] if error else []
        self.annotations: dict[str, Any] = {}
        if self.annotations_path is not None:
            self.annotations, error = _read_object(self.annotations_path)
            if error:
                self.diagnostics.append(error)
        self.aspd_annotations = self._aspd_annotation_index()
        self.phenomena = self._validate_phenomena()
        self.available = bool(self.data and not self.diagnostics)

    def _aspd_annotation_index(self) -> dict[int, dict[str, Any]]:
        rows = self.annotations.get("aspd", [])
        return {
            int(row["feature_id"]): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("feature_id"), int)
        } if isinstance(rows, list) else {}

    def _validate_phenomena(self) -> list[dict[str, Any]]:
        rows = self.data.get("phenomena", [])
        if not isinstance(rows, list):
            self.diagnostics.append("comparison artifact field 'phenomena' must be a list")
            return []
        valid = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                self.diagnostics.append(f"phenomena[{index}] must be an object")
                continue
            status = row.get("status")
            if status not in COMPARISON_STATUSES:
                self.diagnostics.append(f"phenomena[{index}] has unsupported status {status!r}")
                continue
            crosscoder_ids = row.get("crosscoder_features", [])
            aspd_ids = row.get("aspd_components", [])
            if not isinstance(crosscoder_ids, list) or not all(
                isinstance(feature_id, int) for feature_id in crosscoder_ids
            ):
                self.diagnostics.append(f"phenomena[{index}] has invalid Crosscoder references")
                continue
            if not isinstance(aspd_ids, list) or not all(
                isinstance(feature_id, int) for feature_id in aspd_ids
            ):
                self.diagnostics.append(f"phenomena[{index}] has invalid ASPD references")
                continue
            invalid_aspd = [
                feature_id
                for feature_id in aspd_ids
                if self.aspd_store.n_features is None
                or not 0 <= feature_id < self.aspd_store.n_features
            ]
            invalid_crosscoder = (
                [
                    feature_id
                    for feature_id in crosscoder_ids
                    if feature_id not in self.crosscoder_store.features_by_id
                ]
                if self.crosscoder_store is not None
                else crosscoder_ids
            )
            if invalid_aspd or invalid_crosscoder:
                self.diagnostics.append(
                    f"phenomena[{index}] has unresolved references: "
                    f"Crosscoder={invalid_crosscoder}, ASPD={invalid_aspd}"
                )
                continue
            valid.append(row)
        return valid

    def _aspd_evidence(self, component_id: int) -> dict[str, Any]:
        detail = self.aspd_store.feature_detail(component_id)
        annotation = self.aspd_annotations.get(component_id)
        annotation_payload = None
        if annotation is not None:
            metadata = self.annotations.get("metadata", {})
            annotation_payload = {
                "tentative_label": annotation.get("tentative_label"),
                "interpretation_confidence": annotation.get("confidence"),
                "annotation_source": (
                    metadata.get("annotation_source") if isinstance(metadata, dict) else None
                ),
                "warning": annotation.get("warning"),
                "is_measured_ground_truth": False,
                "source": str(self.annotations_path) if self.annotations_path else None,
            }
        return {
            "component_id": component_id,
            "taxonomy": detail["taxonomy"],
            "activation_rho": detail["activation_rho"],
            "mechanism_rho": detail["mechanism_rho"],
            "fire_count": detail["fire_count"],
            "low_support": detail["low_support"],
            "examples": (detail["examples"] or [])[:3] if detail["examples"] is not None else None,
            "semantic_annotation": annotation_payload,
            "sources": detail["sources"],
        }

    def _crosscoder_evidence(self, feature_id: int) -> dict[str, Any]:
        if self.crosscoder_store is None:
            raise IndexError(feature_id)
        detail = self.crosscoder_store.feature_detail(feature_id)
        return {
            "feature_id": feature_id,
            "classification": detail["classification"],
            "inspection_groups": detail["inspection_groups"],
            "relative_decoder_norm": detail["relative_decoder_norm"],
            "fire_count": detail["fire_count"],
            "examples": (detail["examples"] or [])[:3] if detail["examples"] is not None else None,
            "semantic_annotation": detail["semantic_annotation"],
            "sources": detail["sources"],
        }

    def payload(self) -> dict[str, Any]:
        metadata = self.data.get("metadata", {})
        summary = self.data.get("global_summary", {})
        phenomena = []
        for row in self.phenomena:
            phenomena.append(
                {
                    **row,
                    "status_definition": COMPARISON_STATUSES[row["status"]],
                    "crosscoder_evidence": [
                        self._crosscoder_evidence(feature_id)
                        for feature_id in row["crosscoder_features"]
                    ],
                    "aspd_evidence": [
                        self._aspd_evidence(component_id)
                        for component_id in row["aspd_components"]
                    ],
                }
            )
        crosscoder_global = (
            self.crosscoder_store.overview() if self.crosscoder_store is not None else None
        )
        aspd_metadata = self.aspd_store.metadata()
        taxonomy_counts = {
            category: len(feature_ids) if isinstance(feature_ids, list) else None
            for category, feature_ids in self.aspd_store.taxonomy.items()
        }
        return {
            "available": self.available,
            "diagnostics": self.diagnostics,
            "metadata": metadata if isinstance(metadata, dict) else {},
            "global_summary": summary if isinstance(summary, dict) else {},
            "global_measured": {
                "crosscoder": (
                    {
                        "metrics": crosscoder_global["metrics"],
                        "classification_counts": crosscoder_global["classification_counts"],
                        "strict_result": crosscoder_global["strict_result"],
                    }
                    if crosscoder_global is not None
                    else None
                ),
                "aspd": {
                    "taxonomy_counts": taxonomy_counts,
                    "n_features": aspd_metadata["n_features"],
                    "validation_tokens": aspd_metadata["validation_tokens"],
                },
            },
            "phenomena": phenomena,
            "source": str(self.path),
        }

    def related_to_aspd(self, component_id: int) -> list[dict[str, Any]]:
        return [
            {
                "phenomenon_id": row["id"],
                "phenomenon_label": row["label"],
                "status": row["status"],
                "status_definition": COMPARISON_STATUSES[row["status"]],
                "crosscoder_feature_ids": row["crosscoder_features"],
            }
            for row in self.phenomena
            if component_id in row["aspd_components"]
        ]

    def related_to_crosscoder(self, feature_id: int) -> list[dict[str, Any]]:
        return [
            {
                "phenomenon_id": row["id"],
                "phenomenon_label": row["label"],
                "status": row["status"],
                "status_definition": COMPARISON_STATUSES[row["status"]],
                "aspd_component_ids": row["aspd_components"],
            }
            for row in self.phenomena
            if feature_id in row["crosscoder_features"]
        ]

    def verification_summary(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.path),
            "phenomena": len(self.phenomena),
            "diagnostics": self.diagnostics,
        }

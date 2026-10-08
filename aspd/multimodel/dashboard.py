"""FastAPI data layer for the human-reviewed multi-model P1--P5 dashboard.

The store never fabricates a scientific value. Missing tensors become ``None`` plus a diagnostic;
raw analysis files stay read-only, while researcher notes live in a separate JSON file.
"""

from __future__ import annotations

import csv
import io
import json
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from fastapi import Query
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file
from torch import Tensor

from aspd.multimodel.config import MultiModelExperimentConfig, load_experiment_config

CORE_KEYS = (
    "decoder_norm_NC",
    "activation_rho_NC",
    "mechanism_beta_NC",
    "mechanism_rho_NC",
    "fire_count_C",
)
P5_METRICS = (
    "component_cosine",
    "relative_component_change",
    "read_cosine",
    "write_cosine",
)
NOTE_STATUSES = ("unreviewed", "promising", "control", "reject", "P6 candidate")
NOTE_CONFIDENCES = ("low", "medium", "high")


class ResearcherNote(BaseModel):
    """Human interpretation kept deliberately separate from measured tensors."""

    model_config = ConfigDict(extra="forbid")

    tentative_label: str = ""
    notes: str = ""
    semantic_evidence: str = ""
    alternative_interpretation: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
    why_interesting: str = ""
    mentor_notes: str = ""
    candidate_status: Literal[
        "unreviewed", "promising", "control", "reject", "P6 candidate"
    ] = "unreviewed"


class FeatureFilters(BaseModel):
    category: str | None = None
    activation_min: float | None = None
    activation_max: float | None = None
    mechanism_min: float | None = None
    mechanism_max: float | None = None
    beta_min: float | None = None
    fire_min: int | None = None
    component_cosine_min: float | None = None
    component_cosine_max: float | None = None
    read_cosine_min: float | None = None
    read_cosine_max: float | None = None
    write_cosine_min: float | None = None
    write_cosine_max: float | None = None
    relative_change_min: float | None = None
    relative_change_max: float | None = None
    locus_matrix: str | None = None


def _read_json(path: Path, expected: type) -> tuple[Any, str | None]:
    if not path.exists():
        return expected(), f"missing file: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return expected(), f"could not parse {path.name}: {exc}"
    if not isinstance(value, expected):
        return expected(), f"{path.name} must contain a {expected.__name__}"
    return value, None


def _indices(mask: Tensor) -> list[list[int]]:
    """Return every bad tensor index, e.g. a [N,C] mask -> [[n,c], ...]."""

    return mask.nonzero(as_tuple=False).tolist()


def _status(failures: int, warnings: int = 0) -> str:
    return "FAIL" if failures else ("WARN" if warnings else "PASS")


class AnalysisDashboardStore:
    """Read-only tensor store plus separately persisted human notes."""

    def __init__(
        self,
        analysis_dir: str | Path,
        *,
        config_path: str | Path | None = None,
        run_dir: str | Path | None = None,
        notes_path: str | Path | None = None,
        low_support_threshold: int = 32,
    ) -> None:
        self.analysis_dir = Path(analysis_dir).resolve()
        self.run_dir = Path(run_dir).resolve() if run_dir else self.analysis_dir.parent
        self.low_support_threshold = low_support_threshold
        self.startup_diagnostics: list[str] = []

        tensor_path = self.analysis_dir / "posthoc.safetensors"
        if tensor_path.exists():
            try:
                self.tensors: dict[str, Tensor] = load_file(str(tensor_path))
            except Exception as exc:  # safetensors reports format-specific exception types
                self.tensors = {}
                self.startup_diagnostics.append(f"could not parse posthoc.safetensors: {exc}")
        else:
            self.tensors = {}
            self.startup_diagnostics.append("missing file: posthoc.safetensors")

        self.taxonomy, error = _read_json(self.analysis_dir / "taxonomy.json", dict)
        if error:
            self.startup_diagnostics.append(error)
        self.examples, error = _read_json(
            self.analysis_dir / "top_activation_examples.json", dict
        )
        if error:
            self.startup_diagnostics.append(error)

        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config(self.config_path)
        self.provenance, error = _read_json(self.run_dir / "provenance.json", dict)
        if error:
            self.startup_diagnostics.append(error)

        self.n_models, self.n_features = self._discover_dimensions()
        self.model_names = self._discover_model_names()
        self.matrix_names = self._discover_matrix_names()
        self.category_membership = self._build_category_membership()
        self.notes_path = (
            Path(notes_path).resolve()
            if notes_path
            else self.analysis_dir / "researcher_notes.json"
        )
        self._notes_lock = threading.Lock()
        self.notes = self._load_notes()
        self.health = self._build_health()

    def _resolve_config_path(self, supplied: str | Path | None) -> Path | None:
        if supplied is not None:
            return Path(supplied).resolve()
        copied = self.run_dir / "experiment_config.json"
        return copied if copied.exists() else None

    def _load_config(self, path: Path | None) -> MultiModelExperimentConfig | None:
        if path is None:
            self.startup_diagnostics.append("experiment config unavailable")
            return None
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return load_experiment_config(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            return MultiModelExperimentConfig.model_validate(raw)
        except Exception as exc:
            self.startup_diagnostics.append(f"could not parse experiment config: {exc}")
            return None

    def _discover_dimensions(self) -> tuple[int | None, int | None]:
        candidates: list[tuple[int, int]] = []
        for key in ("activation_rho_NC", "mechanism_beta_NC", "mechanism_rho_NC"):
            tensor = self.tensors.get(key)
            if tensor is not None and tensor.ndim == 2:
                candidates.append((int(tensor.shape[0]), int(tensor.shape[1])))
        fire = self.tensors.get("fire_count_C")
        c_from_fire = int(fire.shape[0]) if fire is not None and fire.ndim == 1 else None
        if candidates:
            n, c = Counter(candidates).most_common(1)[0][0]
            if c_from_fire is not None and c_from_fire != c:
                self.startup_diagnostics.append(
                    f"dimension conflict: core N×C tensors imply C={c}, fire_count_C implies {c_from_fire}"
                )
            return n, c
        if self.config is not None:
            return len(self.config.models), self.config.sparsity.n_features
        return None, c_from_fire

    def _discover_model_names(self) -> list[str]:
        if self.config is not None and (
            self.n_models is None or len(self.config.models) == self.n_models
        ):
            return [model.name for model in self.config.models]
        discovered = sorted(
            {
                parts[1]
                for key in self.tensors
                if len(parts := key.split("/")) == 3 and parts[0] in {"beta", "locus"}
            }
        )
        if self.n_models is not None and len(discovered) == self.n_models:
            return discovered
        if self.n_models is not None:
            self.startup_diagnostics.append(
                "model names unavailable from config/tensor keys; generated neutral axis labels"
            )
            return [f"model_{index}" for index in range(self.n_models)]
        return discovered

    def _discover_matrix_names(self) -> list[str]:
        names: set[str] = set()
        for key in self.tensors:
            parts = key.split("/")
            if len(parts) == 3 and parts[0] in {"beta", "locus"}:
                names.add(parts[2])
            elif len(parts) == 3 and parts[0] == "pair":
                names.add(parts[1])
        if self.config is not None:
            names.update(matrix.name for model in self.config.models for matrix in model.matrices)
        return sorted(names)

    def _build_category_membership(self) -> dict[int, list[str]]:
        membership: dict[int, list[str]] = defaultdict(list)
        for category, ids in self.taxonomy.items():
            if not isinstance(ids, list):
                continue
            for feature_id in ids:
                if isinstance(feature_id, int):
                    membership[feature_id].append(str(category))
        return dict(membership)

    def _load_notes(self) -> dict[str, dict[str, str]]:
        if not self.notes_path.exists():
            return {}
        try:
            raw = json.loads(self.notes_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("notes root must be an object")
            validated: dict[str, dict[str, str]] = {}
            for feature_id, note in raw.items():
                validated[str(int(feature_id))] = ResearcherNote.model_validate(note).model_dump()
            return validated
        except Exception as exc:
            self.startup_diagnostics.append(f"could not parse researcher notes: {exc}")
            return {}

    def _shape_check(self) -> dict[str, Any]:
        expected_nc = (
            [self.n_models, self.n_features]
            if self.n_models is not None and self.n_features is not None
            else None
        )
        expected_c = [self.n_features] if self.n_features is not None else None
        rows = []
        failures = 0
        for key in CORE_KEYS:
            tensor = self.tensors.get(key)
            actual = list(tensor.shape) if tensor is not None else None
            expected = expected_c if key == "fire_count_C" else expected_nc
            ok = actual is not None and (expected is None or actual == expected)
            failures += not ok
            rows.append(
                {
                    "key": key,
                    "actual": actual,
                    "expected": expected,
                    "ok": ok,
                    "source": f"posthoc.safetensors → {key}",
                }
            )
        if self.n_features is not None:
            for key, tensor in self.tensors.items():
                if not key.startswith(("beta/", "locus/", "pair/")):
                    continue
                actual = list(tensor.shape)
                expected = [self.n_features]
                ok = actual == expected
                failures += not ok
                rows.append(
                    {
                        "key": key,
                        "actual": actual,
                        "expected": expected,
                        "ok": ok,
                        "source": f"posthoc.safetensors → {key}",
                    }
                )
        return {"name": "Tensor shapes", "status": _status(failures), "rows": rows}

    def _normalization_check(self) -> dict[str, Any]:
        rows = []
        failures = 0
        tolerance = 1e-5
        specs = (
            ("activation_rho_NC", "decoder_norm_NC"),
            ("mechanism_rho_NC", "mechanism_beta_NC"),
        )
        for rho_key, mass_key in specs:
            rho, mass = self.tensors.get(rho_key), self.tensors.get(mass_key)
            if rho is None or mass is None or rho.ndim != 2 or mass.shape != rho.shape:
                failures += 1
                rows.append(
                    {
                        "key": rho_key,
                        "available": False,
                        "diagnostic": f"requires compatible {rho_key} and {mass_key}",
                    }
                )
                continue
            active = mass.float().sum(dim=0) > 0
            errors = (rho.float().sum(dim=0) - 1).abs()
            bad = active & (errors > tolerance)
            failures += int(bad.any())
            rows.append(
                {
                    "key": rho_key,
                    "available": True,
                    "tolerance": tolerance,
                    "max_absolute_error": float(errors[active].max()) if active.any() else None,
                    "failing_components": bad.nonzero().flatten().tolist(),
                    "source": f"posthoc.safetensors → {rho_key}",
                }
            )
        return {"name": "Rho normalization", "status": _status(failures), "rows": rows}

    def _taxonomy_check(self) -> dict[str, Any]:
        all_ids: list[int] = []
        invalid_entries: list[dict[str, Any]] = []
        for category, ids in self.taxonomy.items():
            if not isinstance(ids, list):
                invalid_entries.append({"category": category, "value": ids})
                continue
            for feature_id in ids:
                if not isinstance(feature_id, int) or (
                    self.n_features is not None and not 0 <= feature_id < self.n_features
                ):
                    invalid_entries.append({"category": category, "value": feature_id})
                else:
                    all_ids.append(feature_id)
        counts = Counter(all_ids)
        duplicates = sorted(feature_id for feature_id, count in counts.items() if count > 1)
        missing = (
            sorted(set(range(self.n_features)) - set(all_ids))
            if self.n_features is not None
            else []
        )
        failures = int(bool(invalid_entries or duplicates or missing))
        return {
            "name": "Taxonomy coverage",
            "status": _status(failures),
            "total_listed": len(all_ids),
            "unique_listed": len(counts),
            "missing_feature_ids": missing,
            "duplicated_feature_ids": duplicates,
            "invalid_entries": invalid_entries,
            "source": "taxonomy.json",
        }

    def _locus_check(self) -> dict[str, Any]:
        tolerance = 1e-5
        rows = []
        failures = 0
        beta = self.tensors.get("mechanism_beta_NC")
        for model_index, model_name in enumerate(self.model_names):
            locus_tensors = [
                self.tensors[key]
                for matrix in self.matrix_names
                if (key := f"locus/{model_name}/{matrix}") in self.tensors
            ]
            if (
                beta is None
                or beta.ndim != 2
                or model_index >= beta.shape[0]
                or len(locus_tensors) != len(self.matrix_names)
                or not locus_tensors
            ):
                failures += 1
                rows.append(
                    {
                        "model": model_name,
                        "available": False,
                        "diagnostic": "requires beta plus every discovered locus tensor",
                    }
                )
                continue
            stacked = torch.stack(locus_tensors).float()
            active = beta[model_index].float() > 0
            errors = (stacked.sum(dim=0) - 1).abs()
            bad = active & (errors > tolerance)
            failures += int(bad.any())
            rows.append(
                {
                    "model": model_name,
                    "available": True,
                    "tolerance": tolerance,
                    "max_absolute_error": float(errors[active].max()) if active.any() else None,
                    "failing_components": bad.nonzero().flatten().tolist(),
                    "sources": [
                        f"posthoc.safetensors → locus/{model_name}/{matrix}"
                        for matrix in self.matrix_names
                    ],
                }
            )
        return {"name": "Locus normalization", "status": _status(failures), "rows": rows}

    def _finite_check(self) -> dict[str, Any]:
        affected = []
        for key, tensor in self.tensors.items():
            if not tensor.is_floating_point():
                continue
            bad = ~torch.isfinite(tensor)
            if bad.any():
                affected.append(
                    {
                        "key": key,
                        "indices": _indices(bad),
                        "source": f"posthoc.safetensors → {key}",
                    }
                )
        return {
            "name": "Finite tensor values",
            "status": _status(len(affected)),
            "affected": affected,
        }

    def _example_check(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        for raw_feature_id, examples in self.examples.items():
            try:
                feature_id = int(raw_feature_id)
            except (TypeError, ValueError):
                issues.append({"component": raw_feature_id, "error": "component ID is not integer"})
                continue
            if self.n_features is not None and not 0 <= feature_id < self.n_features:
                issues.append({"component": raw_feature_id, "error": "component ID out of range"})
            if not isinstance(examples, list):
                issues.append({"component": raw_feature_id, "error": "examples value is not a list"})
                continue
            for index, example in enumerate(examples):
                source = f'top_activation_examples.json → "{raw_feature_id}"[{index}]'
                if not isinstance(example, dict):
                    issues.append({"source": source, "error": "example is not an object"})
                    continue
                if not isinstance(example.get("g_s"), int | float):
                    issues.append({"source": source, "error": "g_s is not numeric"})
                token_ids, center = example.get("token_ids"), example.get("center_in_window")
                if token_ids is not None:
                    if not isinstance(token_ids, list):
                        issues.append({"source": source, "error": "token_ids is not a list"})
                    elif not isinstance(center, int) or not 0 <= center < len(token_ids):
                        issues.append(
                            {"source": source, "error": "center_in_window is invalid for token_ids"}
                        )
        return {
            "name": "Activation-example consistency",
            "status": _status(len(issues)),
            "issues": issues,
            "source": "top_activation_examples.json",
        }

    def _build_health(self) -> dict[str, Any]:
        checks = [
            self._shape_check(),
            self._normalization_check(),
            self._taxonomy_check(),
            self._locus_check(),
            self._finite_check(),
            self._example_check(),
        ]
        expected = set(CORE_KEYS)
        expected.update(
            f"locus/{model}/{matrix}"
            for model in self.model_names
            for matrix in self.matrix_names
        )
        expected.update(
            f"beta/{model}/{matrix}"
            for model in self.model_names
            for matrix in self.matrix_names
        )
        if len(self.model_names) == 2:
            expected.update(
                f"pair/{matrix}/{metric}"
                for matrix in self.matrix_names
                for metric in P5_METRICS
            )
        categorized = {
            "core": sorted(key for key in self.tensors if key in CORE_KEYS),
            "beta": sorted(key for key in self.tensors if key.startswith("beta/")),
            "locus": sorted(key for key in self.tensors if key.startswith("locus/")),
            "pair": sorted(key for key in self.tensors if key.startswith("pair/")),
            "other": sorted(
                key
                for key in self.tensors
                if key not in CORE_KEYS
                and not key.startswith(("beta/", "locus/", "pair/"))
            ),
        }
        return {
            "overall_status": (
                "FAIL"
                if any(check["status"] == "FAIL" for check in checks)
                else ("WARN" if self.startup_diagnostics else "PASS")
            ),
            "startup_diagnostics": self.startup_diagnostics,
            "checks": checks,
            "tensor_keys": sorted(self.tensors),
            "categorized_keys": categorized,
            "missing_expected_keys": sorted(expected - set(self.tensors)),
            "tensor_shapes": {key: list(value.shape) for key, value in self.tensors.items()},
        }

    def _tensor_vector(self, key: str, feature_id: int) -> list[float] | None:
        tensor = self.tensors.get(key)
        if tensor is None or tensor.ndim != 2 or feature_id >= tensor.shape[1]:
            return None
        return [float(value) for value in tensor[:, feature_id]]

    def _tensor_scalar(self, key: str, feature_id: int) -> float | None:
        tensor = self.tensors.get(key)
        if tensor is None or tensor.ndim != 1 or feature_id >= tensor.shape[0]:
            return None
        return float(tensor[feature_id])

    def _locus(self, feature_id: int) -> dict[str, dict[str, float | None]]:
        return {
            model: {
                matrix: self._tensor_scalar(f"locus/{model}/{matrix}", feature_id)
                for matrix in self.matrix_names
            }
            for model in self.model_names
        }

    @staticmethod
    def _dominant_locus(values: Mapping[str, float | None]) -> str | None:
        available = {name: value for name, value in values.items() if value is not None}
        return max(available, key=available.get) if available else None  # type: ignore[arg-type]

    def _p5(self, feature_id: int) -> list[dict[str, Any]]:
        return [
            {
                "matrix": matrix,
                **{
                    metric: self._tensor_scalar(f"pair/{matrix}/{metric}", feature_id)
                    for metric in P5_METRICS
                },
                "sources": {
                    metric: f"posthoc.safetensors → pair/{matrix}/{metric}[{feature_id}]"
                    for metric in P5_METRICS
                },
            }
            for matrix in self.matrix_names
        ]

    @staticmethod
    def _available(values: list[float | None] | None) -> list[float]:
        return [value for value in (values or []) if value is not None]

    def feature_summary(self, feature_id: int) -> dict[str, Any]:
        if self.n_features is None or not 0 <= feature_id < self.n_features:
            raise IndexError(feature_id)
        activation = self._tensor_vector("activation_rho_NC", feature_id)
        mechanism = self._tensor_vector("mechanism_rho_NC", feature_id)
        beta = self._tensor_vector("mechanism_beta_NC", feature_id)
        fire_value = self._tensor_scalar("fire_count_C", feature_id)
        locus = self._locus(feature_id)
        dominant = {model: self._dominant_locus(values) for model, values in locus.items()}
        p5 = self._p5(feature_id)
        beta_values = self._available(beta)
        base_activation = activation[0] if activation else None
        base_mechanism = mechanism[0] if mechanism else None
        categories = self.category_membership.get(feature_id, [])
        return {
            "feature_id": feature_id,
            "taxonomy": categories[0] if len(categories) == 1 else None,
            "taxonomy_categories": categories,
            "activation_rho": activation,
            "mechanism_rho": mechanism,
            "beta": beta,
            "beta_total": sum(beta_values) if len(beta_values) == len(self.model_names) else None,
            "fire_count": int(round(fire_value)) if fire_value is not None else None,
            "fire_density": (
                fire_value / self.config.data.validation_tokens
                if fire_value is not None and self.config is not None
                else None
            ),
            "low_support": (
                0 < fire_value < self.low_support_threshold if fire_value is not None else None
            ),
            "dominant_locus": dominant,
            "locus_shift": self._locus_shift(locus),
            "rho_gap": (
                abs(base_activation - base_mechanism)
                if base_activation is not None and base_mechanism is not None
                else None
            ),
            "p5_min_component_cosine": self._aggregate_p5(p5, "component_cosine", min),
            "p5_min_read_cosine": self._aggregate_p5(p5, "read_cosine", min),
            "p5_min_write_cosine": self._aggregate_p5(p5, "write_cosine", min),
            "p5_max_relative_change": self._aggregate_p5(
                p5, "relative_component_change", max
            ),
            "has_examples": str(feature_id) in self.examples,
            "note": self.notes.get(str(feature_id)),
        }

    def _locus_shift(self, locus: Mapping[str, Mapping[str, float | None]]) -> float | None:
        if len(self.model_names) != 2:
            return None
        left, right = (locus.get(name, {}) for name in self.model_names)
        pairs = [(left.get(matrix), right.get(matrix)) for matrix in self.matrix_names]
        if any(a is None or b is None for a, b in pairs):
            return None
        # Transparent derived metric: L1 distance sum_j |pi_base(j)-pi_ft(j)|.
        return sum(abs(float(a) - float(b)) for a, b in pairs if a is not None and b is not None)

    @staticmethod
    def _aggregate_p5(
        p5: list[dict[str, Any]], key: str, operation: Any
    ) -> float | None:
        values = [row[key] for row in p5 if row.get(key) is not None]
        return operation(values) if values else None

    def feature_detail(self, feature_id: int) -> dict[str, Any]:
        summary = self.feature_summary(feature_id)
        decoder = self._tensor_vector("decoder_norm_NC", feature_id)
        locus = self._locus(feature_id)
        examples = self.examples.get(str(feature_id))
        examples_payload = None
        if isinstance(examples, list):
            examples_payload = [
                {
                    **example,
                    "source": f'top_activation_examples.json → "{feature_id}"[{index}]',
                }
                for index, example in enumerate(examples)
                if isinstance(example, dict)
            ]
        dominant = summary["dominant_locus"]
        locus_statement = None
        if len(self.model_names) == 2 and all(dominant.get(name) for name in self.model_names):
            first, second = (dominant[name] for name in self.model_names)
            locus_statement = (
                "Dominant matrix is the same"
                if first == second
                else f"Dominant locus changed: {first} → {second}"
            )
        return {
            **summary,
            "decoder_norm": decoder,
            "locus": locus,
            "p5": self._p5(feature_id),
            "examples": examples_payload,
            "locus_statement": locus_statement,
            "sources": {
                "taxonomy": (
                    f"taxonomy.json → {summary['taxonomy']}" if summary["taxonomy"] else None
                ),
                "decoder_norm": f"posthoc.safetensors → decoder_norm_NC[:,{feature_id}]",
                "activation_rho": f"posthoc.safetensors → activation_rho_NC[:,{feature_id}]",
                "mechanism_beta": f"posthoc.safetensors → mechanism_beta_NC[:,{feature_id}]",
                "mechanism_rho": f"posthoc.safetensors → mechanism_rho_NC[:,{feature_id}]",
                "fire_count": f"posthoc.safetensors → fire_count_C[{feature_id}]",
                "locus": {
                    model: {
                        matrix: f"posthoc.safetensors → locus/{model}/{matrix}[{feature_id}]"
                        for matrix in self.matrix_names
                    }
                    for model in self.model_names
                },
            },
            "missing": self._feature_missing(feature_id),
        }

    def _feature_missing(self, feature_id: int) -> list[str]:
        missing = []
        for key in CORE_KEYS:
            tensor = self.tensors.get(key)
            required_rank = 1 if key == "fire_count_C" else 2
            feature_axis = 0 if required_rank == 1 else 1
            if tensor is None or tensor.ndim != required_rank or feature_id >= tensor.shape[feature_axis]:
                missing.append(f"N/A — data unavailable: posthoc.safetensors key {key!r}")
        for model in self.model_names:
            for matrix in self.matrix_names:
                key = f"locus/{model}/{matrix}"
                if key not in self.tensors:
                    missing.append(f"N/A — data unavailable: posthoc.safetensors key {key!r}")
        return missing

    def metadata(self) -> dict[str, Any]:
        checkpoint = None
        latest = self.run_dir / "latest_checkpoint.txt"
        if latest.exists():
            checkpoint = latest.read_text(encoding="utf-8").strip()
        config = self.config
        return {
            "run_name": config.name if config else None,
            "model_names": self.model_names,
            "model_ids": [model.model_id for model in config.models] if config else None,
            "n_models": self.n_models,
            "n_features": self.n_features,
            "top_k": config.sparsity.top_k if config else None,
            "selected_layers": [model.grounding.module for model in config.models] if config else None,
            "matrix_names": self.matrix_names,
            "validation_tokens": config.data.validation_tokens if config else None,
            "shared_epsilon": config.analysis.shared_epsilon if config else None,
            "concentrated_threshold": config.analysis.concentrated_threshold if config else None,
            "checkpoint": checkpoint,
            "git_commit": self.provenance.get("git_commit"),
            "analysis_dir": str(self.analysis_dir),
            "config_path": str(self.config_path) if self.config_path else None,
            "low_support_threshold": self.low_support_threshold,
            "sources": {
                "experiment": str(self.config_path) if self.config_path else None,
                "provenance": str(self.run_dir / "provenance.json"),
                "checkpoint": str(latest),
            },
        }

    def overview(self) -> dict[str, Any]:
        taxonomy = [
            {
                "category": category,
                "count": len(ids) if isinstance(ids, list) else None,
                "percentage": (
                    100 * len(ids) / self.n_features
                    if isinstance(ids, list) and self.n_features
                    else None
                ),
                "component_ids": ids if isinstance(ids, list) else None,
                "source": f"taxonomy.json → {category}",
            }
            for category, ids in self.taxonomy.items()
        ]
        points = [self.feature_summary(feature_id) for feature_id in range(self.n_features or 0)]
        return {
            "metadata": self.metadata(),
            "taxonomy": taxonomy,
            "points": points,
            "scatter_available": self.n_models == 2,
            "source_templates": {
                "activation_rho": "posthoc.safetensors → activation_rho_NC[{model_index},{feature_id}]",
                "mechanism_rho": "posthoc.safetensors → mechanism_rho_NC[{model_index},{feature_id}]",
                "beta": "posthoc.safetensors → mechanism_beta_NC[{model_index},{feature_id}]",
                "fire_count": "posthoc.safetensors → fire_count_C[{feature_id}]",
            },
        }

    def list_features(
        self,
        filters: FeatureFilters,
        *,
        sort_by: str = "feature_id",
        descending: bool = False,
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        rows = [self.feature_summary(feature_id) for feature_id in range(self.n_features or 0)]
        rows = [row for row in rows if self._matches(row, filters)]
        allowed_sort = set(rows[0]) if rows else {"feature_id"}
        if sort_by not in allowed_sort:
            raise ValueError(f"unsupported sort field {sort_by!r}")
        rows.sort(
            key=lambda row: (row.get(sort_by) is not None, row.get(sort_by)),
            reverse=descending,
        )
        return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset : offset + limit]}

    def _matches(self, row: Mapping[str, Any], filters: FeatureFilters) -> bool:
        if filters.category and filters.category not in row["taxonomy_categories"]:
            return False
        base_activation = row["activation_rho"][0] if row["activation_rho"] else None
        base_mechanism = row["mechanism_rho"][0] if row["mechanism_rho"] else None
        scalar_ranges = (
            (base_activation, filters.activation_min, filters.activation_max),
            (base_mechanism, filters.mechanism_min, filters.mechanism_max),
            (row["p5_min_component_cosine"], filters.component_cosine_min, filters.component_cosine_max),
            (row["p5_min_read_cosine"], filters.read_cosine_min, filters.read_cosine_max),
            (row["p5_min_write_cosine"], filters.write_cosine_min, filters.write_cosine_max),
            (row["p5_max_relative_change"], filters.relative_change_min, filters.relative_change_max),
        )
        for value, minimum, maximum in scalar_ranges:
            if minimum is not None and (value is None or value < minimum):
                return False
            if maximum is not None and (value is None or value > maximum):
                return False
        if filters.beta_min is not None and (
            row["beta_total"] is None or row["beta_total"] < filters.beta_min
        ):
            return False
        if filters.fire_min is not None and (
            row["fire_count"] is None or row["fire_count"] < filters.fire_min
        ):
            return False
        if filters.locus_matrix and filters.locus_matrix not in row["dominant_locus"].values():
            return False
        return True

    def save_note(self, feature_id: int, note: ResearcherNote) -> dict[str, str]:
        self.feature_summary(feature_id)  # validates range without changing measured data
        with self._notes_lock:
            self.notes[str(feature_id)] = note.model_dump()
            self.notes_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.notes_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(self.notes, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.notes_path)
        return self.notes[str(feature_id)]

    def matched_control(self, feature_id: int, attributes: list[str]) -> dict[str, Any]:
        target = self.feature_summary(feature_id)
        candidate_ids = [
            candidate_id
            for candidate_id, categories in self.category_membership.items()
            if "shared_activation_shared_mechanism" in categories and candidate_id != feature_id
        ]
        allowed = {"fire_count", "beta_total", "activation_rho"}
        if not attributes or not set(attributes) <= allowed:
            raise ValueError(f"attributes must be a nonempty subset of {sorted(allowed)}")

        def measured_value(candidate_id: int, attribute: str) -> float | None:
            if attribute == "fire_count":
                return self._tensor_scalar("fire_count_C", candidate_id)
            if attribute == "activation_rho":
                values = self._tensor_vector("activation_rho_NC", candidate_id)
                return values[0] if values else None
            values = self._tensor_vector("mechanism_beta_NC", candidate_id)
            return sum(values) if values and len(values) == len(self.model_names) else None

        target_values = {
            attribute: measured_value(feature_id, attribute) for attribute in attributes
        }
        usable = [
            candidate_id
            for candidate_id in candidate_ids
            if all(measured_value(candidate_id, attribute) is not None for attribute in attributes)
        ]
        if not usable or any(value is None for value in target_values.values()):
            return {
                "match": None,
                "diagnostic": "N/A — data unavailable for requested matching attributes",
            }
        ranges = {}
        for attribute in attributes:
            numeric = [
                value
                for candidate_id in usable
                if (value := measured_value(candidate_id, attribute)) is not None
            ]
            numeric.append(float(target_values[attribute]))  # type: ignore[arg-type]
            ranges[attribute] = max(numeric) - min(numeric)

        def distance(candidate_id: int) -> tuple[float, dict[str, float]]:
            terms = {}
            for attribute in attributes:
                scale = ranges[attribute] or 1.0
                candidate_value = measured_value(candidate_id, attribute)
                terms[attribute] = (
                    float(candidate_value) - float(target_values[attribute])  # type: ignore[arg-type]
                ) / scale
            return math.sqrt(sum(term * term for term in terms.values())), terms

        match_id = min(usable, key=lambda candidate_id: distance(candidate_id)[0])
        match_distance, terms = distance(match_id)
        return {
            "target": target,
            "match": self.feature_summary(match_id),
            "attributes": attributes,
            "distance": match_distance,
            "normalized_deltas": terms,
            "formula": "sqrt(sum(((control-target)/(observed max-observed min))^2))",
            "pool": "shared_activation_shared_mechanism",
        }

    def feature_markdown(self, feature_id: int) -> str:
        feature = self.feature_detail(feature_id)
        note = ResearcherNote.model_validate(
            self.notes.get(str(feature_id), ResearcherNote().model_dump())
        )
        lines = [
            f"# Feature {feature_id}\n",
            "## Researcher Interpretation",
            f"- Tentative label: {note.tentative_label or 'N/A — not entered'}",
            f"- Notes: {note.notes or 'N/A — not entered'}",
            f"- Confidence: {note.confidence}",
            f"- Candidate status: {note.candidate_status}",
            f"- Semantic evidence: {note.semantic_evidence or 'N/A — not entered'}",
            f"- Alternative interpretation: {note.alternative_interpretation or 'N/A — not entered'}",
            f"- Why interesting: {note.why_interesting or 'N/A — not entered'}",
            f"- Mentor notes: {note.mentor_notes or 'N/A — not entered'}\n",
            "## P1 — Activation Representation",
            f"- Decoder norms: {feature['decoder_norm']}",
            f"- Activation rho: {feature['activation_rho']}",
            f"- Source: {feature['sources']['activation_rho']}\n",
            "## P2 — Parameter Mechanism Strength",
            f"- Beta: {feature['beta']}",
            f"- Beta total: {feature['beta_total']}",
            f"- Mechanism rho: {feature['mechanism_rho']}",
            f"- Fire count: {feature['fire_count']}",
            f"- Fire density: {feature['fire_density']}\n",
            "## P3 — Where the Mechanism Lives",
        ]
        for model, loci in feature["locus"].items():
            lines.append(f"- {model}: {loci}")
        lines.extend(["", "## P5 — How the Parameter Mechanism Changed"])
        for row in feature["p5"]:
            lines.append(f"- {row['matrix']}: " + ", ".join(f"{key}={row[key]}" for key in P5_METRICS))
        lines.extend(["", "## Top Activation Examples"])
        if feature["examples"] is None:
            lines.append("N/A — top_activation_examples.json has no key for this component.")
        elif not feature["examples"]:
            lines.append("No retained activation examples for this component in the analyzed context set.")
        else:
            for example in sorted(feature["examples"], key=lambda item: item.get("g_s", 0), reverse=True):
                lines.append(
                    f"- center={example.get('center_token', 'N/A')}, g_s={example.get('g_s', 'N/A')}: "
                    f"{str(example.get('text', 'N/A — data unavailable')).replace(chr(10), ' ')}"
                )
        return "\n".join(lines) + "\n"

    def features_csv(self, filters: FeatureFilters) -> str:
        rows = self.list_features(filters, limit=self.n_features or 0)["items"]
        fields = [
            "feature_id",
            "taxonomy",
            "activation_rho",
            "mechanism_rho",
            "beta",
            "beta_total",
            "fire_count",
            "fire_density",
            "dominant_locus",
            "rho_gap",
            "locus_shift",
            "p5_min_component_cosine",
            "p5_min_read_cosine",
            "p5_min_write_cosine",
            "p5_max_relative_change",
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        return output.getvalue()

    def verification_summary(self) -> dict[str, Any]:
        return {
            "components_discovered": self.n_features,
            "models_discovered": self.n_models,
            "model_names": self.model_names,
            "tensor_keys_discovered": sorted(self.tensors),
            "matrix_names_discovered": self.matrix_names,
            "taxonomy_category_counts": {
                category: len(ids) if isinstance(ids, list) else None
                for category, ids in self.taxonomy.items()
            },
            "components_with_activation_examples": sum(
                isinstance(value, list) and bool(value) for value in self.examples.values()
            ),
            "validation_status": self.health["overall_status"],
            "validation_warnings": self.startup_diagnostics,
        }


def build_dashboard_app(store: AnalysisDashboardStore):
    """Create the API and serve a built React bundle when one is available."""

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

    app = FastAPI(title="Multi-model ASPD P1–P5 Research Dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/verification")
    async def verification() -> dict[str, Any]:
        return store.verification_summary()

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return store.health

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        return store.overview()

    @app.get("/api/features")
    async def features(
        category: str | None = None,
        activation_min: float | None = None,
        activation_max: float | None = None,
        mechanism_min: float | None = None,
        mechanism_max: float | None = None,
        beta_min: float | None = None,
        fire_min: int | None = None,
        component_cosine_min: float | None = None,
        component_cosine_max: float | None = None,
        read_cosine_min: float | None = None,
        read_cosine_max: float | None = None,
        write_cosine_min: float | None = None,
        write_cosine_max: float | None = None,
        relative_change_min: float | None = None,
        relative_change_max: float | None = None,
        locus_matrix: str | None = None,
        sort_by: str = "feature_id",
        descending: bool = False,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=20_000)] = 200,
    ) -> dict[str, Any]:
        filters = FeatureFilters(
            category=category,
            activation_min=activation_min,
            activation_max=activation_max,
            mechanism_min=mechanism_min,
            mechanism_max=mechanism_max,
            beta_min=beta_min,
            fire_min=fire_min,
            component_cosine_min=component_cosine_min,
            component_cosine_max=component_cosine_max,
            read_cosine_min=read_cosine_min,
            read_cosine_max=read_cosine_max,
            write_cosine_min=write_cosine_min,
            write_cosine_max=write_cosine_max,
            relative_change_min=relative_change_min,
            relative_change_max=relative_change_max,
            locus_matrix=locus_matrix,
        )
        try:
            return store.list_features(
                filters,
                sort_by=sort_by,
                descending=descending,
                offset=offset,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/features/{feature_id}")
    async def feature(feature_id: int) -> dict[str, Any]:
        try:
            return store.feature_detail(feature_id)
        except IndexError as exc:
            raise HTTPException(404, f"feature {feature_id} is out of range") from exc

    @app.get("/api/features/{feature_id}/matched-control")
    async def matched_control(
        feature_id: int,
        attributes: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        try:
            return store.matched_control(
                feature_id, attributes or ["fire_count", "beta_total", "activation_rho"]
            )
        except IndexError as exc:
            raise HTTPException(404, f"feature {feature_id} is out of range") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/compare")
    async def compare(ids: Annotated[list[int], Query()]) -> dict[str, Any]:
        if not 2 <= len(ids) <= 5:
            raise HTTPException(400, "comparison requires 2–5 component IDs")
        try:
            return {"items": [store.feature_detail(feature_id) for feature_id in ids]}
        except IndexError as exc:
            raise HTTPException(404, f"feature {exc.args[0]} is out of range") from exc

    @app.get("/api/notes")
    async def notes() -> dict[str, Any]:
        return {"notes": store.notes, "path": str(store.notes_path)}

    @app.put("/api/notes/{feature_id}")
    async def save_note(feature_id: int, note: ResearcherNote) -> dict[str, Any]:
        try:
            return {"feature_id": feature_id, "note": store.save_note(feature_id, note)}
        except IndexError as exc:
            raise HTTPException(404, f"feature {feature_id} is out of range") from exc

    @app.get("/api/export/feature/{feature_id}.md", response_class=PlainTextResponse)
    async def export_feature(feature_id: int) -> PlainTextResponse:
        try:
            return PlainTextResponse(
                store.feature_markdown(feature_id),
                headers={"Content-Disposition": f'attachment; filename="feature_{feature_id}.md"'},
            )
        except IndexError as exc:
            raise HTTPException(404, f"feature {feature_id} is out of range") from exc

    @app.get("/api/export/features.csv", response_class=PlainTextResponse)
    async def export_features(
        category: str | None = None, fire_min: int | None = None
    ) -> PlainTextResponse:
        return PlainTextResponse(
            store.features_csv(FeatureFilters(category=category, fire_min=fire_min)),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="aspd_features.csv"'},
        )

    @app.get("/api/export/selected.json")
    async def export_selected(ids: Annotated[list[int], Query()]) -> JSONResponse:
        try:
            payload = [
                {
                    "feature": store.feature_summary(feature_id),
                    "researcher_note": store.notes.get(str(feature_id)),
                }
                for feature_id in ids
            ]
        except IndexError as exc:
            raise HTTPException(404, f"feature {exc.args[0]} is out of range") from exc
        return JSONResponse(
            payload,
            headers={"Content-Disposition": 'attachment; filename="selected_candidates.json"'},
        )

    dist = Path(__file__).resolve().parents[2] / "multimodel_dashboard" / "dist"
    if dist.exists():
        index = dist / "index.html"

        @app.get("/{full_path:path}")
        async def static_app(full_path: str) -> FileResponse:
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app

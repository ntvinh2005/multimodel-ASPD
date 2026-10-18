"""Read-only store for reduced-scale Delta-Crosscoder comparison artifacts.

The store intentionally serves missing artifacts as ``available: false``.  It never
creates placeholder measurements, which makes the frontend safe to use before the
LMSYS-gated positive-control run has completed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing file: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not parse {path.name}: {exc}"
    if not isinstance(value, dict):
        return {}, f"{path.name} must contain an object"
    return value, None


class DeltaDashboardStore:
    """Expose Delta overview, candidate rows and exact saved contexts."""

    def __init__(self, analysis_dir: str | Path) -> None:
        self.analysis_dir = Path(analysis_dir).resolve()
        self.diagnostics: list[str] = []
        self.summary, error = _read(self.analysis_dir / "summary.json")
        if error:
            self.diagnostics.append(error)
        metrics, error = _read(self.analysis_dir / "feature_metrics.json")
        if error:
            self.diagnostics.append(error)
        raw = metrics.get("features", [])
        self.features = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
        self.features_by_id = {
            int(row["feature_id"]): row for row in self.features if isinstance(row.get("feature_id"), int)
        }
        examples, error = _read(self.analysis_dir / "top_activation_examples.json")
        if error:
            self.diagnostics.append(error)
        raw_examples = examples.get("features", {})
        self.examples = raw_examples if isinstance(raw_examples, dict) else {}
        self.correspondence, error = _read(self.analysis_dir / "delta_aspd_correspondence.json")
        if error and (self.analysis_dir / "delta_aspd_correspondence.json").exists():
            self.diagnostics.append(error)
        self.behavioral, error = _read(self.analysis_dir / "behavioral_activation_metrics.json")
        if error and (self.analysis_dir / "behavioral_activation_metrics.json").exists():
            self.diagnostics.append(error)
        behavioral_rows = self.behavioral.get("features", [])
        self.behavioral_by_id = {int(row["feature_id"]): row for row in behavioral_rows if isinstance(row, dict) and isinstance(row.get("feature_id"), int)} if isinstance(behavioral_rows, list) else {}
        self.behavioral_contexts, error = _read(self.analysis_dir / "behavioral_top_contexts.json")
        if error and (self.analysis_dir / "behavioral_top_contexts.json").exists():
            self.diagnostics.append(error)
        self.available = bool(self.summary and self.features_by_id)

    def verification_summary(self) -> dict[str, Any]:
        return {"available": self.available, "analysis_dir": str(self.analysis_dir),
                "components": self.summary.get("n_features"), "features_loaded": len(self.features_by_id),
                "diagnostics": self.diagnostics}

    def overview(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "diagnostics": self.diagnostics, "features": [], "candidate_groups": {}}
        return {"available": True, "diagnostics": self.diagnostics, "metadata": self.summary.get("metadata", {}),
                "metrics": self.summary.get("metrics", {}), "n_features": self.summary.get("n_features", len(self.features_by_id)),
                "candidate_groups": self.summary.get("candidate_groups", {}),
                "partition_counts": self.summary.get("partition_counts", {}),
                "behavioral": {"available": bool(self.behavioral_by_id), "metadata": self.behavioral.get("metadata", {}),
                               "features_measured": len(self.behavioral_by_id)}}

    def candidates(self, *, partition: str | None = None, min_fire: int = 0,
                   limit: int = 200, offset: int = 0) -> dict[str, Any]:
        rows = [{**row, **self.behavioral_by_id.get(int(row.get("feature_id", -1)), {})} for row in self.features if (partition is None or row.get("partition") == partition)
                and int(row.get("validation_activation_count", row.get("fire_count", 0))) >= min_fire]
        rows.sort(key=lambda row: (-int(row.get("validation_activation_count", row.get("fire_count", 0))), int(row.get("feature_id", 0))))
        return {"available": self.available, "items": rows[offset: offset + limit], "total": len(rows),
                "groups": self.summary.get("candidate_groups", {}), "diagnostics": self.diagnostics}

    def feature_detail(self, feature_id: int) -> dict[str, Any]:
        row = self.features_by_id.get(feature_id)
        if row is None:
            raise IndexError(feature_id)
        behavioral = self.behavioral_by_id.get(feature_id, {})
        examples = self.examples.get(str(feature_id), [])
        behavioral_examples = self.behavioral_contexts.get("features", {}).get(str(feature_id), {}) if isinstance(self.behavioral_contexts.get("features", {}), dict) else {}
        return {"available": True, "feature": {**row, **behavioral}, "examples": examples,
                "behavioral_examples": behavioral_examples,
                "sources": {"metrics": str(self.analysis_dir / "feature_metrics.json"),
                            "examples": str(self.analysis_dir / "top_activation_examples.json"),
                            "behavioral_metrics": str(self.analysis_dir / "behavioral_activation_metrics.json")}}

    def comparison(self) -> dict[str, Any]:
        if not self.correspondence:
            return {"available": False, "diagnostics": ["Delta↔ASPD correspondence artifact is unavailable."], "matches": []}
        return self.correspondence

    def matches(self, feature_id: int) -> dict[str, Any]:
        payload = self.comparison()
        if not payload.get("available", False):
            return payload | {"feature_id": feature_id, "matches": []}
        matches = [row for row in payload.get("matches", []) if isinstance(row, dict) and row.get("delta_feature_id") == feature_id]
        return {"available": True, "feature_id": feature_id, "matches": matches, "diagnostics": payload.get("diagnostics", [])}

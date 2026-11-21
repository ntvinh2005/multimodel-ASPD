"""Post-hoc analysis configuration (YAML) with fail-closed researcher decisions.

Every value the formulation leaves open is a required field without a default.  A
``null`` value blocks only the stages that need it, so ``importance`` can run before
``K_rep``, ``tau``, ... are decided:

==================  =========================================================
stage               required decisions
==================  =========================================================
importance          ``anchors.importance_pools`` + backend options
representatives     ``representatives.{screening_metric,k_rep,tau_rep}``
edges               ``anchors.edge_pools``, ``edges.*``
compare             ``parents.{k_par,tau}``
==================  =========================================================

See ``configs/multimodel/posthoc/`` for a runnable toy configuration and the blocked
v3.1 template, and ``docs/multimodel/posthoc_analysis.md`` for the meaning of each field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aspd.multimodel.posthoc.ids import GateId
from aspd.multimodel.posthoc.parents import ParentRule
from aspd.multimodel.posthoc.representatives import RepresentativeRule

STAGES: tuple[str, ...] = ("importance", "representatives", "edges", "compare")
BACKENDS: tuple[str, ...] = ("toy", "v31_checkpoint")


class PosthocBlocked(RuntimeError):
    """Raised when a stage is requested while a required decision is still null."""


@dataclass(frozen=True)
class DownstreamGateSelection:
    mode: str | None  # "explicit" or "most_frequent"
    gates: tuple[str, ...] = ()  # explicit: e.g. ("pre/12", "mid/7")
    count: int | None = None  # most_frequent: number of gates per site

    def resolve(self, fire_counts: dict[str, Any]) -> tuple[GateId, ...]:
        """Return the analysed downstream gates.

        ``fire_counts[site]`` is the ``[C]`` firing count from the importance stage.
        ``most_frequent`` takes the ``count`` most frequently firing gates of each site
        (ties by feature index); never-firing gates are never selected.
        """

        if self.mode == "explicit":
            return tuple(sorted(GateId.from_key(key) for key in self.gates))
        if self.mode == "most_frequent":
            selected: list[GateId] = []
            for site in sorted(fire_counts):
                counts = fire_counts[site]
                order = sorted(range(len(counts)), key=lambda c: (-int(counts[c]), c))
                selected.extend(
                    GateId(site, c) for c in order[: self.count] if int(counts[c]) > 0
                )
            return tuple(sorted(selected))
        raise PosthocBlocked(f"unknown downstream gate selection mode {self.mode!r}")


@dataclass(frozen=True)
class PosthocConfig:
    schema_version: int
    experiment_id: str
    backend_kind: str
    backend_options: dict[str, Any]
    importance_pools: int | None
    edge_pools: int | None
    screening_metric: str | None
    k_rep: int | None
    tau_rep: float | None
    downstream_gates: DownstreamGateSelection
    max_anchors_per_gate: int | None
    anchors_per_backward: int | None
    ablation_check_records: int | None
    k_par: int | None
    tau: float | None
    model_pairs: tuple[tuple[str, str], ...] | None = None
    source_path: str | None = field(default=None, compare=False)
    source_sha256: str | None = field(default=None, compare=False)

    # ------------------------------------------------------------------ readiness
    def blockers(self, stage: str) -> list[str]:
        """Null decisions that block ``stage`` (stages include their prerequisites'
        decisions only when they recompute them; artifacts are reused otherwise)."""

        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        required: dict[str, Any] = {}
        if stage == "importance":
            required["anchors.importance_pools"] = self.importance_pools
        if stage == "representatives":
            required.update(
                {
                    "representatives.screening_metric": self.screening_metric,
                    "representatives.k_rep": self.k_rep,
                    "representatives.tau_rep": self.tau_rep,
                }
            )
        if stage == "edges":
            required.update(
                {
                    "anchors.importance_pools": self.importance_pools,
                    "anchors.edge_pools": self.edge_pools,
                    "edges.downstream_gates.mode": self.downstream_gates.mode,
                    "edges.max_anchors_per_gate": self.max_anchors_per_gate,
                    "edges.anchors_per_backward": self.anchors_per_backward,
                    "edges.ablation_check_records": self.ablation_check_records,
                }
            )
            if self.downstream_gates.mode == "most_frequent":
                required["edges.downstream_gates.count"] = self.downstream_gates.count
            if self.downstream_gates.mode == "explicit" and not self.downstream_gates.gates:
                required["edges.downstream_gates.gates"] = None
        if stage == "compare":
            required.update({"parents.k_par": self.k_par, "parents.tau": self.tau})
        missing = [name for name, value in required.items() if value is None]
        if stage in ("importance", "edges"):
            missing.extend(
                f"backend.options.{name}"
                for name in required_backend_options(self.backend_kind)
                if self.backend_options.get(name) is None
            )
        return missing

    def require_ready(self, stage: str) -> None:
        missing = self.blockers(stage)
        if missing:
            raise PosthocBlocked(
                f"stage {stage!r} is blocked by unresolved decisions: {', '.join(missing)}"
            )

    # ------------------------------------------------------------------ typed rules
    def representative_rule(self) -> RepresentativeRule:
        self.require_ready("representatives")
        return RepresentativeRule(self.screening_metric, int(self.k_rep), float(self.tau_rep))  # type: ignore[arg-type]

    def parent_rule(self) -> ParentRule:
        self.require_ready("compare")
        return ParentRule(int(self.k_par), float(self.tau))  # type: ignore[arg-type]

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("source_path")
        payload.pop("source_sha256")
        blob = json.dumps(payload, sort_keys=True, default=list).encode()
        return hashlib.sha256(blob).hexdigest()


def required_backend_options(kind: str) -> tuple[str, ...]:
    if kind == "toy":
        return ("seed",)
    if kind == "v31_checkpoint":
        return (
            "training_config",
            "checkpoint",
            "config_file_sha256",
            "dataset_manifest_sha256",
            "split",
            "source_dtype",
            "device",
        )
    raise ValueError(f"unknown backend kind {kind!r}; expected one of {BACKENDS}")


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _optional(section: dict[str, Any], key: str, cast: type) -> Any:
    if key not in section:
        raise ValueError(f"config key {key!r} must be present (use null if unresolved)")
    value = section[key]
    return None if value is None else cast(value)


def load_config(path: str | Path) -> PosthocConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("post-hoc config must be a mapping with schema_version: 1")
    backend = _section(raw, "backend")
    anchors = _section(raw, "anchors")
    reps = _section(raw, "representatives")
    edges = _section(raw, "edges")
    parents = _section(raw, "parents")
    comparison = raw.get("comparison") or {}
    selection = edges.get("downstream_gates") or {}
    pairs = comparison.get("model_pairs")
    config = PosthocConfig(
        schema_version=1,
        experiment_id=str(raw["experiment_id"]),
        backend_kind=str(backend["kind"]),
        backend_options=dict(backend.get("options") or {}),
        importance_pools=_optional(anchors, "importance_pools", int),
        edge_pools=_optional(anchors, "edge_pools", int),
        screening_metric=_optional(reps, "screening_metric", str),
        k_rep=_optional(reps, "k_rep", int),
        tau_rep=_optional(reps, "tau_rep", float),
        downstream_gates=DownstreamGateSelection(
            mode=selection.get("mode"),
            gates=tuple(selection.get("gates") or ()),
            count=None if selection.get("count") is None else int(selection["count"]),
        ),
        max_anchors_per_gate=_optional(edges, "max_anchors_per_gate", int),
        anchors_per_backward=_optional(edges, "anchors_per_backward", int),
        ablation_check_records=_optional(edges, "ablation_check_records", int),
        k_par=_optional(parents, "k_par", int),
        tau=_optional(parents, "tau", float),
        model_pairs=None if pairs is None else tuple(tuple(pair) for pair in pairs),
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    required_backend_options(config.backend_kind)  # validates the backend kind
    return config

"""Post-hoc analysis configuration (YAML) with fail-closed researcher decisions.

Every value the formulation leaves open (including the edge-score conventions and the
anchor-sampling rules, see ``EDGE_CONVENTIONS``) is a required field without a default.  A
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

#: Edge-score conventions the formulation leaves open.  Each must be set explicitly;
#: the tuple lists the values that are implemented (the first entry of each was the
#: implementation's original, undeclared choice).
EDGE_CONVENTIONS: dict[str, tuple[str, ...]] = {
    # d e_a / d e_b through the true source weights (not a replacement model)
    "jacobian_path": ("native_source_model",),
    # g_a stays at its natural value when b is removed (encoder is not re-run)
    "gates_under_ablation": ("frozen_natural",),
    # e_b at the anchor's own token only (no attention paths from earlier tokens)
    "upstream_positions": ("same_anchor_token",),
    # U^{(n)}(a): every selected matrix of the same model earlier in compute order
    "upstream_set": ("all_selected_upstream_in_compute_order",),
    # total effect (indirect paths through intermediate components included)
    "effect": ("total",),
}
#: Which firing anchors are analysed when a gate fires more often than its quota.
ANCHOR_SAMPLING: tuple[str, ...] = ("first_in_pool", "seeded_random")
#: Whether edge anchors are disjoint from the anchors that define beta/sigma.
EDGE_ANCHOR_SOURCES: tuple[str, ...] = ("disjoint_after_importance", "same_as_importance")


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
    #: ``"all"`` (every unordered pair in model order) or explicit pairs; null blocks compare.
    model_pairs: str | tuple[tuple[str, str], ...] | None
    edge_conventions: dict[str, str | None]
    anchor_sampling: str | None
    anchor_seed: int | None
    edge_anchor_source: str | None
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
            required.update(
                {f"edges.conventions.{k}": v for k, v in self.edge_conventions.items()}
            )
            required["edges.anchor_sampling"] = self.anchor_sampling
            if self.anchor_sampling == "seeded_random":
                required["edges.anchor_seed"] = self.anchor_seed
            required["anchors.edge_anchor_source"] = self.edge_anchor_source
        if stage == "compare":
            required.update({
                "parents.k_par": self.k_par,
                "parents.tau": self.tau,
                "comparison.model_pairs": self.model_pairs,
            })
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

    def pairs(self) -> tuple[tuple[str, str], ...] | None:
        """Model pairs for ``compare_all_pairs`` (``None`` means all pairs)."""

        self.require_ready("compare")
        return None if self.model_pairs == "all" else self.model_pairs  # type: ignore[return-value]

    def edge_skip(self) -> int:
        """Index of the first pool used by the edge stage."""

        self.require_ready("edges")
        if self.edge_anchor_source == "same_as_importance":
            return 0
        return int(self.importance_pools)  # type: ignore[arg-type]

    def recorded_conventions(self) -> dict[str, Any]:
        return {
            **self.edge_conventions,
            "anchor_sampling": self.anchor_sampling,
            "anchor_seed": self.anchor_seed,
            "edge_anchor_source": self.edge_anchor_source,
        }

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
    if "model_pairs" not in comparison:
        raise ValueError("config key 'model_pairs' must be present (use null if unresolved)")
    pairs = comparison["model_pairs"]
    if pairs is not None and pairs != "all":
        pairs = tuple(tuple(pair) for pair in pairs)
    conventions_raw = edges.get("conventions")
    if not isinstance(conventions_raw, dict):
        raise ValueError("config section 'edges.conventions' must be a mapping")
    conventions: dict[str, str | None] = {}
    for name, implemented in EDGE_CONVENTIONS.items():
        if name not in conventions_raw:
            raise ValueError(f"config key 'edges.conventions.{name}' must be present "
                             "(use null if unresolved)")
        value = conventions_raw[name]
        if value is not None and value not in implemented:
            raise ValueError(f"edges.conventions.{name}={value!r} is not implemented; "
                             f"implemented: {implemented}")
        conventions[name] = value
    unknown = set(conventions_raw) - set(EDGE_CONVENTIONS)
    if unknown:
        raise ValueError(f"unknown edge conventions: {sorted(unknown)}")
    sampling = _optional(edges, "anchor_sampling", str)
    if sampling is not None and sampling not in ANCHOR_SAMPLING:
        raise ValueError(f"edges.anchor_sampling must be one of {ANCHOR_SAMPLING}")
    source = _optional(anchors, "edge_anchor_source", str)
    if source is not None and source not in EDGE_ANCHOR_SOURCES:
        raise ValueError(f"anchors.edge_anchor_source must be one of {EDGE_ANCHOR_SOURCES}")
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
        model_pairs=pairs,
        edge_conventions=conventions,
        anchor_sampling=sampling,
        anchor_seed=None if edges.get("anchor_seed") is None else int(edges["anchor_seed"]),
        edge_anchor_source=source,
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    required_backend_options(config.backend_kind)  # validates the backend kind
    return config

"""Immutable researcher authorization for ASPD v3.1 controlled continuations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Main500MContinuationAuthorization(BaseModel):
    """One researcher decision pinned to one complete recovery checkpoint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    status: Literal["APPROVED"]
    decision: Literal["CONTINUE", "CONTINUE_TO_FINAL"]
    policy_version: Literal["v1"]
    reasons: tuple[Literal["HEALTH", "MILESTONE"], ...]
    accepted_exposures: int = Field(gt=0)
    approved_through_exposures: int = Field(gt=0)
    execution_contract_fingerprint: str
    checkpoint_manifest_sha256: str
    researcher: str = Field(min_length=1)
    decided_at: str = Field(min_length=1)
    scientific_limitations_acknowledged: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> Main500MContinuationAuthorization:
        canonical_reasons = tuple(
            reason for reason in ("HEALTH", "MILESTONE") if reason in self.reasons
        )
        if not canonical_reasons or self.reasons != canonical_reasons:
            raise ValueError("continuation reasons must be unique and canonically ordered")
        for name in ("execution_contract_fingerprint", "checkpoint_manifest_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.approved_through_exposures <= self.accepted_exposures:
            raise ValueError("continuation approval must advance beyond the paused exposure")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def next_review_exposure(
    accepted_exposures: int,
    staged_milestones: tuple[int, ...],
) -> int:
    for milestone in staged_milestones:
        if milestone > accepted_exposures:
            return milestone
    raise ValueError("no continuation is permitted beyond the final 500M milestone")


def milestone_diagnostic_due(
    accepted_exposures_after_step: int,
    staged_milestones: tuple[int, ...],
) -> bool:
    return accepted_exposures_after_step in staged_milestones


def milestone_pause_due(
    accepted_exposures_after_step: int,
    staged_milestones: tuple[int, ...],
) -> bool:
    """Pause at review milestones but let the final 500M boundary complete normally."""

    return accepted_exposures_after_step in staged_milestones[:-1]


def continuation_authorized_through(
    consumed_authorizations: list[dict[str, Any]],
    accepted_exposures: int,
) -> bool:
    """Return whether an immutable consumed decision covers the exposure boundary."""

    return any(
        record.get("decision") == "CONTINUE_TO_FINAL"
        and int(record.get("approved_through_exposures", 0)) >= accepted_exposures
        for record in consumed_authorizations
    )


def milestone_scientific_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Preserve the exact milestone diagnostics plus compact reconstruction summaries."""

    losses = {str(key): float(value) for key, value in metrics.get("loss", {}).items()}
    activation = {
        key: value
        for key, value in losses.items()
        if key.startswith("diagnostic/activation/") and key.endswith("/full_fvu")
    }
    internal = {
        key: value for key, value in losses.items() if key.startswith("loss/internal/")
    }
    gate = {
        key: value for key, value in losses.items() if key.startswith("diagnostic/gate/")
    }

    def summarize(values: dict[str, float]) -> dict[str, float | int | None]:
        observed = list(values.values())
        return {
            "count": len(observed),
            "mean": None if not observed else sum(observed) / len(observed),
            "maximum": None if not observed else max(observed),
        }

    return {
        "diagnostic_parameter_step": metrics.get("diagnostic_parameter_step"),
        "accepted_exposures": int(metrics["accepted_exposures"]),
        "ordered_anchor_ids_sha256": metrics["ordered_anchor_ids_sha256"],
        "health": metrics.get("health"),
        "gate_metrics": gate,
        "activation_full_fvu": activation,
        "activation_full_fvu_summary": summarize(activation),
        "internal_fvu": internal,
        "internal_fvu_summary": summarize(internal),
    }


def load_continuation_authorization(
    path: Path,
    *,
    authorization_root: Path,
    pending_review: dict[str, Any],
    execution_contract_fingerprint: str,
    checkpoint_manifest_sha256: str,
    approved_through_exposures: int,
    final_approved_through_exposures: int,
) -> dict[str, Any]:
    """Validate an explicit decision without mutating the training configuration."""

    resolved_path = path.resolve()
    resolved_root = authorization_root.resolve()
    if resolved_path.parent != resolved_root or resolved_path.suffix != ".json":
        raise ValueError(
            "continuation authorization must be one JSON file directly under the configured root"
        )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"continuation authorization is missing: {resolved_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    authorization = Main500MContinuationAuthorization.model_validate(payload)
    expected_approval_boundary = (
        final_approved_through_exposures
        if authorization.decision == "CONTINUE_TO_FINAL"
        else approved_through_exposures
    )
    expected = {
        "policy_version": pending_review["policy_version"],
        "reasons": tuple(pending_review["reasons"]),
        "accepted_exposures": int(pending_review["accepted_exposures"]),
        "approved_through_exposures": int(expected_approval_boundary),
        "execution_contract_fingerprint": execution_contract_fingerprint,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
    }
    observed = authorization.model_dump(mode="python")
    mismatch = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatch:
        raise ValueError(f"continuation authorization provenance mismatch: {mismatch}")
    return {
        **authorization.model_dump(mode="json"),
        "artifact_path": str(resolved_path),
        "artifact_sha256": file_sha256(resolved_path),
    }

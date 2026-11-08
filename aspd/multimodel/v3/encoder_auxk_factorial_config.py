"""Fail-closed contract for the ASPD v3 encoder/AuxK 2x2 screening pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.granite_normalized_config import (
    GraniteNormalizedTargetConfig,
    load_granite_normalized_target_config,
)

EXPERIMENT_FAMILY = "aspd_v3_granite_encoder_auxk_factorial_v1"
SOURCE_CONFIG = "granite_normalized_target_unconstrained_decoder_reliability_pilot_v1.yaml"
SOURCE_CONFIG_SHA256 = "6fbbace8188cb0267b36f8ef99f9b7b3345267f0f93b8ba6ff2077e5b0b631a4"
SOURCE_CONFIG_FINGERPRINT = "04599059ec494ebe485d5e2a17780eb983a049404b2bac552f3b2c55e886dc6f"
ARM_SETTINGS = {
    "transformer_current_auxk": ("hierarchical_transformer", "post_relu", None),
    "linear_current_auxk": ("hierarchical_linear", "post_relu", None),
    "transformer_revival_auxk": ("hierarchical_transformer", "softplus_raw_logits", 1.0),
    "linear_revival_auxk": ("hierarchical_linear", "softplus_raw_logits", 1.0),
    "linear_current_auxk_confirmation": ("hierarchical_linear", "post_relu", None),
}


class FactorialRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: str
    checkpoint_root: str
    preflight_evidence: str


class FactorialConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_granite_encoder_auxk_factorial_v1"]
    scientific_status: Literal[
        "RESEARCHER_AUTHORIZED_256_STEP_FACTORIAL_SCREEN",
        "RESEARCHER_AUTHORIZED_1221_STEP_LINEAR_CONFIRMATION",
    ]
    authorized_by: Literal["researcher"]
    authorized_on: Literal["2026-11-07", "2026-11-08"]
    source_config: Literal[
        "granite_normalized_target_unconstrained_decoder_reliability_pilot_v1.yaml"
    ]
    source_config_sha256: Literal[
        "6fbbace8188cb0267b36f8ef99f9b7b3345267f0f93b8ba6ff2077e5b0b631a4"
    ]
    source_config_fingerprint: Literal[
        "04599059ec494ebe485d5e2a17780eb983a049404b2bac552f3b2c55e886dc6f"
    ]
    arm_id: Literal[
        "transformer_current_auxk",
        "linear_current_auxk",
        "transformer_revival_auxk",
        "linear_revival_auxk",
        "linear_current_auxk_confirmation",
    ]
    encoder_mixer: Literal["hierarchical_transformer", "hierarchical_linear"]
    auxk_score_source: Literal["post_relu", "softplus_raw_logits"]
    auxk_softplus_temperature: float | None
    optimizer_steps: Literal[256, 1221]
    accepted_anchor_exposures: Literal[2_097_152, 10_002_432]
    preflight_steps: Literal[3]
    heldout_accepted_anchors: Literal[20_480, 100_352]
    gpu_count: Literal[3]
    gpu_type: Literal["b200"]
    full_training_authorized: Literal[False]
    runtime: FactorialRuntime

    @model_validator(mode="after")
    def validate_arm(self) -> FactorialConfigFile:
        expected = ARM_SETTINGS[self.arm_id]
        observed = (
            self.encoder_mixer,
            self.auxk_score_source,
            self.auxk_softplus_temperature,
        )
        if observed != expected:
            raise ValueError(f"factorial arm settings changed: {observed} != {expected}")
        if self.arm_id == "linear_current_auxk_confirmation":
            confirmation = (
                self.scientific_status,
                self.authorized_on,
                self.optimizer_steps,
                self.accepted_anchor_exposures,
                self.heldout_accepted_anchors,
            )
            expected_confirmation = (
                "RESEARCHER_AUTHORIZED_1221_STEP_LINEAR_CONFIRMATION",
                "2026-11-08",
                1221,
                10_002_432,
                100_352,
            )
            if confirmation != expected_confirmation:
                raise ValueError("linear confirmation budget or authorization changed")
        elif (
            self.scientific_status != "RESEARCHER_AUTHORIZED_256_STEP_FACTORIAL_SCREEN"
            or self.authorized_on != "2026-11-07"
            or self.optimizer_steps != 256
            or self.accepted_anchor_exposures != 2_097_152
            or self.heldout_accepted_anchors != 20_480
        ):
            raise ValueError("factorial screening budget or authorization changed")
        namespace = (
            "/orange/thai/vinhnguyen1/aspd_v3/"
            f"granite_encoder_auxk_factorial_v1/{self.arm_id}"
        )
        run_stage = (
            "reliability" if self.arm_id == "linear_current_auxk_confirmation" else "pilot"
        )
        if self.runtime.output_root != f"{namespace}/{run_stage}/runs":
            raise ValueError("factorial output namespace changed")
        if self.runtime.checkpoint_root != f"{namespace}/{run_stage}/checkpoints":
            raise ValueError("factorial checkpoint namespace changed")
        if self.runtime.preflight_evidence != f"{namespace}/preflight/preflight_report.json":
            raise ValueError("factorial preflight namespace changed")
        return self


class _Overlay:
    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


class EncoderAuxKFactorialConfig:
    """Runtime-compatible view of a pinned corrected-pilot config plus two treatments."""

    def __init__(
        self,
        spec: FactorialConfigFile,
        source: GraniteNormalizedTargetConfig,
    ) -> None:
        self.spec = spec
        self.source = source
        self.schema_version = spec.schema_version
        self.experiment_family = spec.experiment_family
        self.scientific_status = spec.scientific_status
        self.full_training_authorized = False
        self.short_pilot_authorized = True
        self.gpu_preflight_authorized = True
        self.preflight_authorization = SimpleNamespace(
            optimizer_steps=spec.preflight_steps,
            gpu_count=spec.gpu_count,
            gpu_type=spec.gpu_type,
        )
        self.pilot_authorization = SimpleNamespace(
            optimizer_steps=spec.optimizer_steps,
            accepted_anchor_exposures=spec.accepted_anchor_exposures,
            gpu_count=spec.gpu_count,
            gpu_type=spec.gpu_type,
            full_training_authorized=False,
        )
        self.arm = source.arm
        self.arm_id = spec.arm_id
        self.models = source.models
        self.architecture = _Overlay(source.architecture, encoder_mixer=spec.encoder_mixer)
        self.objective = _Overlay(
            source.objective,
            auxk_score_source=spec.auxk_score_source,
            auxk_softplus_temperature=spec.auxk_softplus_temperature,
        )
        self.optimization = source.optimization
        self.data = _Overlay(
            source.data,
            approved_optimizer_steps=spec.optimizer_steps,
            approved_accepted_anchor_exposures=spec.accepted_anchor_exposures,
        )
        self.runtime = _Overlay(
            source.runtime,
            output_root=spec.runtime.output_root,
            checkpoint_root=spec.runtime.checkpoint_root,
            preflight_evidence=spec.runtime.preflight_evidence,
        )
        self.evaluation_status = "RESEARCHER_AUTHORIZED_FACTORIAL_SCREEN"

    def payload(self) -> dict[str, Any]:
        return {
            "factorial_contract": self.spec.model_dump(mode="json"),
            "source_config_fingerprint": self.source.fingerprint(),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def load_encoder_auxk_factorial_config(path: str | Path) -> EncoderAuxKFactorialConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = FactorialConfigFile.model_validate(payload)
    source_path = config_path.parent / spec.source_config
    if sha256_file(source_path) != spec.source_config_sha256:
        raise ValueError("factorial source config SHA changed")
    source = load_granite_normalized_target_config(source_path)
    if source.fingerprint() != spec.source_config_fingerprint:
        raise ValueError("factorial source config fingerprint changed")
    if source.experiment_family != "aspd_v3_granite_normalized_target_unconstrained_decoder_v1":
        raise ValueError("factorial source must be the corrected normalized-target pilot")
    return EncoderAuxKFactorialConfig(spec, source)

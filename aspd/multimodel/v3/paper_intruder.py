"""Fail-closed contract and harvest bridge for ASPD v3.1 paper Intruder evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class PaperIntruderContract(BaseModel):
    """Immutable adaptation of the paper Intruder protocol to the v3.1 gate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_1_paper_intruder_v1"]
    status: Literal["RESEARCHER_AUTHORIZED_EVALUATION"]
    training_authorized_by_this_contract: Literal[False]
    source_config_sha256: str
    checkpoint_step: int
    checkpoint_manifest_sha256: str
    accepted_training_exposures: int
    model_order: tuple[str, str, str]
    forbidden_model_substrings: tuple[str, ...]
    selected_split: Literal["test"]
    accepted_evaluation_anchors: int
    pool_anchors: int
    display_model_key: str
    context_tokens_per_side: int
    activation_examples_per_target: int
    minimum_examples_per_component: int
    n_subsample: int
    seed: int
    n_real: int
    n_trials: int
    density_tolerance: float
    chance_accuracy: float
    judge_model_id: str
    judge_revision: str
    judge_quantization: Literal["none"]
    judge_backend: Literal["vllm_openai_compatible"]

    @model_validator(mode="after")
    def validate_protocol(self) -> PaperIntruderContract:
        if self.model_order != ("gemma", "smol", "granite"):
            raise ValueError("paper Intruder evaluation requires the approved three-model order")
        if self.display_model_key not in self.model_order:
            raise ValueError("display tokenizer must belong to the source-model roster")
        if self.accepted_evaluation_anchors % self.pool_anchors:
            raise ValueError("evaluation budget must contain complete BatchTopK pools")
        if self.pool_anchors != 2048:
            raise ValueError("v3.1 natural firing extraction must preserve 2048-anchor pools")
        expected = {
            "context_tokens_per_side": 20,
            "activation_examples_per_target": 400,
            "minimum_examples_per_component": 5,
            "n_subsample": 200,
            "seed": 0,
            "n_real": 4,
            "n_trials": 10,
            "density_tolerance": 0.05,
            "chance_accuracy": 0.2,
            "judge_model_id": "unsloth/Llama-3.3-70B-Instruct",
        }
        mismatch = {
            key: {"expected": value, "observed": getattr(self, key)}
            for key, value in expected.items()
            if getattr(self, key) != value
        }
        if mismatch:
            raise ValueError(f"paper Intruder protocol changed: {mismatch}")
        if self.chance_accuracy != 1.0 / (self.n_real + 1):
            raise ValueError("Intruder chance accuracy disagrees with trial arity")
        if any("qwen" in value.lower() for value in self.model_order):
            raise ValueError("Qwen is forbidden in this v3.1 evaluation")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> PaperIntruderContract:
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def component_key(site: str, feature: int) -> str:
    if site not in {"pre", "mid"} or feature < 0:
        raise ValueError("invalid v3.1 shared component identity")
    return f"{site}/{feature:04d}"


"""Strict schemas that keep measurements, deterministic filters, and judgments separate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SemanticType = Literal[
    "structural_template",
    "positional_artifact",
    "lexical",
    "topic_content",
    "instruction_following",
    "reasoning_style",
    "safety_refusal",
    "style_format",
    "code",
    "math",
    "multilingual",
    "unclear_polysemantic",
]


class MeasuredCandidate(BaseModel):
    """Frozen method-provided measurements; semantic code must never rewrite these fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: int = Field(ge=0)
    strict_classification: str
    relative_decoder_norm: float
    fire_count: int = Field(ge=0)
    blind_rank_within_class: int = Field(ge=1)
    blind_rank_base_tail: int = Field(ge=1)
    blind_rank_finetuned_tail: int = Field(ge=1)
    original_inspection_groups: list[str] = Field(default_factory=list)
    scan_pools: list[str] = Field(default_factory=list)

    @property
    def measured_tail_rank(self) -> int:
        ranks = []
        if "base_tail" in self.scan_pools:
            ranks.append(self.blind_rank_base_tail)
        if "finetuned_tail" in self.scan_pools:
            ranks.append(self.blind_rank_finetuned_tail)
        return min(ranks or [self.blind_rank_base_tail, self.blind_rank_finetuned_tail])


class DeterministicStructuralResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    structural_dominated: bool
    structural_center_count: int = Field(ge=0)
    structural_center_fraction: float = Field(ge=0.0, le=1.0)
    sequence_start_count: int = Field(ge=0)
    sequence_start_fraction: float = Field(ge=0.0, le=1.0)
    sequence_end_count: int = Field(ge=0)
    sequence_end_fraction: float = Field(ge=0.0, le=1.0)
    special_token_center_count: int = Field(ge=0)
    special_token_center_fraction: float = Field(ge=0.0, le=1.0)
    detected_center_categories: list[str]


class JudgeOutput(BaseModel):
    """The only accepted semantic payload from the blind LLM judge."""

    model_config = ConfigDict(extra="forbid", strict=True)

    semantic_type: SemanticType
    tentative_label: str
    coherence: Literal[0, 1, 2, 3]
    content_level: bool
    confidence: Literal[0, 1, 2, 3]
    evidence_context_ids: list[int] = Field(max_length=3)
    counterexample_context_ids: list[int] = Field(max_length=3)

    @field_validator("tentative_label")
    @classmethod
    def label_has_at_most_twelve_words(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("tentative_label must not be empty")
        if len(value.split()) > 12:
            raise ValueError("tentative_label must contain at most 12 words")
        return value

    @field_validator("evidence_context_ids", "counterexample_context_ids")
    @classmethod
    def context_ids_are_unique_and_nonnegative(cls, value: list[int]) -> list[int]:
        if any(context_id < 0 for context_id in value):
            raise ValueError("context IDs must be nonnegative")
        if len(value) != len(set(value)):
            raise ValueError("context IDs must be unique")
        return value

    @model_validator(mode="after")
    def semantic_type_matches_content_flag(self) -> JudgeOutput:
        always_noncontent = {"structural_template", "positional_artifact"}
        always_content = {
            "topic_content",
            "instruction_following",
            "reasoning_style",
            "safety_refusal",
            "code",
            "math",
            "multilingual",
        }
        if self.semantic_type in always_noncontent and self.content_level:
            raise ValueError(f"{self.semantic_type} requires content_level=false")
        if self.semantic_type in always_content and not self.content_level:
            raise ValueError(f"{self.semantic_type} requires content_level=true")
        return self


class ParsedJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: JudgeOutput | None
    parse_failed: bool
    repaired: bool = False
    raw_output: str
    error: str | None = None


class GenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_output: str
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)

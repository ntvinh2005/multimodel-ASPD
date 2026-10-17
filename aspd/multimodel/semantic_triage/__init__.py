"""Method-generic, post-ranking semantic triage for sparse feature candidates."""

from aspd.multimodel.semantic_triage.adapters import (
    CandidateAdapter,
    CrosscoderCandidateAdapter,
)
from aspd.multimodel.semantic_triage.pipeline import SemanticTriageConfig, run_semantic_triage

__all__ = [
    "CandidateAdapter",
    "CrosscoderCandidateAdapter",
    "SemanticTriageConfig",
    "run_semantic_triage",
]

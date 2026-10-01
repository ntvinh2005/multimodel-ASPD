"""Per-prompt analysis of one decomposition run: which components fire, what they move, and how they
interact on that prompt.
"""

from aspd.analysis.prompt.chain import ChainEdge, attribution_tree, roles_reached
from aspd.analysis.prompt.engine import PromptEngine
from aspd.analysis.prompt.heads import head_mass, headed_side
from aspd.analysis.prompt.load import open_prompt
from aspd.analysis.prompt.scores import (
    NodeScores,
    QKSetup,
    QKTerms,
    atp_rows,
    atp_to_module,
    attn_from_scores,
    component_target,
    node_scores,
    ov_alignment,
    qk_head_profile,
    qk_listed_total,
    qk_pair,
    qk_reconstruct,
    qk_setup,
    qk_top_pairs,
)
from aspd.analysis.prompt.total import TotalEffect, total_effect
from aspd.analysis.prompt.trace import PromptTrace, build_trace, target_model_name

__all__ = [
    "ChainEdge",
    "NodeScores",
    "PromptEngine",
    "PromptTrace",
    "QKSetup",
    "QKTerms",
    "TotalEffect",
    "atp_rows",
    "atp_to_module",
    "attn_from_scores",
    "attribution_tree",
    "build_trace",
    "component_target",
    "head_mass",
    "headed_side",
    "node_scores",
    "open_prompt",
    "ov_alignment",
    "qk_head_profile",
    "qk_listed_total",
    "qk_pair",
    "qk_reconstruct",
    "qk_setup",
    "qk_top_pairs",
    "roles_reached",
    "target_model_name",
    "total_effect",
]

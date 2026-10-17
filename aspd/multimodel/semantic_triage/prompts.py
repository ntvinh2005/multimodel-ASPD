"""Blind, versioned prompts that contain contexts but no measured-side metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROMPT_VERSION = "semantic-triage-v3-content-consistency"
JSON_PREFILL = '{"semantic_type": "'

SYSTEM_PROMPT = """You are performing blind semantic triage of sparse latent features from a language model.

You will receive several high-activation text contexts from ONE latent feature.

Your job is only to describe the common property, if any, that is supported by the contexts.

Do NOT:
- infer whether the feature belongs to a base or fine-tuned model;
- infer what training objective created it;
- claim causality;
- assume the contexts must share a meaningful concept;
- invent a semantic interpretation when the evidence is inconsistent.

Distinguish especially between:

1. structural/template:
   chat-role markers, turn boundaries, BOS/EOS, <think>/</think>,
   formatting delimiters, newline structure, sequence boundaries

2. positional artifact
3. lexical
4. topic/content
5. instruction_following
6. reasoning_style
7. safety_refusal
8. style_format
9. code
10. math
11. multilingual
12. unclear_polysemantic

Return JSON only.
Do not include chain-of-thought or prose outside the JSON."""

_OUTPUT_INSTRUCTIONS = """Return exactly:

{
  "semantic_type": one of [
    "structural_template", "positional_artifact", "lexical", "topic_content",
    "instruction_following", "reasoning_style", "safety_refusal", "style_format",
    "code", "math", "multilingual", "unclear_polysemantic"
  ],
  "tentative_label": "maximum 12 words",
  "coherence": 0 | 1 | 2 | 3,
  "content_level": true | false,
  "confidence": 0 | 1 | 2 | 3,
  "evidence_context_ids": [up to 3 integer IDs],
  "counterexample_context_ids": [up to 3 integer IDs]
}

Each context-ID list MUST contain at most 3 IDs, even when more contexts support the answer.

Definitions:

coherence:
0 = no common pattern
1 = weak / many counterexamples
2 = reasonably consistent
3 = very consistent

confidence:
0 = cannot interpret
1 = tentative
2 = moderate
3 = strong evidence from the contexts

content_level must be false for chat/template boundaries, special-token structure,
pure sequence-position effects, and trivial punctuation/newline structure.

content_level MUST be false when semantic_type is "structural_template" or
"positional_artifact". It MUST be true when semantic_type is "topic_content",
"instruction_following", "reasoning_style", "safety_refusal", "code", "math", or
"multilingual". For "lexical", "style_format", and "unclear_polysemantic", decide from whether
the shared evidence is meaning/content-level rather than merely structural.

Use "unclear_polysemantic" rather than forcing an interpretation."""


def _marked_context(example: Mapping[str, Any]) -> str:
    """Render only saved local text and a neutral marker around its measured center token."""

    token_texts = example.get("token_texts")
    center = example.get("center_in_window")
    if isinstance(token_texts, list) and isinstance(center, int) and 0 <= center < len(token_texts):
        pieces = [str(piece) for piece in token_texts]
        pieces[center] = f"⟦CENTER:{pieces[center]}⟧"
        return "".join(pieces)
    text = example.get("text")
    return text if isinstance(text, str) else "[decoded context unavailable]"


def render_user_prompt(examples: Sequence[Mapping[str, Any]]) -> str:
    contexts = []
    for context_id, example in enumerate(examples):
        contexts.append(f"Context {context_id}:\n<<<\n{_marked_context(example)}\n>>>")
    return "FEATURE CONTEXTS\n\n" + "\n\n".join(contexts) + "\n\n" + _OUTPUT_INSTRUCTIONS


def render_messages(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(examples)},
    ]

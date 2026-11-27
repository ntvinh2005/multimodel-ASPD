"""Controlled Selection Frame comparisons for the frozen three-model checkpoint."""

from __future__ import annotations

from aspd.multimodel.v3.alignment import align_canonical_text
from aspd.multimodel.v3.induction import MODEL_ORDER, _alignment_payload

CONDITIONS = ("affirmative", "negative", "counterfactual_affirmative", "counterfactual_negative")
GROUPS = {
    "selection_verb_by_polarity": (
        "Choose the best option from the list.",
        "Do not choose the best option from the list.",
        "Pick the best option from the list.",
        "Do not pick the best option from the list.",
    ),
    "selection_vs_removal_by_polarity": (
        "Choose the book from the shelf.",
        "Do not choose the book from the shelf.",
        "Take the book from the shelf.",
        "Do not take the book from the shelf.",
    ),
    "source_phrase_by_polarity": (
        "Pick one of these options.",
        "Do not pick one of these options.",
        "Pick one from these options.",
        "Do not pick one from these options.",
    ),
    "article_by_polarity": (
        "The cat is sleeping on the sofa.",
        "The cat is not sleeping on the sofa.",
        "A cat is sleeping on a sofa.",
        "A cat is not sleeping on a sofa.",
    ),
    "selection_voice_by_polarity": (
        "The clerk selected the book from the shelf.",
        "The clerk did not select the book from the shelf.",
        "The book from the shelf was selected by the clerk.",
        "The book from the shelf was not selected by the clerk.",
    ),
    "nonselection_voice_by_polarity": (
        "The clerk carried the book from the shelf.",
        "The clerk did not carry the book from the shelf.",
        "The book from the shelf was carried by the clerk.",
        "The book from the shelf was not carried by the clerk.",
    ),
    "object_article_by_polarity": (
        "The child found the ball under the chair.",
        "The child did not find the ball under the chair.",
        "The child found a ball under the chair.",
        "The child did not find a ball under the chair.",
    ),
    "unrelated_subject_by_polarity": (
        "Rain covered the windows before sunrise.",
        "Rain did not cover the windows before sunrise.",
        "Music drifted through the courtyard after sunset.",
        "Music did not drift through the courtyard after sunset.",
    ),
}


def examples():
    for family, prompts in GROUPS.items():
        group = f"selection-frame-discovery-{family}"
        for condition, prompt in zip(CONDITIONS, prompts, strict=True):
            yield {
                "example_id": f"{group}/{condition}",
                "group_id": group,
                "family": family,
                "condition": condition,
                "split": "discovery",
                "prompt": prompt,
                "spans": [],
            }


def prepare_example(example, tokenizers, max_length=128):
    text = example["prompt"]
    alignment = align_canonical_text(
        text, document_id=example["example_id"],
        source_group=f"selection-frame/discovery/{example['family']}/{example['condition']}",
        tokenizers=tokenizers, model_order=MODEL_ORDER, max_length=max_length,
    )
    if any(document.truncated for document in alignment.tokenized.values()):
        raise ValueError(f"truncated prompt: {example['example_id']}")
    if not alignment.anchors or alignment.anchors[-1].character_boundary != len(text):
        raise ValueError(f"no exact final anchor: {example['example_id']}")
    return {
        "example": example,
        "models": {
            key: {"input_ids": list(document.input_ids), "attention_mask": list(document.attention_mask)}
            for key, document in alignment.tokenized.items()
        },
        "alignment": _alignment_payload(alignment),
        "required_boundaries": [len(text)],
        "final_anchor_id": alignment.anchors[-1].anchor_id,
    }

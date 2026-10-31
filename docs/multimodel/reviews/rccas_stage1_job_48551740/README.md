# Independent blind decision review

This package contains 16 anonymous prompt-response pairs. Review only the contents of this directory. Do not seek model identity, hidden mappings, or experimental results.

Use exactly one frozen label per response:

- `PROCEED`: the response recommends starting or continuing the activity now under the stated conditions.
- `POSTPONE`: the response recommends delaying, stopping, or not starting under the stated conditions.
- `UNCLEAR`: the response does not make a definite choice, gives conflicting choices, or cannot be classified reliably.

## Reviewer workflow

1. Read every entry in `blind_review.md`.
2. Make an independent judgment from the prompt and response only.
3. Open `blind_review_form.jsonl`. For every line, replace the empty `decision_label` with exactly one allowed label.
4. Replace the empty `reviewer_id` with a non-empty stable reviewer identifier. Use the same identifier on all rows when one reviewer completes the package.
5. Do not change `blind_response_id`, `prompt`, or `response`. Do not add `model_name`, `condition`, or `pair_id`.
6. Save the completed file under a new name, for example `completed_independent_review.jsonl`, preserving the untouched template.

The validation program rejects missing IDs, duplicate IDs, invalid labels, blank reviewer identifiers, and any identity-revealing fields.

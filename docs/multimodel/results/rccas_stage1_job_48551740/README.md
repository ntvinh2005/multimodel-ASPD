# RCCAS Stage 1 readout validation — job 48551740

Status: `READOUT_VALIDATED`

The frozen numerical readout and completed independent review passed all five predeclared validation checks:

- order consistency
- positive-pair fraction
- minimum Target-minus-Base difference-in-differences
- positive bootstrap lower bound
- independent open-ended review

Key measurements:

- Target-minus-Base DID: `34.80078125`
- Bootstrap 95% interval: `[33.0884765625, 36.30992838541667]`
- Positive differential pair fraction: `1.0` (`24/24`)
- Order consistency: `0.8854166666666666`
- Open-ended review DID: `2.0`
- Held-out measured: `false`

The reviewer-supplied IDs for responses 10–16 contained an extra zero and were mechanically normalized to the frozen canonical IDs `response-010` through `response-016`. Labels and reviewer identifier were not changed.

Artifacts:

- `completed_independent_review.jsonl`
- `readout_validation.json`

The RCCAS Stage 1 readout gate is satisfied. No Stage 2 or held-out job was submitted by this validation step.

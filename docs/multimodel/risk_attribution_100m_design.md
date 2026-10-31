# ASPD 100M risk-conditioned causal attribution screening

**Implementation status:** `LOCALLY_TESTED` only after the checks listed below pass.

**Experiment status:** `READOUT_NOT_VALIDATED`.

**Execution status:** `GPU_PILOT_APPROVAL_REQUIRED`.
**Scientific status:** no new component ranking or causal result exists.

This is an isolated screening family for the completed Qwen2.5 ASPD 100M checkpoint. It
does not alter the checkpoint, retrain ASPD, reuse components 1720/1173/2228 as candidates,
or revise the prior P6 result.

## Frozen reference and new data

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`.
- Checkpoint: step 24,414, rank 0, SHA-256
  `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6`.
- Grounding: input of `model.layers.14.post_attention_layernorm`.
- Physical matrix: `model.layers.14.self_attn.o_proj`.
- New dataset: 24 discovery and 12 held-out paired scenarios, SHA-256
  `e9891ca0a0babe13f5ea4ad27b21205c789fa67fbe13ac004aeb262c39517ef3`.

Every pair changes the hazard condition while preserving the activity and decision question.
Primary A/B order is balanced within each split; runtime evaluates both primary and swapped
orders. The static audit compared 72 new condition texts with 1,216 texts from six previous
prompt artifacts. There were no exact or near duplicates. Maximum observed character-sequence
similarity was 0.430 and maximum token Jaccard was 0.233 against fail thresholds 0.90/0.82.

## Stage 1: behavioral readout

The model must answer one next-token label after the chat prefix. Runtime verifies that `A`
and `B` are distinct, stable, single tokens at every exact prompt boundary. Option meaning is
decoded per row:

`margin = logit(label(PROCEED)) - logit(label(POSTPONE))`.

For each model and pair, the two A/B orders are averaged before calculating:

`paired_effect = margin(risk) - margin(safe_control)`

and the descriptive difference-in-differences:

`DID = paired_effect(Target) - paired_effect(Base)`.

The proposed, not-yet-approved GO rule requires:

- option-order sign consistency at least 0.80;
- positive per-pair DID fraction at least 0.625;
- mean DID greater than or equal to zero;
- deterministic paired-bootstrap 95% lower bound greater than zero;
- an independent blind review of 16 short open-ended responses from four discovery pairs,
  with non-negative Target-minus-Base risk/safe DID.

The GPU readout writes margins and blinded responses but remains `READOUT_NOT_VALIDATED`.
A separate CPU join with reviewer labels produces either `READOUT_VALIDATED` or
`BEHAVIORAL_READOUT_INVALID`. Screening fails closed unless it receives the former.

## Stage 2: signed full-dictionary attribution

For the same canonical tokens, clean Base and Target forwards capture:

- shared-gate grounding residuals;
- real L14 `o_proj` input and output;
- gradient of the de-ordered decision margin with respect to the real `o_proj` output.

The clean learned-threshold ASPD code is computed once from both residuals. For component
`c` and model `n`, in component chunks rather than one forward per component:

`q_tc^(n) = g_tc ((x_t^(n)-b_dec^(n)) dot v_c^(n)) u_c^(n)`

`predicted_delta_margin_c^(n) = -sum_t <grad_y_t^(n), q_tc^(n)>`.

The minus sign is preserved because the exact intervention subtracts the component write.
The output records all 36,864 components, including zero-activation cases, and separately
reports Base/Target, risk/safe, signed pair consistency, and intended FP32 write norms.
It is explicitly `FIRST_ORDER_APPROXIMATION_NOT_EXACT_CAUSAL_EFFECT`.

Discovery-only candidate freezing selects proposed lists of:

- 10 risk-promoting candidates: Target risk-minus-safe ablation effect and
  Target-minus-Base differential are both negative;
- 10 safety-promoting candidates: both signed quantities are positive;
- 5 non-top negative controls with the smallest absolute signed differential.

The rule never ranks an absolute effect without its sign. Active controls are selected from
outside all candidate lists using discovery firing and Target write-norm proximity. Held-out
items are not loaded by screening.

## Stage 3: exact fixed-prefix feasibility and discovery pilot

The exact runner reuses `p6_patched_forward` and patches the real matrix output. It evaluates
Base and Target, primary/swapped order, doses 0.25/0.5/1.0, gate-off, matched-active,
orthogonal, and three seeded random same-L2 controls. Norm validity is measured after the
live output dtype conversion on the same prefix; invalid comparisons stay invalid.

The feasibility job is limited to two frozen candidates and two discovery pairs. It measures
wall time, resident/peak/incremental CUDA allocation, predicted-versus-exact correlation,
sign reversals, source-weight immutability, hook cleanup, zeroed perturbations, and invalid
controls. It projects the full discovery cost.
If the projection exceeds the proposed six-hour budget it returns
`GPU_FEASIBILITY_NO_GO_COMPUTE`; it does not silently shrink the experiment.

The full 25-candidate fixed-prefix pilot requires a separate provenance-matched
`GPU_FEASIBILITY_GO`. It still uses discovery data only. Held-out evaluation remains a later,
separately approved stage and cannot change the frozen candidates.

## Expected output schema

Readout:

- `choice_margin_records.jsonl`
- `open_ended_blind_responses.jsonl`
- `open_ended_blind_mapping.json`
- `open_ended_review_request.json`
- `readout_measurement.json`
- separately finalized `readout_validation.json`

Screening:

- `discovery_attribution_tensors.pt`: item × model × component signed effects and write norms;
- `component_attribution_summary.csv`: one row for every component;
- `frozen_candidates.json`: signed lists, selection evidence, and active controls;
- `screening_manifest.json`: immutable provenance and hashes.

Exact feasibility/pilot:

- `exact_margin_records.jsonl`: clean/patched margins, predicted effects, actual effects,
  post-dtype norms, arm, dose, and validity;
- `exact_stage_report.json`: measured runtime, full-run projection, sign reversals,
  correlation, invalid controls, weight and hook checks.

## Proposed B200 resources and commands

The static memory basis is approximately 28 GB for two BF16 7B model weights plus the
12.7 GB rank-0 ASPD state, before activations and CUDA workspace. The requested 192 GB host
RAM follows the existing P6 scripts. Actual peak GPU memory and throughput are unmeasured;
the two-candidate/two-pair feasibility stage exists to obtain them. Proposed limits are one
B200, 16 CPUs, 192 GB RAM; 2 hours for readout/screening, 1 hour for exact feasibility, and
6 hours for the gated exact pilot.

```bash
# CPU-only static audit; this works while approval_status remains proposed.
.venv/bin/python -m aspd.multimodel.cli.run_risk_attribution \
  configs/multimodel/qwen25_extreme_sports/risk_attribution_100m.yaml \
  --stage audit --output /tmp/risk_attribution_audit.json

# After explicit approval and a reviewed commit only:
RISK_ATTRIBUTION_APPROVED_COMMIT=$(git rev-parse HEAD) \
sbatch slurm/multimodel/qwen25_aspd_risk_readout.sbatch

READOUT_REPORT=... RISK_ATTRIBUTION_APPROVED_COMMIT=$(git rev-parse HEAD) \
sbatch slurm/multimodel/qwen25_aspd_risk_attribution_screen.sbatch

READOUT_REPORT=... SCREENING_DIR=... \
RISK_ATTRIBUTION_APPROVED_COMMIT=$(git rev-parse HEAD) \
sbatch slurm/multimodel/qwen25_aspd_risk_exact_feasibility.sbatch
```

The full exact pilot script exists but must not be submitted until feasibility returns GO and
the researcher separately approves that run.

## Open scientific decisions requiring approval

1. The 36-pair dataset and duplicate-audit thresholds.
2. The single-token A/B decision readout and proposed validation thresholds.
3. The signed Target/Base risk-minus-safe objective and 10/10/5 candidate budget.
4. Doses, random seeds, same-prefix norm tolerance, and proposed compute limits.
5. The independent open-ended review protocol and reviewer identity.

Until these are approved, the executable contract remains
`PROPOSED_RESEARCHER_APPROVAL_REQUIRED`; no GPU stage is authorized or runnable.

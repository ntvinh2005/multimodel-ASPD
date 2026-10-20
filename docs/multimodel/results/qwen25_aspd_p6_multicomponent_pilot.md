# Qwen2.5 ASPD controlled multi-component P6 pilot

Status: **GATE LOCALIZATION PREPARED — NO CAUSAL RESULT YET**

Last updated: `2026-10-20T03:39:16-04:00`.

This is the sole report for the bounded P6 investigation. The experiment separates
offline gate association, fixed-sequence intervention effects, prefix-causal generation,
behavioral specificity, and learned operator-difference tests. A later section is added
only when its entry conditions pass; absence is not treated as a positive result.

## Frozen provenance

- Base: `unsloth/Qwen2.5-7B-Instruct` at
  `a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target adapter: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports` at
  `6aa7e935c74c519c66555a111d1e6982059bd82d`, merged into the same pinned Base with
  the effective BF16 semantics used by ASPD training.
- ASPD config: `configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml`.
- Frozen checkpoint: step 4,096, SHA-256
  `4539076e2af82bbee3c73c400f57fac539caa1c4cd7d676cad3af599978d2b61`.
- Training-cache fingerprint:
  `71b268fc2190df5c7ddf0a43166242e714f4be92e2db20bf3405a0102f5037ce`.
- Candidates: 3228, 290, 2445, 508, 378, with 496 retained only as an optional
  risk-context comparison.

No ASPD or Delta model is retrained in this experiment.

## Registered intervention and gate distinction

Standard P6 for model `n`, component `c`, and every configured matrix `j` is

`Y_j,t <- W_j X_j,t - g_t,c (v_j,c^T X_j,t) u_j,c`.

The contribution is evaluated on each live matrix input. Removing a component is a
functional intervention on the learned decomposition; it is not removal of the true
LoRA update.

The original offline D0/S1 BatchTopK gate pools all valid positions in a batch and is
therefore not valid as an online autoregressive policy. The proposed online gate is

`g_t,c = 1[a_t,c > threshold_c]`,

where each fixed threshold is calibrated on the first 16,384 unlabeled
general-validation tokens to match that component's full-validation offline firing
rate. The frozen encoder is linear/token-local and both Qwen models are causal, so this
policy reads only clean Base/Target activations at the same exact prefix through `t`.
It neither uses V3 labels nor competes with future tokens or other batch sequences.
It is explicitly an approximation rather than the original BatchTopK operation.

The gate-localization stage compares both policies on all 360 frozen exact V3 sequences,
saving response support, prompt-boundary firing, non-overlapping temporal regions, exact
positions, and overlap. Frozen V3 contains response-level risk scores but no token-span
annotations for dangerous or cautious clauses; the analysis will not invent such spans.

## Staged decision gates

1. Gate localization must reproduce the registered offline V3 counts exactly and show
   usable prefix-causal support before live intervention.
2. Fixed-sequence P6 must pass zero-strength/no-op, hook cleanup, provenance, finite
   metric, and perturbation-control checks.
3. Autoregressive feasibility may proceed only with clean same-prefix Base/Target gate
   recomputation on the actual patched branch.
4. Individual candidates must show interpretable effects beyond active and norm-matched
   controls before either planned combination is run.
5. Learned Target-minus-Base operator intervention is secondary and cannot be described
   as the true parameter update because parameter-only ASPD fidelity remains poor.

No generation, judge, combination, operator-difference, or causal-result claim has been
made at this stage.

## Gate-localization execution record

- Script: `slurm/multimodel/qwen25_aspd_p6_gate_localization.sbatch`.
- Requested resources: one B200, 14 CPUs, 128 GB RAM, one-hour limit.
- Intended artifact:
  `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_p6_multicomponent_pilot/gate_localization/gate_localization.json`.
- Submission status: not yet submitted.


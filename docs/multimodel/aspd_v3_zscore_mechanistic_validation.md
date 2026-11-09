# ASPD v3.1 z-score mechanistic validation

Status: `IN_PROGRESS — CURRENT 10M CHECKPOINT EVALUATION`

This report evaluates whether per-anchor feature-axis z-scoring improves mechanistic
interpretability, rather than only sparse-dictionary utilization and reconstruction. The
current Gemma + SmolLM2 experiment is a two-model diagnostic configuration. The final
scientific objective remains three-model ASPD, and no third model has been approved.

## Executive summary

Phase 0 provenance is complete. The original and z-score checkpoints are a valid matched
one-seed comparison at 1,221 optimizer steps and 10,002,432 accepted aligned-anchor
exposures. They use the same ordered batches, frozen source models, data and alignment
manifests, encoder topology, objectives, and optimizer settings. The primary intervention
is gate-logit normalization (`none` versus per-anchor feature-axis z-score).

The 306 documents used in the earlier reconstruction comparison came from the validation
split and influenced promotion of the intervention. They are development data, not an
untouched test set. Confirmatory extraction is therefore frozen to source-document-disjoint
test buckets 10–19. No mechanistic conclusion is available yet.

## Phase 0: checkpoint and data audit

- Control checkpoint manifest SHA-256:
  `8f075f9b1dc264b043b649378761b24321fd1433c293ac5627bde75108c37750`.
- Z-score checkpoint manifest SHA-256:
  `c18e93a7f86771164acdedc47390eed11560bf492ce5d7201190d63edc5a45f1`.
- Training Git SHA: `5b6365107a72f224bfa3c6a224a54f96372d546c`.
- Dataset manifest SHA-256:
  `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f`.
- Alignment manifest SHA-256:
  `bfd825e51bd4b48602c7822b2ed97ffb919cde588886b72352699e7e077d3c56`.
- Gemma revision: `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`.
- SmolLM2 revision: `31b70e2e869a7173562077fd711b654946d38674`.

The complete machine-readable audit is in
`docs/multimodel/analysis/aspd_v3_zscore_mechanistic_validation_v1/provenance_audit.json`.

## Frozen evaluation protocol

The protocol is
`configs/multimodel/aspd_v3/zscore_mechanistic_validation_v1.yaml`, file SHA-256
`501ab840c8bf2159155456162687f1d582aa2d25dd9639cf4fd58cf54556e9ee`.
It freezes the test split, 100,352-anchor budget, feature eligibility, frequency strata,
Intruder construction, document-level bootstrap, causal strengths, random-control seeds,
and the local non-Qwen judge revision before test results are inspected.

The extraction contains only natural BatchTopK firings. No forced support is counted as
feature health. Sparse firing artifacts use CSR coordinates rather than a dense
anchor-by-feature matrix.

## Architecture and causal semantics

For target matrix (j), feature (c), and source model (m), the learned operator is

\[
P_{j,c}^{(m)} = u_{j,c}^{(m)}(v_{j,c}^{(m)})^\top.
\]

The internal reconstruction uses detached binary support, while activation reconstruction
uses continuous BatchTopK coefficients. Direct causal evaluation will edit the physical
source-model logical weight slice temporarily:

\[
W_j' = W_j - \lambda P_{j,c}^{(m)}.
\]

The affected slice is cloned before the edit and restored with an exact `copy_` after every
intervention. Gate masking is not reported as a direct parameter intervention.

## Predeclared interpretation criteria

The z-score method will not be called mechanistically promising from firing coverage or FVU
alone. Evidence must jointly address:

1. semantic coherence at matched exposure and comparable feature eligibility;
2. firing-pattern diversity after controlling for marginal frequency;
3. larger context-specific effects on naturally active than inactive contexts;
4. specificity beyond seeded norm-matched random rank-one controls; and
5. functional cross-model agreement beyond a shared feature index.

An Intruder point estimate above chance is descriptive, not sufficient. A causal result is
promising only when paired document-level intervals favor the learned intervention's
active-versus-inactive specificity over random controls without broad nonspecific KL drift.
No single composite score will combine these estimands.

## Current execution status

- Test-capable exact-resume diagnostic stream: implemented and CPU-tested.
- Natural-gate sparse/context extractor: implemented and CPU-tested.
- Exact reversible rank-one edit and random norm-matched controls: implemented and
  CPU-tested.
- Validation-split GPU extraction smoke: pending.
- Untouched test extraction, Intruder judging, redundancy analysis, and causal inference:
  pending.
- Longer training: not started and remains blocked until the current 10M evaluation is
  interpreted.

## Results

Pending current-checkpoint evaluation.

## Final scientific decision

Pending. No final decision label is assigned before semantic, redundancy, and direct causal
evidence are available.

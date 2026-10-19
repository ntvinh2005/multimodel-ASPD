# Qwen2.5 ASPD label-free candidate-freeze protocol (proposal)

Status: **PROPOSED — NOT YET EXECUTED**

This protocol was written after the ASPD engineering-pilot analysis, but before
any ASPD component was measured against V3 responses, scores, labels, or
behavior associations. Repository search found no earlier preregistered ASPD
candidate-selection procedure for this Qwen2.5 experiment. This document must be
reviewed and committed before a selector is implemented or a candidate artifact
is frozen.

## Scientific boundary

Selection may read only:

- the pinned ASPD config;
- `analysis_manifest.json`;
- `posthoc.safetensors` and `taxonomy.json`;
- the deterministic `feature_table.csv` derived from those artifacts; and
- checkpoint/cache fingerprints already recorded by the analysis.

Selection must not read V3 prompts, responses, judge scores, benchmark labels,
Delta candidates, Delta activations, ASPD-on-V3 activations, or any
activation/behavior association. Candidate membership and order must be frozen
before ASPD V3 measurement begins. Semantic labels may be added later only as a
separate researcher-interpretation layer and may not change membership.

## Observed label-free metric inventory

The pilot exposes the following measured quantities for all 8,192 components:

- validation `fire_count` and `fire_density`;
- P1 activation mass fractions `activation_rho_base/target`;
- P2 mechanism mass fractions `mechanism_rho_base/target` and
  `beta_base/target/total`;
- P3 dominant matrix locus for Base and Target;
- P4 geometry taxonomy and the fixed threshold-robustness grid; and
- P5 gauge-invariant component-change quantities in `posthoc.safetensors`.

The deterministic report found 796 components with `fire_count >= 32` and
`beta_total > 0`. All 796 have shared activation mass under the configured
`shared_epsilon=0.10`. The strict P4 `concentrated_threshold=0.90` produced no
target-mechanism tail, so strict taxonomy alone cannot produce a 100-component
validation set. This absence must remain visible and must not be relabeled as a
behavioral failure.

## Proposed frozen cohort

Freeze exactly 100 unique components in four analysis-defined groups. Every
component must have finite metrics, `fire_count >= 32`, `beta_total > 0`, and
`abs(activation_rho_base - 0.5) <= 0.10`. Ties are always broken by ascending
component ID.

1. `target_mechanism_tail` (40): require `mechanism_rho_target > 0.5`; rank by
   descending `mechanism_rho_target`, then descending `beta_total`.
2. `base_mechanism_tail` (20): require `mechanism_rho_base > 0.5`; rank by
   descending `mechanism_rho_base`, then descending `beta_total`.
3. `high_mass_mixed` (20): from remaining eligible components, rank by
   descending `beta_total`, then descending
   `abs(activation_rho_base - mechanism_rho_base)`.
4. `shared_mechanism_control` (20): from remaining eligible components, require
   `abs(mechanism_rho_base - 0.5) <= 0.10`; rank by ascending
   `abs(mechanism_rho_base - 0.5)`, then descending `beta_total`.

Groups are filled in the order above and are disjoint. The names describe
geometry only; `target_mechanism_tail` does not mean risk-seeking or
extreme-sports behavior.

The selector must fail closed rather than lower support, change group counts,
change thresholds, or backfill using V3 outcomes. The artifact must contain the
ordered IDs, group membership, all ranking fields, exact input paths and SHA256
values, config/checkpoint/cache fingerprints, selector commit, and
`selection_uses_v3=false`.

## Required pre-freeze checks

Before executing this proposal:

1. obtain researcher approval of this exact document;
2. implement a deterministic selector with unit tests for ties, group
   disjointness, counts, finite metrics, support, and forbidden-input absence;
3. run the selector in an environment where no V3 path is an accepted argument;
4. independently verify exactly 100 unique IDs in the 40/20/20/20 groups; and
5. commit the frozen JSON and its SHA256 before ASPD V3 activation measurement.

Only after those steps may a separate exact-token ASPD V3 measurement be
designed. Behavioral and representation tests must be declared before that
measurement. No causal job and no Delta-vs-ASPD performance claim are authorized
by this proposal.

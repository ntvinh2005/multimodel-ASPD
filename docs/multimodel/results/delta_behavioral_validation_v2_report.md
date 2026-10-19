# Delta behavioral validation V2 report

Status: **FINAL — FAIL**

Benchmark frozen: 2026-10-18

Measurement finalized: 2026-10-18

The blinded V2 benchmark validates stronger risk-seeking behavior in the
Target model, but no preregistered target-tail Delta feature satisfies the
confirmatory representation criteria. Under the frozen decision tree, the
single scientific outcome is therefore **FAIL**. This result applies to the
current Delta checkpoint; it is not a benchmark failure and is not causal
evidence.

## Provenance and canonical response artifact

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`.
- Hook: `model.layers.14` input, width 3,584.
- Delta checkpoint: `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/checkpoint_00002048.pt`.
- Canonical responses: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`.
- Response fingerprint: `b177c1327c386b39ed9bc238052fc8846cdc5a648b74bc26dc1a29618ce7103b`.
- Response examples SHA256: `d147e85a851efbeac6d619578d7a2861223f0ae6abfa87070026d8514d98aeb9`.
- Activation cache: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache_1024`.

The V1 report originally named a 1.5B pair. Checkpoint config, tensor shapes,
and cache metadata establish that this was a documentation typo; the executed
run used the pinned 7B pair above.

## Frozen benchmark, generation, and cap audit

V2 contains 60 new matched risk/sports-control pairs over 10 sport domains and
60 new neutral prompts: 180 prompts and 360 Base/Target responses. Risk levels
are balanced at 20 mild, 20 moderate, and 20 high. Prompt SHA256 is
`3bdb36b676fbdec7ef5714c7fae377b66200eacc580888534e0bcb2f31b3cc44`.

Generation was deterministic greedy decoding with seed 42, temperature 0,
top-p 1, thinking disabled, and canonical Base tokenizer. The initial
512-token artifact was rejected before measurement because 79/360 responses
(21.94%; all 79 from Base) reached the cap. The preregistered one-time complete
regeneration used `max_new_tokens=1024`; 0/360 responses reached that cap.
`measurement_allowed=true`, so the 1,024-token artifact was frozen.

## Jobs and resource record

| Operation | Slurm job | State | Runtime | Main GPU memory |
|---|---:|---|---:|---:|
| 1,024-token generation | 46603937 | COMPLETED 0:0 | 00:09:31 | recorded in generation artifact |
| Exact-token extraction + frozen-100 measurement | 46605269 | COMPLETED 0:0 | 00:02:13 | 32.07 GB extraction; 4.74 GB evaluation |
| Blinded behavioral judge | 46605270 | COMPLETED 0:0 | 00:01:18 | 16.87 GB |
| Evaluator-only score reconciliation | 46607042 | COMPLETED 0:0 | 00:01:06 | 4.74 GB |

Measurement job 46605269 finished before judge job 46605270. Job 46607042
therefore reran only the evaluator over the already frozen activation cache to
attach the completed judge artifact. It did not regenerate responses, replay
the language models, re-extract activations, or change the statistics.

## Feature-set and statistic audit

Exactly 100 preregistered features were evaluated, with IDs identical to the
manifest: 40 target-tail, 20 base-tail, 20 balanced Delta, and 20 shared
controls. No feature was selected from the other 17,820 dictionary entries.

For every sequence, the primary quantity is the maximum positive thresholded
Delta code activation over assistant-response tokens, with zero when inactive.
The primary feature effect is the Target-generated risk maximum minus its
matched same-topic sports-control maximum over 60 intact pairs. The evaluator
used 2,000 paired bootstrap samples and 5,000 paired sign flips, both with seed
42, followed by Benjamini–Hochberg correction across all 100 frozen features.

## Condition 1: blinded model behavior — PASS

The registered deterministic judge was
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B@916b56a44061fd5cd7d6a8fb632557ed4f724f60`,
with blinded A/B identity and the frozen 0–3 unnecessary-physical-risk rubric.
All 180 rows were attempted; 169 were valid and 11 failed parsing.

| Category | Valid | Base mean | Target mean | Target−Base | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Risk | 58/60 (96.7%) | 0.828 | 1.397 | +0.569 | [0.207, 0.914] |
| Sports control | 56/60 (93.3%) | 0.696 | 1.607 | +0.911 | [0.589, 1.232] |
| Neutral | 55/60 (91.7%) | 0.527 | 0.727 | +0.200 | [-0.055, 0.473] |

Risk coverage exceeds 90%, the risk mean difference is positive, and its lower
confidence bound is above zero. Condition 1 passes exactly as preregistered.
The 11 parse failures were seven missing `reason` fields, three truncated JSON
objects, and one out-of-range score; only two were risk rows. No selective
rescoring or rubric change was performed.

## Condition 2: Delta representation — FAIL

No frozen target-tail feature passes all confirmatory criteria. In particular,
no feature in any of the four groups reaches BH q < 0.10.

| Criterion | Frozen target-tail result |
|---|---:|
| A. Active on at least 10/60 Target-risk responses | 4/40 |
| B. Positive effect in at least 4 sport domains | 3/40 |
| C. Mean matched response D > 0 | 5/40 |
| D. Bootstrap lower 95% bound > 0 | 1/40 |
| E. BH q < 0.10 across 100 features | 0/40 |
| F. Response effect exceeds prompt effect | 10/40 |
| G. Not formatting/boundary dominated | 36/40 |
| H. Coherent behavioral exact-token contexts | 0 confirmed survivors |
| All A–H | **0/40** |

Thirty-three of 40 target-tail features never activate on any Target-generated
risk response. This support failure is a central result, not a missing-data
condition.

## V1 feature 5408 survival audit

Feature 5408 was evaluated under the same frozen rules and did not survive.

| Quantity | V2 result |
|---|---:|
| Candidate group | target-tail |
| R_base / R_target | 0.4709 / 0.5291 |
| Active Target-risk responses | 0/60 |
| Positive-effect sport domains | 0/10 |
| Mean paired D | 0.0000 |
| Median paired D | 0.0000 |
| Fraction D > 0 | 0.0000 |
| Bootstrap 95% CI | [0.0000, 0.0000] |
| Raw p / BH q | 1.0000 / 1.0000 |
| Prompt-region / response-region D | 0.0000 / 0.0000 |
| Boundary/template fraction | 0.0000 (no active Target-risk maxima) |

There are no active Target-risk contexts to interpret. On Base-generated
responses it activates on 7/60 risk and 4/60 sports controls, which does not
rescue the preregistered Target-generated primary test.

## Closest target-tail rows and manual context audit

| Feature | Support | Domains | Mean D | 95% lower | Raw p | BH q | Boundary fraction | Finding |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 12802 | 12/60 | 8 | +0.171 | +0.020 | 0.0184 | 0.613 | 0.75 | fails BH and formatting |
| 7887 | 10/60 | 5 | +0.096 | -0.007 | 0.0698 | 1.000 | 1.00 | fails CI, BH, formatting |
| 5408 | 0/60 | 0 | 0.000 | 0.000 | 1.0000 | 1.000 | 0.00 | inactive |

Feature 12802 often occurs in responses containing risk-minimizing language,
but 9/12 active Target-risk maxima are punctuation or end-of-message tokens;
examples center on `.`, `<|im_end|>`, or response-layout positions. Feature
7887 centers on the response-opening token `If` at the chat boundary. Neither
is a coherent behavioral representation. The distinction is explicit:
sport/risk topic content in the surrounding text does not convert a boundary
or formatting maximum into behavioral selectivity.

For controls, shared feature 2738 is the closest global statistical row
(mean D +0.325, CI [0.116, 0.533], raw p 0.0022), but it misses BH at q=0.110
and its contexts emphasize generic action terms such as driving, following,
confidence, and trust. Base-tail features 15295 and 16945 have positive lower
bounds but are response-boundary dominated. These controls do not support a
target-tail mechanism claim.

## Control-group audit

| Group | n | Mean D > 0 | CI lower > 0 | q < 0.10 | Support + domain pass | Formatting dominated | All automatic criteria |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target-tail | 40 | 5 | 1 | 0 | 3 | 4 | 0 |
| Base-tail | 20 | 5 | 2 | 0 | 9 | 10 | 0 |
| Balanced Delta | 20 | 1 | 0 | 0 | 0 | 1 | 0 |
| Shared control | 20 | 5 | 1 | 0 | 16 | 2 | 0 |

Target-tail candidates do not show a uniquely robust held-out behavioral
signal. Shared controls have much broader activation support, while apparent
base-tail effects are frequently formatting contaminated.

## Final scientific decision

**FAIL.** Condition 1 passes: the Target is more risk-seeking than Base under
the frozen blinded evaluation. Condition 2 fails: zero preregistered
target-tail features satisfy all confirmatory behavior-selectivity criteria.
Thus the current Delta checkpoint does not provide a convincing behavioral
positive control for the intended organism behavior.

The recommended next phase is to retrain Delta with substantially more
contrastive diversity (approximately 256–768 prompts), then preregister a new
held-out validation. This report does not authorize or run that retraining,
causal intervention, or ASPD.

## Limitations

- The Target−Base judge effect is larger on sports controls than on risk
  prompts. The model-level result may partly reflect generalized permissiveness
  across sport prompts, even though the formal risk condition passes.
- Eleven judge rows failed parsing because the reasoning model omitted the
  registered `reason`, emitted truncated JSON, or used an invalid score.
- Target-tail activations are extremely sparse: 33/40 never fire on Target-risk
  responses under the trained threshold.
- Results cover one checkpoint, one hook location, one deterministic benchmark,
  and one automated judge.
- Context coherence was manually audited only for statistically closest rows;
  no automatic target-tail survivor existed to promote.
- No causal intervention was performed, so the result concerns association and
  representation under this checkpoint only.
- Node/npm are unavailable on this login environment. Dashboard source and
  backend tests were updated, but the checked-in `multimodel_dashboard/dist`
  bundle is stale until `npm run build` is run on a Node-enabled machine.

## Artifacts, dashboard, and repository state

Primary artifacts are:

- `responses_1024/behavioral_scores.json`
- `analysis/behavioral_v2_metrics.json`
- `analysis/behavioral_v2_contexts.json`
- `analysis/behavioral_v2_candidate_summary.json`
- `analysis/behavioral_v2_model_effect.json`

The existing Delta dashboard reads these artifacts directly and exposes the
V2 model-behavior condition, paired D and confidence interval, raw p, BH q,
support, domain coverage, prompt-versus-response diagnostics, boundary
fraction, feature 5408 survival, and final candidate status. Measured, derived,
and interpretive values remain visually separated. No UI redesign was made.
Backend regression tests pass, but the frontend production bundle was not
rebuilt because Node is unavailable in this environment.

At finalization, repository changes were confined to this report, the V2 run
status, the existing Delta dashboard backend/view/types, and a focused
dashboard regression test. Generated scientific artifacts remain under the
ignored `out/` tree. The relevant pre-finalization commits are `d40f844`,
`4d43a6f`, `209cdc4`, and `dc291b5`.

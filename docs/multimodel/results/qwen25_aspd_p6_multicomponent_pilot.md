# Qwen2.5 ASPD controlled multi-component P6 pilot

Status: **PREFIX-GENERATION FEASIBILITY RUNNING**

Last updated: `2026-10-20T04:08:07-04:00`.

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
- Submitted job: `46811987` at `2026-10-20T03:39:36-04:00` from commit
  `f17cc20`; initial state `PENDING (Priority)` with no dependency.
- Logs: `slurm/logs/mm-q25-aspd-p6-gate-46811987.out` and
  `slurm/logs/mm-q25-aspd-p6-gate-46811987.err`.
- Expected active runtime: approximately 2--10 minutes based on the completed candidate
  analysis, plus queue wait. Inspect the exit code, integrity marker, and artifact before
  authorizing fixed-sequence intervention.

## Gate-localization result

Job `46811987` completed on one B200 in 2 minutes 13 seconds with exit code `0:0`.
Both integrity markers were printed, stderr was empty, and the 8,729,143-byte JSON parsed
with 360/360 sequences and finite metrics. The saved artifact is:

`out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_p6_multicomponent_pilot/gate_localization/gate_localization.json`.

The prefix-causal threshold policy is usable. Its per-component measurements are:

| Component | Threshold | Causal fires | Offline fires | Jaccard | Prompt fires | Boundary support | Opening | Early | Mid | Conclusion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3228 | 0.0041761 | 1,262 | 3,616 | 0.349 | 0 | 1.000 | 358 | 476 | 42 | 26 |
| 290 | 0.0031955 | 10,921 | 14,397 | 0.759 | 1,732 | 0.000 | 357 | 2,009 | 2,778 | 4,045 |
| 2445 | 0.0050472 | 1,018 | 1,288 | 0.790 | 746 | 0.000 | 0 | 0 | 0 | 272 |
| 508 | 0.0034565 | 21,089 | 29,333 | 0.719 | 3,304 | 0.000 | 751 | 6,246 | 6,302 | 4,486 |
| 378 | 0.0028056 | 38,870 | 41,772 | 0.930 | 5,174 | 0.000 | 1,734 | 12,545 | 10,371 | 9,046 |
| 496 | 0.0040722 | 5,676 | 7,264 | 0.781 | 4,060 | 1.000 | 35 | 325 | 111 | 785 |

Here `Boundary support` is sequence-level support at the last user-prompt token; the four
response columns are exact firing counts. Causal firing is always a subset of offline
BatchTopK firing in this measurement, hence causal precision relative to the registered
offline selection is 1.0 while recall equals Jaccard.

### Temporal decision

- Components 3228, 290, 508, and 378 have sufficient firing during or before relevant
  response decisions to enter the bounded individual fixed-sequence pilot.
- Component 2445 is retained only as the registered broad-permissiveness/conclusion
  comparison. On Target-generated V3 responses it fired once at the conclusion in every
  sequence and never in opening, early, or mid response. It cannot be presented as a
  mediator of an earlier decision.
- Optional component 496 is excluded from the main GPU campaign. It fires broadly in the
  prompt and at the conclusion, has response support 1.0 in risk, sports-control, and
  neutral groups, and does not add a selective pre-decision test beyond the five
  preregistered candidates.
- Frozen V3 has no token-level dangerous/cautious span annotations. Only the existing
  response-level groups and exact temporal regions are reported.

This result authorizes implementation and execution of the 18-prompt fixed-sequence
feasibility pilot. It does **not** yet authorize either planned combination or the
operator-difference intervention. Prefix-causal generation remains contingent on the
fixed-sequence no-op, hook-lifecycle, perturbation-control, and provenance gates.

## Fixed-sequence feasibility protocol

The next bounded job uses 18 Target-generated frozen V3 sequences: the first frozen
example from each of the first six sport domains for risk and matched sports-control,
plus the first six neutral IDs. This rule was fixed without inspecting intervention
outcomes. Both Base and Target receive the same exact token IDs.

For each of 3228, 290, 2445, 508, and 378, each model runs candidate ablation, an active
component control, and a norm-matched orthogonal-output control. The active mappings are
3228→1591, 290→310, 2445→960, 508→320, and 378→131. They were selected using general
fire prevalence, Target-generated response support and relative activation position,
total mechanism beta, matrix-locus profile, and weak V3 group contrasts--never P6
outcomes. There is no equally early active match for 3228, so that mismatch is explicit;
its norm control retains exactly the candidate gate, read vector, positions, and output
norm.

The job also requires an alpha-zero no-op, exact clean replay after hook removal,
unchanged matrix-weight audits, zero perturbation at gate-off tokens, finite metrics, all
seven matrix records, and exact frozen provenance. It reports teacher-forced KL, logit
RMS, maximum logit change, observed-next-token log-probability change, temporal regions,
gate-active/inactive subsets, per-matrix perturbation magnitude, runtime, peak VRAM, and
process RSS. Expected active runtime is approximately 8--25 minutes on one B200; the
one-hour allocation is a failure margin, not an estimate.

### Fixed-sequence execution record

- Script: `slurm/multimodel/qwen25_aspd_p6_fixed_feasibility.sbatch`.
- First attempt `46813108` started immediately but failed after 2 minutes 14 seconds
  with exit `1:0`, before any intervention, because the engineering weight audit read a
  version counter from a tensor created under `torch.inference_mode()`. It produced no
  scientific artifact. The audit now supports such tensors, the outer orchestration uses
  `no_grad`, and a regression test covers the failure (`21 passed`).
- Retry job: `46813296` at `2026-10-20T03:57:42-04:00` from commit `0d3e876`,
  with no dependency.
- Requested resources: one B200, 14 CPUs, 128 GB RAM, one-hour limit.
- Observed initial retry state: `RUNNING` on `c0910a-s5`.
- Intended artifact:
  `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_p6_multicomponent_pilot/fixed_sequence/job_46813296/fixed_sequence_feasibility.json`.
- Retry logs: `slurm/logs/mm-q25-aspd-p6-fixed-46813296.out` and
  `slurm/logs/mm-q25-aspd-p6-fixed-46813296.err`.
- Expected completion window, excluding unexpected I/O/model-download delay: roughly
  `2026-10-20 04:06--04:23 EDT`. Check after 04:23; the hard Slurm deadline is
  approximately 04:58 EDT.

No prefix-generation, judge, combination, operator-difference, retraining, cache, or
dependent job was submitted. Exit code, integrity marker, no-op metrics, perturbation
scale, and resource measurements must be inspected before the next stage is authorized.

## Fixed-sequence feasibility result

Retry `46813296` completed in 4 minutes 48 seconds with exit `0:0`. Both completion
markers were printed and the 2.3 MB artifact passed the fail-closed integrity check.
Python runtime was 248.93 seconds, peak allocated CUDA memory was 49.80 GB, and process
maximum RSS was 15.44 GB. Alpha-zero and post-hook clean replay both had exactly zero
maximum logit difference for Base and Target. All Qwen matrix checksums were unchanged,
all seven hooks fired in every arm, and every gate-off perturbation was exactly zero.

The Target teacher-forced mean KL measurements are:

| Candidate | Risk | Sports control | Neutral | Risk - control | Active-control risk | Norm-control risk |
|---:|---:|---:|---:|---:|---:|---:|
| 3228 | 0.001018 | 0.000967 | 0.001846 | +0.000051 | 0.000946 | 0.001429 |
| 290 | 0.001183 | 0.001060 | 0.000967 | +0.000123 | 0.001128 | 0.001045 |
| 2445 | 0.000480 | 0.000850 | 0.000660 | -0.000371 | 0.019170 | 0.000405 |
| 508 | 0.003184 | 0.002605 | 0.001196 | +0.000579 | 0.001627 | 0.002730 |
| 378 | 0.001495 | 0.001307 | 0.001327 | +0.000188 | 0.001529 | 0.002069 |

These are logit effects on frozen continuations, not behavioral scores. Component 508
has the strongest Target risk-versus-neutral separation and exceeds its active control,
but it only modestly exceeds the norm control. Components 3228, 290, and 378 are mostly
comparable to at least one matched control on Target. Component 2445 remains a
conclusion/post-decision comparator. On Base, 3228 produces a larger risk KL (0.007366)
than its active (0.001718) and norm (0.003377) controls, which is a model-asymmetric
observation worth preserving but not yet a behavioral mediation result.

Every selected sequence had at least one causal-gate fire for every candidate. Across
the 18 rows, mean fires per sequence were 2.7 for 3228, 11.6 for 290, 3.0 for 2445,
28.2 for 508, and 42.4 for 378. For 2445 those fires are prompt/conclusion dominated,
so raw support does not rescue its temporal limitation.

The no-op, hook, provenance, finite-metric, resource, and perturbation gates therefore
authorize a small prefix-causal autoregressive feasibility run for the two primary
candidates 3228 and 290 plus the strongest fixed-sequence comparative candidate 508.
This does not authorize 3228+290, 3228+2445, operator-difference, or any causal claim;
individual behavioral outputs and controls must be evaluated first.

## Prefix-generation feasibility execution

- Script: `slurm/multimodel/qwen25_aspd_p6_prefix_generation.sbatch`.
- Job: `46814945`, submitted at `2026-10-20T04:08:07-04:00` from commit
  `ee7976e`, with no dependency.
- Initial state: briefly `PENDING` with no blocking reason, then `RUNNING` on
  `c1101a-s5`.
- Resources: one B200, 14 CPUs, 128 GB RAM, one-hour limit.
- Scope: 18 prompt-only inputs; components 3228, 290, and 508; clean Base/Target,
  Target alpha-zero, and candidate/active/norm arms on both models; greedy decoding for
  at most 64 new tokens.
- Every patched decoding step recomputes clean Base and clean Target activations on the
  same actual current branch prefix. No response from the frozen V3 artifact supplies a
  future gate after a branch diverges.
- Intended artifact:
  `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_p6_multicomponent_pilot/prefix_generation/job_46814945/prefix_generation_feasibility.json`.
- Logs: `slurm/logs/mm-q25-aspd-p6-prefix-46814945.out` and
  `slurm/logs/mm-q25-aspd-p6-prefix-46814945.err`.
- Estimated active runtime is 12--35 minutes based on the 4-minute fixed-sequence run;
  check about 35 minutes after the job enters `RUNNING`. Queue delay is not included.

No behavioral judge, combination, operator-difference, retraining, cache, or dependent
job was submitted. Those stages remain scientifically gated on inspecting this artifact.


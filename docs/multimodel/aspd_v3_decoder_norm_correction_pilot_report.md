# ASPD v3 activation-decoder norm correction and normalized-target pilot report

Date: 2026-11-07

Branch: `aspd-v3-decoder-norm-correction`

Implementation/evaluation commit: `654feac6363b6f6dceb153e3b71eeed0afbbd508`

Historical comparison commit: `35711f9daa4629aea20c579f909db13501c56a52`

## Executive decision

`CORRECTION_VALIDATED`

The historical normalized-target implementation enforced an activation-decoder
unit-norm constraint that was not part of the earliest recoverable
researcher-approved multimodel mathematical design. Removing that constraint,
while preserving the approved objective and all other scientific settings,
eliminated the near-zero gate-firing collapse in the matched 10M-exposure pilot.
Held-out full-prefix normalized FVU improved from approximately 1.00--1.04 to
0.67--0.72, and the pooled held-out internal FVU improved from 2.568 to 0.600.

This is not authorization for a 500M run. Although each held-out anchor now has
32 positive firings, feature use remains extremely concentrated: PRE and MID
have event-entropy effective dictionary sizes of only 47.4 and 49.7, and the
top 1% of feature IDs account for more than 99.9% of firing events. The
correction fixes the catastrophic absorbing state; it does not establish a
healthy 8,192-feature dictionary.

## A. Mathematical review

The detailed source-by-source review is in
`docs/multimodel/aspd_v3_mathematical_specification_audit.md`.

### Authority and limitation

The original verbatim researcher prompt was not present in the repository. Its
authentication hash was available, but a hash cannot recover text. Therefore,
the audit did not invent missing requirements. It used the earliest committed
multimodel specification and subsequent versioned documents as the recoverable
authority:

- the earliest multimodel deviation record explicitly states that decoder-row
  normalization is not used;
- the original multimodel method uses learned decoder norms in component
  analysis and BatchTopK ranking;
- the initial ASPD v3 implementation had no activation-decoder projection;
- unit normalization was introduced later as a paper-derived implementation
  decision and was subsequently copied into the Granite notation.

The researcher clarification in this task resolves any remaining ambiguity:
activation-decoder vectors are unconstrained trainable parameters. Their norms
may be measured post hoc but are neither projected nor penalized.

### Corrected mathematical statement

For each model, site and layer, the activation decoder is

\[
D_{n,s,\ell}=[d_{n,s,\ell,1},\ldots,d_{n,s,\ell,C}],
\]

with no constraint \(\lVert d_{n,s,\ell,c}\rVert_2=1\). The treatment remains

\[
\tilde r = r/\max(\lVert r\rVert_2,\epsilon),\qquad
\hat{\tilde r}^{(k)}=b+\sum_{c\le k}g_c d_c,
\]

and the six-prefix FVU, AuxK, internal rank-one objective, coefficients and
model/site aggregation are unchanged. Decoder norms remain valid read-only
diagnostics.

### Discrepancy resolution

| Element | Historical implementation/notation | Approved specification | Resolution |
|---|---|---|---|
| Activation decoder | Every column projected to L2 norm 1 | Magnitudes learned by the objective | Projection disabled for the new family |
| Initialization | Decoder normalized immediately | Native controlled initialization retained | New mode skips initialization projection |
| Optimizer step | Projection after every `optimizer.step()` | No post-step constraint | New mode makes the activation-decoder projection a no-op |
| Gradient projection | Radial component could be removed | Full decoder gradient permitted | Disabled for the unconstrained mode |
| Checkpoint restore | Normalization behavior inherited from model configuration | Restore must preserve learned norms | Mode is fingerprinted and restored without projection |
| Norm analysis | Norm was degenerate at 1 | Norm is a post-hoc quantity | Reporting preserved and expanded |
| Rank-one writes | Separate normalization behavior | Not changed by this correction | Preserved |
| Final loss | Internal + activation + 1/32 AuxK | Same | Preserved exactly |

The previous Granite Draft v4 equation and equation-to-code map were corrected
without adding a new objective.

## B. Code review and correction

The detailed mapping is in
`docs/multimodel/aspd_v3_decoder_norm_code_audit.md`.

### Where normalization occurred

`ActivationReconstructionBank` normalized activation-decoder columns during
initialization. The training loop subsequently called decoder normalization
after each optimizer update, and the helper could also project radial gradient
components. These operations constrained optimization even though they were not
terms in the final loss.

### Minimal implementation change

The implementation adds an explicit activation-decoder normalization mode:

- historical configurations retain `unit_norm` for reproducibility;
- the corrected experiment family uses `none`;
- `none` skips initialization normalization, radial-gradient projection and
  post-update projection;
- read-only norm summaries are retained;
- the normalization mode is part of configuration validation, experiment
  identity and checkpoint compatibility;
- rank-one factor handling is unchanged.

No activation function, sparsity rule, learning rate, loss coefficient,
Matryoshka boundary, model, layer, target matrix, target-normalization formula
or internal-reconstruction formula was changed.

### Changed files

The implementation changes relative to `35711f9` cover:

- activation reconstruction and model-owned configuration wiring;
- training/evaluation decoder-norm telemetry;
- a new isolated unconstrained-decoder config family;
- fail-closed preflight, pilot and evaluation Slurm entry points;
- regression tests;
- corrected notation and mathematical/code audits.

The complete list is available from:

```text
git diff --name-status 35711f9daa4629aea20c579f909db13501c56a52..654feac6363b6f6dceb153e3b71eeed0afbbd508
```

Historical experiment configs and checkpoints were not overwritten.

## C. Verification before execution

### Local verification

| Check | Result | Evidence |
|---|---|---|
| Granite normalized-target test file | PASS, 29 tests | `pytest -q tests/test_multimodel_v3_granite_normalized_target.py` |
| Relevant ASPD v3 suite | PASS, 83 tests; one distributed test deselected | bounded v3 pytest selection |
| Distributed CPU/Gloo test | PASS | separate bounded run, 123.26 s |
| Relevant Ruff checks | PASS | modified Python files and test file |
| Compileall | PASS | modified ASPD modules |
| Slurm shell syntax | PASS | all three new scripts |
| `git diff --check` | PASS | no whitespace errors |

One initial test invocation used a nonexistent path and was corrected. The first
sandboxed Gloo attempt was denied by the environment; the same bounded test was
rerun with the required permission and passed. A repository-wide Ruff scan also
showed a pre-existing import-order issue in `serve_analysis.py`; it is unrelated
to this branch and was not changed.

### Decoder-specific assertions

The tests demonstrate that an activation decoder initialized away from norm 1
can move up or down under an optimizer update and is not projected back to 1.
Gradients reach decoder weights, norm diagnostics remain available, checkpoint
restore preserves the mode, and historical unit-norm configurations retain
their old behavior.

### GPU preflight

Job `50135740` completed with exit `0:0` in 3:27. The preflight validated source
immutability, checkpoint save/reload, optimizer restoration and a three-step
update. Mean initial decoder norms were approximately 0.306 for Gemma and 0.289
for SmolLM2/Granite, and changed after optimization rather than being reset to
1. The immutable preflight report SHA-256 is
`8d3e58ef981401e2a03d9b88e60f69ce26f27377fbbbf214ca9d80e063597c8b`.

An earlier fail-closed attempt, job `50135074`, correctly rejected incompatible
historical preflight provenance and produced no training checkpoint.

## D. Pilot execution

### Provenance

| Field | Value |
|---|---|
| Pilot job | `50138545`, `COMPLETED`, exit `0:0`, 44:27 |
| Evaluation job | `50148074`, `COMPLETED`, exit `0:0`, 50:47 |
| Optimizer steps | 1,221 |
| Accepted exposures | 10,002,432 |
| Held-out accepted anchors | 100,352 from 323 source documents |
| Git SHA | `654feac6363b6f6dceb153e3b71eeed0afbbd508` |
| Config fingerprint | `04599059ec494ebe485d5e2a17780eb983a049404b2bac552f3b2c55e886dc6f` |
| Config SHA-256 | `6fbbace8188cb0267b36f8ef99f9b7b3345267f0f93b8ba6ff2077e5b0b631a4` |
| Alignment manifest SHA-256 | `e3fb039d2f12f3f81fef40fa064eb5b189dcfb8c99ba8fd0b6c0bccfe7a50777` |
| Dataset manifest SHA-256 | `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f` |
| Checkpoint manifest SHA-256 | `246094cafbd01b05a77762936902a70e92bd11df9b26b602c6b6ace1ef53903a` |
| Model order | Gemma, SmolLM2, Granite |

All 168 native targets passed source-correctness checks with maximum native
projection absolute error 0. No nonfinite loss or gradient was reported.

### Runtime

Median optimizer-step time was 1.988 s and p95 was 2.297 s, corresponding to
4,074 accepted anchors/s. The preference of under 2 s is met at p50 but not at
p95; no further performance optimization was mixed into this scientific
correction.

### Training trajectory

| Step | Activation loss | Internal loss | AuxK | PRE positive/anchor | MID positive/anchor |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.486 | 2.164 | 0 | 32.0 | 32.0 |
| 50 | 1.549 | 1.954 | 0 | 32.0 | 32.0 |
| 128 | 1.246 | 1.376 | 0 | 32.0 | 32.0 |
| 256 | 1.044 | 1.066 | 0.992 | 32.0 | 32.0 |
| 300 | 1.014 | 0.989 | 0.593 | 32.0 | 32.0 |
| 500 | 0.892 | 0.776 | 0.579 | 32.0 | 31.70 |
| 800 | 0.840 | 0.696 | 0.713 | 32.0 | 32.0 |
| 1,000 | 0.831 | 0.679 | 0.715 | 32.0 | 32.0 |
| 1,221 | 0.813 | 0.656 | 0.701 | 32.0 | 32.0 |

The historical constrained treatment started at loss 885.125 and ended with
near-zero positive gate activity. The corrected treatment started an order of
magnitude lower, decreased smoothly, and AuxK remained active at the final
step instead of becoming identically zero after step 500.

Raw logits still became predominantly negative (approximately 99.5% by the
end), but BatchTopK retained 32 strictly positive events per anchor. This is a
warning about feature competition and concentration, not the historical
all-zero gate state.

### Decoder magnitudes

Mean decoder norms changed from approximately 0.306/0.289 at initialization to:

| Model | PRE mean | MID mean | Observed min--max |
|---|---:|---:|---:|
| Gemma | 0.3152 | 0.3081 | 0.0963--0.7081 |
| SmolLM2 | 0.2972 | 0.2905 | 0.1004--0.6092 |
| Granite | 0.2977 | 0.2910 | 0.1174--0.6027 |

The non-degenerate distributions and movement establish that decoder magnitude
was trainable and not silently projected to 1.

## E. Held-out results

### Full-prefix activation reconstruction

These are 8,192-component reconstructions, not six-prefix means.

| Model/site | Historical normalized FVU | Corrected normalized FVU | Historical oracle-original FVU | Corrected oracle-original FVU |
|---|---:|---:|---:|---:|
| Gemma PRE | 1.0414 | 0.6951 | 1.0254 | 0.6879 |
| Gemma MID | 1.0424 | 0.7002 | 1.0244 | 0.6682 |
| SmolLM2 PRE | 1.0309 | 0.7166 | 0.7496 | 0.7740 |
| SmolLM2 MID | 1.0347 | 0.7179 | 0.7530 | 0.0262 |
| Granite PRE | 1.0052 | 0.6678 | 1.2488 | 1.2060 |
| Granite MID | 1.0014 | 0.6704 | 1.2442 | 0.0333 |

Normalized-space full FVU improves materially for every model/site. The
oracle-original metric is heterogeneous: Gemma and both MID sites improve
strongly, Granite PRE improves only slightly, and SmolLM2 PRE is slightly worse.
The oracle metric uses the true target norm and is not a self-contained
magnitude prediction.

Corrected six-prefix normalized Matryoshka means range from 0.779 to 0.822,
while full-prefix values range from 0.668 to 0.718. This confirms that training
loss and final full reconstruction must not be conflated.

### Internal reconstruction

| Model/site | Historical pooled FVU | Corrected pooled FVU | Corrected equal-matrix mean FVU |
|---|---:|---:|---:|
| Gemma PRE | 1.7609 | 0.5274 | 0.5469 |
| Gemma MID | 1.7677 | 0.6605 | 0.6820 |
| SmolLM2 PRE | 3.3739 | 0.6303 | 0.6462 |
| SmolLM2 MID | 1.2492 | 0.8079 | 0.7979 |
| Granite PRE | 3.7466 | 0.4157 | 0.4917 |
| Granite MID | 1.6849 | 0.7244 | 0.7403 |

Across all 168 matrices, pooled FVU improved from 2.5676 to 0.6003. The
balanced mean of the six model/site equal-matrix means improved from 2.0453 to
0.6508. This is held-out evidence; the step-1,221 training internal loss was
0.6562.

### Gate and dictionary health

| Metric | Historical PRE | Corrected PRE | Historical MID | Corrected MID |
|---|---:|---:|---:|---:|
| Positive firings/anchor | 0.00860 | 32.0 | 0.00151 | 32.0 |
| Features firing at least once held out | 41 | 983 | 18 | 1,264 |
| Event-entropy effective features | 33.37 | 47.42 | 9.06 | 49.72 |
| Recently dead at training end | 8,135 | 5,674 | 8,040 | 4,442 |
| Top-1% event share | 100% | 99.9436% | 100% | 99.9022% |

The categorical gate collapse is eliminated: all anchors have the intended 32
positive firing events, and substantially more feature IDs appear at least
once. Nevertheless, useful diversity remains poor. Roughly 47--50 effective
features dominate an 8,192-feature dictionary, and more than 99.9% of events
fall in the top 1% of IDs. This is a separate scientific failure mode that must
be addressed or accepted explicitly before scaling.

## F. Root-cause assessment

1. **Did the previous implementation incorrectly enforce activation-decoder
   unit norm?** Yes, relative to the recoverable original multimodel
   specification and the researcher's explicit clarification.
2. **Did the correction permit magnitudes to adapt?** Yes. Norm distributions
   remained non-unit and changed during optimization.
3. **Did magnitudes actually change?** Yes, with broad final ranges and
   model/site-specific means.
4. **Did gate collapse persist?** The near-zero firing collapse did not persist.
   A severe concentration problem remains.
5. **Did reconstruction improve?** Yes in normalized full-prefix FVU for every
   model/site and dramatically in internal FVU. Oracle-original results are
   mixed by model/site.
6. **Is the normalized-target objective trainable under the corrected
   architecture?** At the 10M single-seed scale, yes: losses are finite and
   declining, firing remains 32/anchor, and held-out FVU is below 1.
7. **What remains missing before claiming unit norm was the sole cause?** A
   paired multi-seed comparison and a design that separately controls initial
   decoder scale versus continued post-step projection. The current
   intervention necessarily changes both. It establishes that the constraint
   was a material causal contributor, not that it was the only contributor.

The most defensible conclusion is that the unapproved unit-norm constraint was
a major cause of the historical catastrophic collapse. It is not a complete
explanation of dictionary concentration, and the corrected pilot does not yet
justify a full 500M run.

## G. Artifacts and remaining boundary

Checkpoint:

```text
/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/reliability/normalized_target/runs/job_50138545/checkpoints/step-000001221
```

Training artifacts:

```text
/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/reliability/normalized_target/runs/job_50138545
```

Held-out evaluation artifacts:

```text
/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/evaluation/normalized_target/job_50148074/heldout
```

The checkpoint is approximately 127.3 GB. Artifacts were preserved. No full
training was submitted, no historical checkpoint was modified, and no branch
was merged.

Recommended next gate: investigate fixed-index/feature concentration and run a
paired additional seed or a short controlled initialization-vs-projection
ablation. Do not extrapolate the 10M result directly to 500M.

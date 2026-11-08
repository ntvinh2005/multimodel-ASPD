# ASPD v3 encoder-logit drift audit

Date: 2026-11-07

Branch: `aspd-v3-decoder-norm-correction`

Source checkpoints: step 1,000 and step 1,221 from training job `50138545`

## Executive finding

The proposed correction to the mechanism description is supported:
BatchTopK does not directly reduce the candidate set to about 39. The final
linear readout and hard ReLU produce only about 39 positive candidates per
anchor, and BatchTopK then selects from that already-starved set.

Checkpoint-only analysis rules out readout bias as the main explanation of the
late 99.52% negative-logit state. At both sites, mean bias is only about
`-0.007` to `-0.008`, and only 58--60% of feature biases are negative. Raw
logits are about 99.52% negative with means between `-0.75` and `-1.55`.
Therefore the negative inner-product term \(w_c^\top h_a\), not \(b_c\),
dominates late logit negativity.

Readout weights change little globally from step 1,000 to 1,221, while winner
rows retain Adam first moments thousands of times larger than held-out-inactive
rows. This directly supports unequal sparse-gradient allocation. It does not
yet establish whether the late \(w^\top h\) state is primarily caused by
earlier readout rotation or by representation drift, because fixed-anchor
representations were not retained.

The minimum decisive next measurement is a read-only, fixed-anchor,
cross-checkpoint encoder replay. No GPU job was submitted in this audit.

## 1. Verified computation path

The encoder readout is exactly:

\[
u_{a,c}=w_c^\top h_a+b_c,
\qquad
s_{a,c}=\operatorname{ReLU}(u_{a,c}).
\]

`FeatureReadout.forward_with_logits()` mean-pools the Cross-Model Transformer
tokens, applies one linear layer, and immediately applies ReLU. There is no
intermediate normalization or activation. Model-owned training passes those
post-ReLU scores to both BatchTopK and AuxK.

The upstream representation path is:

1. model-specific packer and Intra-Model Transformer on each owner rank;
2. differentiable three-rank gather;
3. shared input adapter and Cross-Model Transformer;
4. mean pooling and shared readout.

Thus a readout row can receive little direct gradient while its logit changes
because the shared and local representation functions continue changing for
other features.

## 2. Provenance and method

The audit selectively loaded only the shared readout weight, bias and Adam
first-moment tensors from the step-1,000 and step-1,221 distributed
checkpoints. All loaded shared tensors were bitwise equal across ranks 0, 1 and
2.

| Field | Value |
|---|---|
| Training job | `50138545` |
| Training Git SHA | `654feac6363b6f6dceb153e3b71eeed0afbbd508` |
| Config fingerprint | `04599059ec494ebe485d5e2a17780eb983a049404b2bac552f3b2c55e886dc6f` |
| Step-1,221 checkpoint manifest SHA-256 | `246094cafbd01b05a77762936902a70e92bd11df9b26b602c6b6ace1ef53903a` |
| Readout weight shape | `[8192, 512]` per site |
| Readout bias shape | `[8192]` per site |
| Held-out firing reference | evaluation job `50148074` |

The endpoint raw-logit means come from each step's training diagnostics. The
training anchors differ across steps, so they support aggregate scale
accounting, not fixed-anchor sign attribution.

Machine-readable outputs are in
[`analysis/aspd_v3_encoder_logit_drift_1000_1221`](analysis/aspd_v3_encoder_logit_drift_1000_1221/).

## 3. Bias versus inner-product scale

Because bias is constant across anchors for a feature,

\[
\mathbb E_{a,c}[w_c^\top h_a]
=
\mathbb E_{a,c}[u_{a,c}]-\mathbb E_c[b_c].
\]

| Site/step | Mean raw logit | Negative raw logits | Mean bias | Negative biases | Derived mean \(w^\top h\) |
|---|---:|---:|---:|---:|---:|
| PRE 1,000 | -1.55142 | 99.531% | -0.00714 | 57.837% | -1.54427 |
| PRE 1,221 | -1.24966 | 99.521% | -0.00734 | 58.008% | -1.24232 |
| MID 1,000 | -0.75319 | 99.540% | -0.00774 | 58.936% | -0.74545 |
| MID 1,221 | -0.78281 | 99.523% | -0.00826 | 59.521% | -0.77455 |

Bias q01--q99 at step 1,221 is approximately `[-0.054, 0.038]` PRE and
`[-0.056, 0.038]` MID. In contrast, even the sampled raw-logit q99 is negative
at step 1,221 (`-0.630` PRE and `-0.222` MID). Bias alone therefore cannot
explain the broad negative-logit distribution.

From step 1,000 to 1,221:

- PRE mean bias changes by only `-0.000193`; 15 features cross from nonnegative
  to negative bias and one crosses in the reverse direction.
- MID mean bias changes by `-0.000521`; 49 cross to negative and one crosses to
  nonnegative.
- MID bias drift is only about 1.8% of its raw-logit mean drift. PRE raw-logit
  mean becomes less negative despite slightly more-negative mean bias.

Conclusion: **readout bias is refuted as the primary cause of the late
candidate starvation**. A fixed-anchor replay is still required to count exact
sign flips attributable to bias near zero.

## 4. Readout-weight drift

| Metric, step 1,000→1,221 | PRE | MID |
|---|---:|---:|
| Relative Frobenius change | 0.653% | 1.443% |
| Mean row cosine | 0.999979 | 0.999896 |
| Minimum row cosine | 0.998314 | 0.997219 |
| Mean row norm at 1,000 | 0.576469 | 0.576355 |
| Mean row norm at 1,221 | 0.576475 | 0.576417 |
| Exactly unchanged rows | 2,000 | 1,004 |
| Exactly unchanged bias entries | 4,016 | 1,586 |

The late readout is globally stable in norm and direction. This makes a large
step-1,000→1,221 readout rotation an unlikely sole explanation of the already
collapsed state. It does not rule out:

- larger readout changes before step 1,000;
- small, feature-aligned changes that move near-zero logits across zero;
- interactions between small \(\Delta W\) and large representation changes.

No checkpoint before step 1,000 exists, so the parameter trajectory during the
observed step-128→500 onset cannot be recovered.

## 5. Sparse-gradient allocation

Held-out firing is used only to define winner/inactive groups; it is not treated
as an independent sample.

| Site | Quantity | Top 82 | Held-out inactive | Ratio |
|---|---|---:|---:|---:|
| PRE | Mean row \(\lVert\Delta W\rVert_2\) | 0.004184 | 0.001251 | 3.35x |
| MID | Mean row \(\lVert\Delta W\rVert_2\) | 0.010558 | 0.005293 | 1.99x |
| PRE | Adam weight first-moment row norm | 8.51e-4 | 1.25e-7 | ~6,816x |
| MID | Adam weight first-moment row norm | 6.96e-4 | 1.94e-7 | ~3,596x |
| PRE | Adam bias first-moment magnitude | 2.34e-5 | 6.36e-9 | ~3,674x |
| MID | Adam bias first-moment magnitude | 2.97e-5 | 9.25e-9 | ~3,207x |

The association between held-out firing and late readout update is positive:
Spearman correlation between `log1p(fires)` and row delta norm is 0.401 PRE and
0.168 MID; for absolute bias delta it is 0.422 and 0.222.

This is direct evidence for the proposed feedback mechanism's gradient
allocation component: winner rows continue to receive optimization signal,
whereas many inactive rows have near-zero current Adam moments. It is not by
itself evidence that readout-row updates, rather than changing \(h\), caused
their logits to become negative.

Late row and bias drift are fairly similar by Matryoshka band. The step
1,000→1,221 interval therefore cannot attribute the earlier band advantage to
late readout drift. Matryoshka's cumulative effect must be tested with the
matched loss ablation already proposed.

## 6. What is established and unresolved

| Question | Current answer | Evidence status |
|---|---|---|
| Does BatchTopK directly limit candidates to 39? | No; ReLU has already reduced positive candidates | Verified code and logs |
| Is negative readout bias the main late cause? | No | Refuted by magnitude/distribution |
| Is \(w^\top h\) the dominant late negative term? | Yes | Strong aggregate evidence |
| Do winner rows receive much more current gradient? | Yes | Checkpoint Adam moments |
| Did \(W\) or \(h\) initiate collapse? | Unknown | Fixed-anchor representations absent |
| Did the decisive change occur before step 1,000? | Almost certainly, given step-500 logs | Correlational timeline; early checkpoints absent |

The best-supported current mechanism is:

1. early sparse competition and Matryoshka opportunity differences concentrate
   direct reconstruction gradients;
2. winning readout/decoder paths and shared representation modules adapt;
3. the resulting \(w^\top h\) distribution becomes negative for most features;
4. hard ReLU removes direct recovery gradients for negative rows;
5. AuxK can promote only dead features whose post-ReLU score is still positive,
   and does not restore meaningful diversity.

Steps 1--3 remain a supported causal model, not a fully identified temporal
causal decomposition.

## 7. Minimum fixed-anchor replay

### Scope

Use one fixed validation pool of 2,048 aligned anchors on three B200 GPUs.
Capture frozen-backbone residuals once, then evaluate encoder-only state from
both checkpoints. Do not allocate rank-one banks or activation decoders, run an
optimizer, call backward, modify checkpoints or train.

Estimated request: 15--20 minutes, approximately 0.75--1 B200 GPU-hour total.
This is a planning estimate, not measured runtime.

### Four-factor swap matrix

Let checkpoint 0 be step 1,000 and checkpoint 1 step 1,221:

- \(L_i\): three model-specific packer/Intra-Transformer states;
- \(C_j\): shared input-adapter/Cross-Transformer state;
- \(W_k\): readout weight;
- \(b_l\): readout bias.

Define

\[
h_{ij}(a)=\operatorname{MeanPool}(C_j(L_i(a))),
\]

\[
u_{ijkl}(a,c)=W_k[c]^\top h_{ij}(a)+b_l[c],
\qquad i,j,k,l\in\{0,1\}.
\]

Evaluate all 16 combinations. Important endpoints/counterfactuals are:

| Combination | Meaning |
|---|---|
| `0000` | Exact step-1,000 encoder |
| `0001` | Bias drift only |
| `0010` | Readout-weight drift only |
| `0100` | Shared adapter/cross drift only |
| `1000` | Local packer/intra drift only |
| `1100` | Complete representation drift with old readout |
| `0011` | Complete readout drift on old representation |
| `1111` | Exact step-1,221 encoder |

For continuous logits, report the exact symmetric decomposition

\[
\Delta u=(b_1-b_0)
+(W_1-W_0)\frac{h_0+h_1}{2}
+\frac{W_0+W_1}{2}(h_1-h_0).
\]

This splits the bilinear \(\Delta W\Delta h\) interaction equally. For
nonlinear negative-fraction, candidates-per-anchor and sign-death metrics, use
exact four-factor Shapley attribution over the 16 counterfactuals.

### Required outputs

Report separately for PRE/MID, feature band and top/rare groups:

- raw-logit mean, standard deviation and quantiles;
- negative fraction and positive candidates/anchor;
- step-1,000-positive→step-1,221-negative and reverse crossings;
- bias, readout-weight, shared-cross and local-encoder Shapley contributions;
- native BF16 versus FP32 sign disagreement near zero;
- ReLU and BatchTopK support only as downstream secondary outcomes.

Verify `0000` and `1111` against native `forward_with_logits`, use
`eval()`, `no_grad()` and `update_statistics=False`, and stream feature chunks
rather than saving all 16 dense logit tensors. Expected outputs should remain
well below 100 MB.

### Interpretation gate

- Bias-dominant Shapley attribution would reopen bias initialization/update
  investigation, although checkpoint-only evidence makes this unlikely.
- Readout-weight dominance would support direct sparse row-training as the main
  late mechanism.
- Local/shared representation dominance would support the proposed shared-\(h\)
  drift mechanism and imply that changing Matryoshka alone may be insufficient.
- A large interaction would mean no single-module correction is justified.

Even this replay diagnoses only persistence and drift between steps 1,000 and
1,221. A future causal pilot must preserve checkpoints at 50, 128, 256 and 500
to identify initiation.

## 8. Local verification

- CPU selective checkpoint loading: PASS for both sites, both checkpoints and
  all three replicated ranks.
- Readout/bias/Adam-moment analysis CLI: PASS.
- Symmetric decomposition tests: PASS.
- Existing dictionary/AuxK tests used alongside the new tests: PASS.
- Ruff: PASS.
- No Slurm job, GPU allocation, checkpoint mutation or training was performed.

## Decision

`FIXED_ANCHOR_GPU_REPLAY_REQUIRED`

The current evidence is already sufficient to stop treating readout bias as
the leading explanation. It is not sufficient to choose between readout-weight
and representation drift, nor to establish which initiated the original
step-128→500 collapse.

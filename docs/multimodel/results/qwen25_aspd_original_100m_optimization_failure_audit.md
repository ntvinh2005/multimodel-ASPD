# Qwen2.5 two-model ASPD 100M: scientific optimization-failure audit

**Audit date:** 2026-10-25  
**Code audited:** `763fe7c1c0fec610933e598c7f006e78c4ce0bcc`  
**Run:** `out/multimodel/runs/qwen25_aspd_original_100m/`  
**Scope:** post-training diagnosis only; no retraining, hyperparameter change, Slurm submission,
checkpoint mutation, or P6 mutation was performed.

## A. Bottom line

The evidence does **not** support a corrupt run, faulty evaluation, generally broken encoder, or
simple “not enough tokens” explanation. It supports two objective/optimization mechanisms:

1. **The internal objective cannot train the shared encoder's selection policy.** The component
   gate is a hard boolean support. `L_internal` gives the encoder gradient `None`, while
   `L_activation` trains the encoder and activation decoders. This is intentional and matches the
   original ASPD implementation; it is an optimization consequence, not a newly discovered code
   bug. The internal U/V and transcoder biases can only adapt to the support selected for residual
   reconstruction.
2. **Parameter-only Base→Target difference is not an optimized target.** Base and Target have
   separate U/V/`b_dec`/`b_out` banks. Their internal FVUs are averaged, but there is no loss on
   `(W_target-W_base)X`, no factor-alignment constraint, and no requirement that their reconstruction
   errors cancel. The true parameter delta is tiny, so modest disagreement between two much larger
   per-model reconstructions overwhelms it.

Matryoshka training strongly amplifies the first mechanism: the first 6.25% of components carries
69.8% of held-out fires and 83.3% of standalone computation-contribution energy, while the final
50% carries only 0.28% of fires and 0.11% of energy. Thus nominal `C=36,864` is not the effective
capacity available to the internal computation.

### Supported-cause ranking

| Explanation | Finding | Evidence strength |
|---|---|---|
| Internal loss cannot change encoder support | Supported and reproduced with autograd; upstream behavior matches | Direct |
| Independent per-model reconstruction errors swamp tiny ΔW signal | Supported quantitatively; no delta term exists | Direct |
| Matryoshka prefix concentration reduces effective internal capacity | Supported by all-component held-out statistics | Direct |
| Internal optimization had not converged | Mostly contradicted as a primary explanation; tail is noise-dominated/plateaued | Strong |
| AuxK materially affected training | Contradicted; active in 35/2,442 logged rows with negligible weighted scale | Direct |
| Encoder is generally faulty | Contradicted by residual FVU, support agreement, and stable threshold | Strong |
| Evaluation/arithmetic bug | Contradicted by hashes, finiteness, train/eval agreement, and delta identities | Direct |
| `K=32` or rank-one expressivity is independently limiting | Plausible but not isolated by existing artifacts | Undetermined |
| `R_mid` versus `X_O` mismatch is independently limiting | Mechanistically plausible; population measurement needs bounded GPU diagnostic | Undetermined |
| More training tokens alone would solve the issue | Not supported by the observed tail; cannot be ruled out counterfactually | Weak/undetermined |

## B. Authoritative evidence verification

At audit start, `main`, `origin/main`, and `origin/HEAD` were all at `763fe7c`; the working tree was
clean. The result package was rehashed independently:

- status `P6_HANDOFF_READY`;
- 26/26 manifest artifacts present;
- 26/26 sizes and SHA-256 values matched;
- final step 24,414 and 99,999,744 exposures;
- source weights unchanged, finite loss/gradients, DDP parameters equal;
- full frozen validation: 3,906 sequences and 1,999,872 positions.

Recomputed final checkpoint hashes:

| Artifact | SHA-256 |
|---|---|
| Rank 0 | `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6` |
| Rank 1 | `eda8b22938f2aa70ae946e00d4dc1a929012686748827f9e6006f5167cdacbd7` |
| Atomic marker | `00a9d9344c39b47f1d582bee78c491186437c34be8047a88f87abb7311da7d52` |
| Frozen config | `904a91d2ab2c63cafd5f62c9c012bcfbd418f72de8114fffa8f4d14469626c40` |
| Dataset manifest | `82fbb6a33b0e6104a165a4bd2f88526ee7175744de92de17bb8a2ee71dc04da8` |

All reference measurements in the task reproduce from the final JSON/PT/CSV artifacts. No value in
this report is inferred from a missing artifact.

## C. Effective mathematical objective

For model `n∈{Base,Target}`, token `t`, and component `c`, the shared preactivation is

```text
a_tc = ReLU( 1/2 Σ_n W_e^(n)(R_t^(n) - center^(n)) + b_e )
g^s_tc = a_tc · BatchTopKSupport(a)_tc
g_tc = 1[g^s_tc > 0]
```

The two models share `g^s`/`g`, but their encoder projections and all activation/component decoders
are model-specific. The encoder bias is shared. The center for model `n` is the same Parameter
object as its activation-reconstruction bias.

For the Matryoshka boundaries `M={0,2304,4608,9216,18432,36864}`:

```text
R_hat_m^(n) = center^(n) + Σ_(c<m) g^s_tc d_c^(n)
L_activation^(n) = 1/6 Σ_(m∈M) FVU(R^(n), R_hat_m^(n))
```

The empty-prefix term is included. Therefore a training activation loss near `1/6≈0.1667` is
compatible with a near-zero full-prefix FVU; it must not be compared directly with the full held-out
residual FVU.

For `X=X_O`, `Y=W_O X`, and the model-specific transcoder factors:

```text
Y_hat^(n) = b_out^(n)
          + Σ_c g_tc [v_c^(n)ᵀ(X_t^(n)-b_dec^(n))] u_c^(n)
L_internal^(n) = FVU(Y^(n), Y_hat^(n))
```

There is one matrix, so no additional matrix-axis scaling. The complete D0 objective is

```text
L = 1/2 Σ_n [L_internal^(n) + 1.0 · L_activation^(n)]
  + 0.03125 · L_AuxK
```

`L_AuxK` uses the top 2,048 currently-dead encoder preactivations to reconstruct the full activation
residual, averages over models, and is normalized by the original activation variance. It receives
no computation target. After backward, activation-decoder rows and component U rows are projected
to unit norm; radial U gradients are removed. V, `b_dec`, and `b_out` are unconstrained by that
projection.

### Gradient recipients by objective

The table reports L2 gradient norms from a deterministic small CPU fixture using the **actual
production classes and loss functions**. The numbers establish graph connectivity and relative
local scale; they are not claimed to be final-checkpoint population gradients.

| Parameter group | Activation only | Internal only | AuxK forced-dead path | Full, no dead feature |
|---|---:|---:|---:|---:|
| Encoder projection, Base input | 0.046718 | `None` | 1.52194 | 0.046718 |
| Encoder projection, Target input | 0.036540 | `None` | 1.29839 | 0.036540 |
| Shared encoder bias | 0.029735 | `None` | 1.20883 | 0.029735 |
| Base center / activation bias | 0.020117 | `None` | 0.459593 | 0.020117 |
| Target center / activation bias | 0.024472 | `None` | 0.440788 | 0.024472 |
| Base activation decoder | 0.079782 | `None` | 1.09776 | 0.079782 |
| Target activation decoder | 0.092841 | `None` | 1.20262 | 0.092841 |
| Base U / V | `None` / `None` | 0.174207 / 0.223739 | `None` / `None` | 0.174207 / 0.223739 |
| Target U / V | `None` / `None` | 0.320922 / 0.322098 | `None` / `None` | 0.320922 / 0.322098 |
| Base `b_dec` / `b_out` | `None` / `None` | 0.069813 / 0.054789 | `None` / `None` | 0.069813 / 0.054789 |
| Target `b_dec` / `b_out` | `None` / `None` | 0.105591 / 0.221244 | `None` / `None` | 0.105591 / 0.221244 |

With no dead feature, AuxK was exactly zero. Its encoder gradients existed as exact zeros and other
parameter gradients were `None`. The explicitly labeled forced-dead probe verifies the dormant
path without claiming it was active in the final state.

The result is unambiguous: `L_internal` cannot alter which components the shared encoder selects.
It only optimizes component factors and component biases conditional on the support supplied by the
activation objective.

## D. Original-ASPD parity

The comparison used upstream `tue147/weights-read-write` commit
[`47d12ed`](https://github.com/tue147/weights-read-write/tree/47d12edc229e334427ce6c4195dc5c44606b724a),
including [`aspd/losses.py`](https://github.com/tue147/weights-read-write/blob/47d12edc229e334427ce6c4195dc5c44606b724a/aspd/losses.py),
[`aspd/ci/aspd.py`](https://github.com/tue147/weights-read-write/blob/47d12edc229e334427ce6c4195dc5c44606b724a/aspd/ci/aspd.py),
and the [`Qwen3 ASPD config`](https://github.com/tue147/weights-read-write/blob/47d12edc229e334427ce6c4195dc5c44606b724a/configs/qwen3/aspd.yaml).

Parity findings:

- Both implementations use BatchTopK support, a binary internal component mask, continuous
  preactivation magnitudes for activation reconstruction, internal FVU, normalized Matryoshka
  activation loss, normalized AuxK, unit U/activation-decoder rows, `K=32`, pool 2,048, `C=36,864`,
  and the same five group fractions.
- An independent PyTorch check confirmed upstream's constructed BatchTopK support has no autograd
  graph. Its later lower-leaky clamp cannot recreate a path that the support construction removed.
  Thus the encoder-disconnected internal loss is upstream behavior too.
- The local two-model extension averages losses over the model axis and uses two encoder projections
  before their mean. This preserves objective scale but does not align the model-specific U/V banks.
- Local AuxK retains the reconstruction-residual gradient path, matching upstream. This distinction
  was immaterial here because AuxK was almost always inactive.

No mathematical-parity defect explaining FVU≈0.5 was found.

## E. Training trajectory and convergence classification

Mean logged internal FVU by step interval:

| Steps | Base | Target | Base activation loss | Target activation loss |
|---|---:|---:|---:|---:|
| 1–100 | 1.1743 | 1.2135 | 0.3619 | 0.3651 |
| 101–1,000 | 0.7392 | 0.7331 | 0.1722 | 0.1722 |
| 1,001–5,000 | 0.6312 | 0.6242 | 0.1711 | 0.1710 |
| 5,001–10,000 | 0.5950 | 0.5897 | 0.1706 | 0.1705 |
| 10,001–20,000 | 0.5792 | 0.5732 | 0.1703 | 0.1702 |
| 20,001–23,414 | 0.5690 | 0.5630 | 0.1702 | 0.1701 |
| 23,415–24,414 | 0.5629 | 0.5575 | 0.1701 | 0.1701 |

Classification:

- The objective was stable, not divergent or oscillatory beyond ordinary batch noise.
- Internal loss improved substantially early and slowly through the middle of training.
- From step 20,001 onward the fitted change was only −0.00535 Base and −0.00404 Target with
  `R²≤0.0015`. Over the final 1,000 steps the slopes were slightly positive with `R²≤0.0015`.
  The strongest supported label is **late plateau/noise-dominated tail**, not meaningful continued
  descent.
- Final held-out internal FVU (0.5016/0.4961 under JumpReLU) is close to, and slightly better than,
  the tail training BatchTopK losses. Support Jaccard is 0.9649. This does not suggest a large final
  train/validation gap.
- No longitudinal held-out series was logged, so “improving on train but not validation” cannot be
  tested over time.

In the final 1,000 logged steps, scalar loss was approximately 76.7% internal and 23.3% activation.
This does not mean the encoder was dominated by internal gradients: the recipient sets are largely
disjoint.

AuxK was nonzero in only 35 of 2,442 logged rows, from step 19,210 to 22,070, corresponding to one
dead component (`1/36,864`). Mean nonzero raw AuxK was 0.001922; after the `1/32` coefficient its
mean contribution was `6.01e-05`, negligible beside total loss around 0.73.

## F. Why residual FVU is excellent while `o_proj` FVU is about 0.5

The two objectives reconstruct different targets with different write mechanisms:

- Residual reconstruction uses continuous `g^s` magnitudes and a direct activation decoder trained
  jointly with the encoder.
- Internal reconstruction uses only binary support plus a signed read from `X_O`; its encoder
  support cannot react to internal error.

The residual encoder therefore succeeds at the task it directly optimizes: held-out FVU is
0.003803 Base and 0.003679 Target, with cosine about 0.998. This is evidence against a generally
faulty encoder.

The component banks also learn: internal FVU falls from about 1.2 to about 0.56 during training and
reaches 0.5016/0.4961 held out. But they are fitting `W_O X_O` under a support allocation optimized
for `R_mid`, not an internal-loss-optimized allocation. The final outputs remain well-conditioned
(actual RMS about 0.240; variance/energy checks pass), so FVU≈0.5 is not a denominator artifact.

## G. Matryoshka concentration and effective capacity

Standalone contribution energy is defined here as the sum across components of the squared
per-validation-token contribution RMS stored by the evaluator. It is not the energy of the summed
reconstruction because cross-component terms are not included.

| Feature group | Feature share | Active at least once | Fire share | Base energy share | Target energy share | Top-20 / Top-100 |
|---|---:|---:|---:|---:|---:|---:|
| 0–2,303 | 6.25% | 2,279/2,304 | 69.770% | 83.335% | 83.317% | 20 / 90 |
| 2,304–4,607 | 6.25% | 2,266/2,304 | 18.385% | 11.023% | 10.997% | 0 / 10 |
| 4,608–9,215 | 12.5% | 4,518/4,608 | 9.359% | 4.525% | 4.550% | 0 / 0 |
| 9,216–18,431 | 25% | 9,021/9,216 | 2.205% | 1.011% | 1.029% | 0 / 0 |
| 18,432–36,863 | 50% | 18,283/18,432 | 0.280% | 0.107% | 0.107% | 0 / 0 |

Cumulative prefixes:

| Prefix | Fire share | Base energy share | Target energy share | Top-100 |
|---:|---:|---:|---:|---:|
| 2,304 (6.25%) | 69.77% | 83.33% | 83.32% | 90 |
| 4,608 (12.5%) | 88.16% | 94.36% | 94.31% | 100 |
| 9,216 (25%) | 97.51% | 98.88% | 98.86% | 100 |
| 18,432 (50%) | 99.72% | 99.89% | 99.89% | 100 |

Each nonempty prefix enters the mean activation loss once. Therefore groups receive activation-loss
exposure weights proportional to `5:4:3:2:1`; the first group participates in every nonempty prefix
and the last group only in the full prefix. The measured concentration is stronger than feature
width alone and is consistent with this asymmetry.

Nearly all features firing at least once (98.65%) is therefore misleading as a capacity measure.
Effective utilization is only 1,141.13 components. All U write norms remain approximately one in
every group (group means 1.00003–1.00007), confirming that post-backward normalization worked and
that the concentration is not a collapsed U-norm artifact.

Existing evidence does not isolate whether `K=32`, rank-one form, or decoder normalization would
limit a computation-optimized gate. Changing any of them is outside this audit and is not
recommended from these measurements alone.

## H. Why parameter-only ΔY is extremely poor

Key scale comparison:

| Quantity | RMS |
|---|---:|
| Base true `o_proj` output | 0.239556 |
| Target true `o_proj` output | 0.240433 |
| True parameter-only delta | 0.00298118 |
| Predicted parameter-only delta | 0.0316238 |
| Parameter-only delta error | 0.0313363 |
| Base reconstruction error | 0.158130 |
| Target reconstruction error | 0.154583 |

The true parameter-only signal is only 1.24% of Base output RMS. The predicted delta is 10.61× the
true signal and has cosine 0.143, giving NSE 110.489. Individual reconstruction errors are roughly
52× the true parameter-delta RMS. The two full-model errors are strongly correlated (cosine 0.9625),
which allows substantial cancellation for total Base→Target differences, but the remaining
cross-bank mismatch is still much larger than the tiny parameter-only signal.

This is not an arithmetic-identity problem: the true decomposition identity has relative residual
`4.00e-06` and the ASPD identity `1.80e-08`. It is an objective-identifiability issue. Shared
component indices and a shared gate do not make separately trained model-specific rank-one factors
match closely enough to recover a small weight difference by subtraction.

## I. Open hypotheses and bounded next measurement

The existing artifacts establish that `R_mid` and `X_O` are captured from different sites and that
`o_proj` executes before the `R_mid` hook. They do not contain the raw validation activations needed
to quantify their population relationship or actual final-checkpoint gradient norms.

A bounded, read-only GPU diagnostic is prepared at
`slurm/multimodel/qwen25_aspd_original_100m_optimization_audit.sbatch`. It uses one held-out sequence,
the immutable rank-0 checkpoint, and five isolated backward probes. It writes only under
`out/multimodel/diagnostics/` and measures:

- final-checkpoint gradient norms for each objective/parameter group;
- BatchTopK support share by Matryoshka group on a real batch;
- RMS/cosine/difference between `R_mid` and `X_O`;
- actual versus forced-dead AuxK connectivity.

It was syntax-checked but **not submitted**, per task restriction. Researcher execution, if desired:

```bash
ALLOW_DIAGNOSTIC=1 sbatch \
  slurm/multimodel/qwen25_aspd_original_100m_optimization_audit.sbatch
```

That job can resolve the `R_mid`/`X_O` mismatch hypothesis and replace fixture gradient magnitudes
with checkpoint-specific magnitudes. It cannot by itself establish counterfactual benefits from a
different `K`, `C`, objective, or token budget.

## J. Reproducibility and claim ledger

The completed CPU audit command was:

```bash
.venv/bin/python -m aspd.multimodel.cli.audit_original_optimization \
  out/multimodel/runs/qwen25_aspd_original_100m \
  out/multimodel/diagnostics/qwen25_aspd_original_100m_optimization_audit_final
```

It produced `audit.json` and `training_objective_trajectory.png` outside the immutable run directory.

**Verified:** artifact integrity; objective equations; gradient connectivity; trajectory; final
held-out metrics; Matryoshka concentration; unit write norms; delta signal/error scales; upstream
hard-gate parity.

**Strong inference from verified measurements:** the residual-trained hard support and highly
concentrated prefix allocation constrain internal reconstruction; independent per-model objectives
cannot reliably recover a much smaller parameter delta by subtraction.

**Not established:** how much a computation-aware differentiable support, different Matryoshka
weights, larger `K`, larger `C`, another factorization, or more tokens would improve results. Those
would be new scientific methods/experiments and were neither run nor recommended by this audit.

# ASPD v3 mathematical specification audit

Date: 2026-11-07

Phase: **Phase 1 complete — mathematical review only**

Code basis inspected: `35711f9daa4629aea20c579f909db13501c56a52`

## Decision

The v4.1 Granite normalized-target notation did **not** faithfully represent the
researcher-approved multimodel ASPD design. It incorrectly elevated an implementation
choice—unit-normalizing every activation-decoder row—into Equation (8), then derived a
degenerate post-hoc relative decoder norm in Equation (30).

The authoritative correction is:

\[
d_{n,s,\ell,c}\in\mathbb R^{d_n}
\quad\text{with no imposed norm constraint},
\]

and the reconstruction remains

\[
\hat r_{n,s,\ell,a}^{(k)}
=b_{n,s,\ell}+\sum_{c\le k}g^{\mathrm{cont}}_{s,a,c}d_{n,s,\ell,c}.
\]

No norm penalty, learned scale, fixed scale, or new loss term is introduced. Decoder
norms remain measurable post hoc and may change through the approved reconstruction
objective.

## 1. Source hierarchy and limitations

### 1.1 Authoritative evidence

The specification decision uses the following sources, in precedence order:

1. **Current explicit researcher instruction, 2026-11-07.** It states that unit-norm
   activation decoders were not approved, forbids silently imposing
   `||D_c||_2=1`, requires magnitudes to remain trainable, and preserves read-only norm
   analysis.
2. **Earliest committed multimodel design contract, commit `d67fdef` (2026-10-06),
   `docs/multimodel/deviations.md`, item 5.** It explicitly says: “No decoder row
   normalization.” Its stated reason is that cross-model decoder norms are required for
   P1 and S2, while the common scaling gauge leaves relative norms invariant.
3. **Earliest multimodel equations, commit `d67fdef`,
   `docs/multimodel/method.md` and `aspd/multimodel/model.py`.** They define
   `r_hat = b + sum_c g_c d_c`, expose `||d_c||` for post-hoc/S2 analysis, and contain
   no activation-decoder projection or norm penalty.
4. **Initial fail-closed ASPD v3 infrastructure, commit `8671b24` (2026-10-30).** Its
   `ActivationReconstructionBank` is an ordinary collection of trainable linear
   decoders and has no normalization method. The architecture-decision document does
   not approve an activation-decoder norm constraint.

### 1.2 Non-authoritative or superseded descriptions

- Commit `7676bff` (2026-10-31) added activation-decoder unit normalization and labeled
  it “paper-derived” in the implementation report. This was an adaptation from the
  parent single-model implementation, not evidence in the original multimodel design
  contract.
- `docs/multimodel/aspd_v3_two_model_scale_gauge_audit.md` audited that implemented
  constraint and designed a treatment around it. It describes historical code; it does
  not override the original design or the current clarification.
- Granite Draft v4/v4.1, introduced at `55dfb33`, explicitly described itself as an
  implementation mapping and copied the existing unit-norm behavior into Equation (8).
  It therefore documented the discrepancy instead of establishing approval for it.
- Historical YAML files call their contents researcher-approved and include
  `activation_decoder_normalization: unit_row_norm_with_radial_gradient_projection`.
  The repository does not retain the plaintext authorization statement—only SHA-256
  `831723eacce597b04f4ee824ea223ec871f322798c9be2b768814ebb8448591f`—so that label
  cannot independently prove that this specific constraint was present in the approved
  prompt.

### 1.3 Missing original prompt

The verbatim original ASPD v3 researcher prompt is not versioned in this repository.
That is a provenance limitation. This audit does not reconstruct missing text from the
current code. It relies on the earliest committed mathematical contract, the initial v3
implementation boundary, and the researcher's present explicit clarification.

## 2. Mathematical discrepancy table

| Mathematical element | Original researcher specification | Current notation before correction | Discrepancy | Required correction |
|---|---|---|---|---|
| Activation decoder | Model/site/layer-specific free vectors `d`; magnitudes carry P1/S2 information (`d67fdef`, method + deviations item 5) | Described as a “direction” constrained to unit norm | Free trainable magnitude was removed | Define `d in R^d` without a norm constraint |
| Decoder norm | Read-only `||d||` and relative decoder mass; not a training constraint (`d67fdef`) | Eq. (8) enforces one; Eq. (30) makes all relative norms `1/3` | Measurement was confused with projection; useful post-hoc signal was destroyed | Remove projection/penalty; retain detached post-hoc norm measurement |
| Activation reconstruction | `b + sum g d` with continuous shared coefficients (`d67fdef`; current researcher instruction) | Same sum, but under unit-row constraint | Equation form matches; feasible parameter space does not | Preserve sum with unconstrained `d` |
| Target normalization | Treatment-only per-anchor FP32 L2 normalization; encoder input remains raw (researcher Draft v4 treatment, retained by current instruction) | Eqs. (15)–(18) | None identified | Preserve unchanged |
| Internal reconstruction | Detached binary shared support and model/target-specific rank-one `uv^T`; raw physical matrix outputs | Eqs. (10)–(12), (24), (29) | No activation-decoder discrepancy | Preserve; do not alter rank-one constraints in this correction |
| Matryoshka | Bias-only plus cumulative 512/1024/2048/4096/8192 prefixes, equal mean | Eqs. (14), (17), (18) | None identified | Preserve unchanged |
| AuxK | Dead-feature activation residual, coefficient `1/32`, same target space as main activation loss | Eqs. (22), (23), (26), (27) | None identified in the mathematical objective | Preserve unchanged |
| Final loss | Site/model/layer-balanced internal + activation + `1/32` auxiliary, averaged over four pools | Eqs. (24)–(28) | None identified | Preserve unchanged |
| Post-hoc relative norm | `rho_n = ||d_n|| / sum_m ||d_m||`, denominator positive (`d67fdef`) | Eq. (30) states `rho=1/3` | Historical consequence of the unapproved constraint | Restore nondegenerate relative-norm definition |
| Equation-to-code map | Intended mathematics is authoritative | Eq. (8) marks `normalize_decoders_` as `EXACT_MATCH` | Mapping treated code as specification | Mark current normalization path `MISMATCH` until corrected |

## 3. Corrected complete objective

### 3.1 Encoder and gates

For site `s` and aligned anchor `a`, the hierarchical encoder consumes the raw residual
activations from Gemma, SmolLM2, and Granite and produces nonnegative scores. BatchTopK
selects a pool-wide support:

\[
g^{\mathrm{cont}}_{s,a,c}=q_{s,a,c}\mathbf 1[(a,c)\in\Omega_s],
\qquad q=\operatorname{ReLU}(W_O h+b_O).
\]

Internal reconstruction uses detached binary support

\[
m_{s,a,c}=\mathbf1[g^{\mathrm{cont}}_{s,a,c}>0].
\]

There remains exactly one shared PRE code and one shared MID code.

### 3.2 Activation reconstruction

Activation decoder parameters are unconstrained:

\[
d_{n,s,\ell,c}\in\mathbb R^{d_n}.
\]

For raw targets,

\[
\hat r^{(k)}_{n,s,\ell,a}=b_{n,s,\ell}+
\sum_{c\le k}g^{\mathrm{cont}}_{s,a,c}d_{n,s,\ell,c}.
\]

For the normalized treatment,

\[
\nu_{n,s,\ell,a}=\max(\|r_{n,s,\ell,a}\|_2,10^{-6}),
\qquad \widetilde r=r/\nu,
\]

and the same decoder expression predicts `tilde r`. There is no learned or fixed
reconstruction scale and no decoder-norm term.

### 3.3 FVU and Matryoshka

\[
\operatorname{FVU}(X,\hat X)=
\frac{\sum_a\|X_a-\hat X_a\|_2^2}
{\sum_a\|X_a-\bar X\|_2^2}.
\]

For `Kset={0,512,1024,2048,4096,8192}`,

\[
L_{\mathrm{act},n,s,\ell}=\frac16\sum_{k\in Kset}
\operatorname{FVU}(T_{n,s,\ell},\hat T^{(k)}_{n,s,\ell}),
\]

where `T=r` for baseline and `T=tilde r` for treatment.

### 3.4 Internal and auxiliary reconstruction

\[
P_{j,c}^{(n)}=u_{j,c}^{(n)}(v_{j,c}^{(n)})^\top,
\qquad
\hat y_j^{(n)}=b_{j,\mathrm{native}}^{(n)}+
\sum_c m_{s(j),c}P_{j,c}^{(n)}x_j^{(n)}.
\]

The normalized-treatment AuxK residual is

\[
e^{\mathrm{norm}}=\widetilde r-
\operatorname{sg}(\hat{\widetilde r}^{(C)}),
\]

with the existing dead-feature selection and target-variance denominator. This audit
does not redesign its revival behavior.

### 3.5 Final objective

\[
L_{\mathrm{internal}}=\frac12\sum_s\frac1{|J_s|}
\sum_{(n,j)\in J_s}\operatorname{FVU}(Y_j^{(n)},\hat Y_j^{(n)}),
\]

\[
L_{\mathrm{activation}}=\frac12\sum_s\frac1{3\cdot8}
\sum_{n,\ell}L_{\mathrm{act},n,s,\ell},
\]

\[
L_{\mathrm{auxiliary}}=\frac12\sum_s\frac1{3\cdot8}
\sum_{n,\ell}L_{\mathrm{aux},n,s,\ell},
\]

\[
\boxed{\mathcal L=L_{\mathrm{internal}}+L_{\mathrm{activation}}+
\frac1{32}L_{\mathrm{auxiliary}}.}
\]

The four pool objectives are averaged per optimizer step. This objective is unchanged by
the decoder-norm correction.

## 4. Measurement versus enforcement

The corrected method distinguishes three operations:

1. **Measurement:** `q_{n,s,l,c}=||d_{n,s,l,c}||_2` may be logged or analyzed without
   altering parameters or gradients.
2. **Enforcement:** dividing decoder rows by their norms during initialization or after
   `optimizer.step()` changes the feasible optimization trajectory and is not approved.
3. **Regularization:** adding a function of decoder norms to the loss changes the
   objective and is not approved.

Only operation 1 is part of this corrected experiment family.

## 5. Corrected notation status

`docs/multimodel/aspd_v3_granite_normalized_target_notation.md` is updated to Draft
v4.2:

- Equation (8) now declares free activation-decoder vectors.
- Equation (30) restores nondegenerate post-hoc relative decoder mass.
- The equation-to-code map marks the historical projection as `MISMATCH` until Phase 3.
- The reverse map no longer represents radial projection/post-step normalization as an
  approved equation.

No other equation or objective coefficient was changed.

## 6. Phase 1 conclusion

**Does current mathematical notation faithfully represent the original approved ASPD v3
design?** No. Draft v4.1 incorrectly imposed activation-decoder unit norms and derived an
invalid degenerate post-hoc norm metric for the intended method.

**Are other objective discrepancies established by the available mathematical sources?**
No. Encoder/gating, normalized targets, FVU, Matryoshka, internal reconstruction, AuxK,
and the final loss are retained exactly. Rank-one factor normalization is a separate
parameterization and is not changed by this activation-decoder correction.

Phase 1 is complete. Phase 2 may now audit every implementation and checkpoint path
against this corrected specification.

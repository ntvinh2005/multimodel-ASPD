# Multimodel ASPD v3 — Revised Mathematical Notation

Draft v4.1 — researcher-review and code-mapping specification

Scope: Gemma-2-2B-IT + SmolLM2-1.7B-Instruct + Granite-3.3-2B-Instruct

Treatment: per-anchor L2 normalization of original activation targets

Status: implementation mapping, not an efficacy claim

Interpretation keys: **baseline** means raw-target ASPD v3; **treatment** means the
per-anchor normalized-target objective; **oracle** means evaluation uses the true target
norm; **post-hoc** means analysis-only and not training code.  Source locations below refer
to branch `aspd-v3-granite-normalized-target`.  Line numbers are review aids; function names
are the stable identifiers.

## 1. Setup, alignment, and targets

The frozen model order is

1. `google/gemma-2-2b-it`, layers 9–16;
2. `HuggingFaceTB/SmolLM2-1.7B-Instruct`, layers 8–15;
3. `ibm-granite/granite-3.3-2b-instruct`, layers 16–23.

Canonical anchors are shared text-prefix boundaries.  Native token positions may differ:

\[
\tau_n(a)=\text{native token position for canonical anchor }a\text{ in model }n.
\]

There are eight layers and seven logical projections per model.  For a physical projection:

\[
\boxed{y_{j,a}^{(n)}=W_j^{(n)}x_{j,a}^{(n)}+b_{j,\mathrm{native}}^{(n)}} \tag{1}
\]

Thus \(|\mathcal J|=3\cdot8\cdot7=168\).  PRE grounds Q/K/V; MID grounds O and
MLP gate/up/down.  Granite's residual multiplier is applied after attention or MLP output is
formed; it is not part of Eq. (1).

For one site and model, eight residuals are concatenated:

\[
\boxed{R_{n,s,a}=\operatorname{Concat}_{\ell\in\mathcal L^{(n)}}r_{n,s,\ell,a}}. \tag{2}
\]

## 2. Hierarchical shared encoder

\[
\boxed{P_{n,s,a}=\operatorname{Reshape}_{8\times512}
 (W_{P,n,s}R_{n,s,a}+b_{P,n,s})} \tag{3}
\]

\[
\boxed{H_{n,s,a}=\operatorname{IntraTransformer}_{n,s}(P_{n,s,a})}. \tag{4}
\]

The three 8-token sequences and two separator tokens form a sequence of length 26.  Five
identity channels make the adapter input width 517:

\[
\boxed{\widetilde H_{s,a}=\operatorname{CrossTransformer}_s(
 \operatorname{PackModels}(H_{1,s,a},H_{2,s,a},H_{3,s,a}))}. \tag{5}
\]

With \(q=\operatorname{ReLU}(W_O\operatorname{MeanPool}(\widetilde H)+b_O)\),
BatchTopK selects \(K|\mathcal B|\) entries over each pool:

\[
\boxed{g^{\mathrm{cont}}_{s,a,c}=q_{s,a,c}\mathbf1[(a,c)\in\Omega_s]},
\qquad C=8192,\;K=32. \tag{6}
\]

Internal reconstruction receives only detached binary support:

\[
\boxed{m_{s,a,c}=\mathbf1[g^{\mathrm{cont}}_{s,a,c}>0]}. \tag{7}
\]

There is exactly one PRE gate and one MID gate, each shared across all three models.  Encoder
inputs remain raw activations in both experimental arms.

## 3. Activation and rank-one reconstruction

For every model/site/layer/feature, the activation decoder direction is normalized
independently:

\[
\boxed{\|d_{n,s,\ell,c}\|_2=1}. \tag{8}
\]

The raw baseline prediction is

\[
\boxed{\hat r_{n,s,\ell,a}=b_{n,s,\ell}+
 \sum_{c=1}^{C}g^{\mathrm{cont}}_{s,a,c}d_{n,s,\ell,c}}. \tag{9}
\]

Rank-one operators and input-conditioned contributions are

\[
\boxed{P_{j,c}^{(n)}=u_{j,c}^{(n)}(v_{j,c}^{(n)})^\top}, \tag{10}
\]

\[
\boxed{e_{j,a,c}^{(n)}=m_{s(j),a,c}u_{j,c}^{(n)}
 ((v_{j,c}^{(n)})^\top x_{j,a}^{(n)})}, \tag{11}
\]

\[
\boxed{\hat y_{j,a}^{(n)}=b_{j,\mathrm{native}}^{(n)}+
 \sum_{c=1}^{C}e_{j,a,c}^{(n)}}. \tag{12}
\]

## 4. Baseline and normalized-target objectives

Accepted-anchor FVU is coordinate-centered and reduced in FP32:

\[
\boxed{\operatorname{FVU}_{\mathcal B}(X,\hat X)=
 \frac{\sum_{a\in\mathcal B}\|X_a-\hat X_a\|_2^2}
 {\sum_{a\in\mathcal B}\|X_a-\bar X_{\mathcal B}\|_2^2}}. \tag{13}
\]

For \(\mathcal K=(0,512,1024,2048,4096,8192)\), baseline Matryoshka loss is

\[
\boxed{L_{\mathrm{act},n,s,\ell}^{\mathrm{raw}}=
 \frac16\sum_{k\in\mathcal K}\operatorname{FVU}
 (r_{n,s,\ell},\hat r_{n,s,\ell}^{(k)})}. \tag{14}
\]

The treatment computes, in FP32 and separately per valid model/site/layer/anchor,

\[
\boxed{\nu_{n,s,\ell,a}=\max(\|r_{n,s,\ell,a}\|_2,\epsilon)},
\qquad\epsilon=10^{-6}, \tag{15}
\]

\[
\boxed{\widetilde r_{n,s,\ell,a}=r_{n,s,\ell,a}/\nu_{n,s,\ell,a}}. \tag{16}
\]

The coefficient is detached.  Invalid anchors are zero-filled for transport and excluded by
the loss mask.  For every prefix, the unchanged decoder expression predicts normalized space:

\[
\boxed{\hat{\widetilde r}_{n,s,\ell,a}^{(k)}=b_{n,s,\ell}+
 \sum_{c\le k}g^{\mathrm{cont}}_{s,a,c}d_{n,s,\ell,c}}. \tag{17}
\]

\[
\boxed{L_{\mathrm{act},n,s,\ell}^{\mathrm{norm}}=
 \frac16\sum_{k\in\mathcal K}\operatorname{FVU}
 (\widetilde r_{n,s,\ell},\hat{\widetilde r}_{n,s,\ell}^{(k)})}. \tag{18}
\]

Full-prefix restoration is explicitly oracle-norm:

\[
\boxed{\hat r_{n,s,\ell,a}^{\mathrm{oracle}}=
 \nu_{n,s,\ell,a}\hat{\widetilde r}_{n,s,\ell,a}^{(C)}}. \tag{19}
\]

\[
\boxed{\operatorname{FVU}_{\mathrm{original}}=
 \operatorname{FVU}(r,\nu\hat{\widetilde r}^{(C)})}, \tag{20}
\]

\[
\boxed{\operatorname{FVU}_{\mathrm{norm}}=
 \operatorname{FVU}(\widetilde r,\hat{\widetilde r}^{(C)})}. \tag{21}
\]

Equation (19) does not predict magnitude independently.

## 5. AuxK, internal loss, and reduction

The treatment residual is

\[
\boxed{e^{\mathrm{norm}}=\widetilde r-
 \operatorname{sg}(\hat{\widetilde r}^{(C)})}. \tag{22}
\]

With dead-feature auxiliary reconstruction \(\hat e^{\mathrm{aux}}\),

\[
\boxed{L_{\mathrm{aux}}=
 \frac{\sum_a\|e_a^{\mathrm{norm}}-\hat e_a^{\mathrm{aux}}\|_2^2}
 {\sum_a\|\widetilde r_a-\overline{\widetilde r}\|_2^2}}. \tag{23}
\]

The internal objective remains raw and site-balanced:

\[
\boxed{L_{\mathrm{internal}}=\frac12\sum_s\frac1{|\mathcal J_s|}
 \sum_{(n,j)\in\mathcal J_s}\operatorname{FVU}(Y_j^{(n)},\hat Y_j^{(n)})}, \tag{24}
\]

where \(|\mathcal J_{\mathrm{PRE}}|=72\) and \(|\mathcal J_{\mathrm{MID}}|=96\).

\[
\boxed{L_{\mathrm{activation}}=\frac12\sum_s\frac1{3\cdot8}
 \sum_{n,\ell}L_{\mathrm{act},n,s,\ell}^{\mathrm{norm}}}, \tag{25}
\]

\[
\boxed{L_{\mathrm{auxiliary}}=\frac12\sum_s\frac1{3\cdot8}
 \sum_{n,\ell}L_{\mathrm{aux},n,s,\ell}}, \tag{26}
\]

\[
\boxed{\mathcal L=L_{\mathrm{internal}}+L_{\mathrm{activation}}+
 \tfrac1{32}L_{\mathrm{auxiliary}}}, \tag{27}
\]

\[
\boxed{\mathcal L_{\mathrm{step}}=\frac14\sum_{p=1}^{4}\mathcal L(\mathcal B_p)}. \tag{28}
\]

Model-owned ranks sum-reduce shared-encoder gradients.  The detached support guarantees

\[
\boxed{\partial L_{\mathrm{internal}}/\partial\theta_{\mathrm{encoder}}=0}. \tag{29}
\]

## 6. Gauge and proposed post-hoc analysis

Unit activation-decoder norms make the legacy relative decoder norm degenerate:

\[
\boxed{\rho_{n,s,\ell,c}^{D}=1/N=1/3}. \tag{30}
\]

The following definitions remain post-hoc proposals and are not executed by training:

\[
\boxed{\Delta_{n,s,\ell,c}^{\mathrm{act}}=
 \operatorname{FVU}(\widetilde r,\hat{\widetilde r}^{(-c)})-
 \operatorname{FVU}(\widetilde r,\hat{\widetilde r}^{(C)})}, \tag{31}
\]

\[
\boxed{\beta_{j,c}^{(n)}=
 \frac{\sqrt{\mathbb E[m_c\|P_{j,c}^{(n)}x_j^{(n)}\|_2^2]/\mathbb E[m_c]}}
 {\sigma_j^{(n)}}}, \tag{32}
\]

\[
\boxed{\beta_c^{(n)}=\sqrt{\sum_{j\in\mathcal J^{(n)}}(\beta_{j,c}^{(n)})^2}}, \tag{33}
\]

\[
\boxed{\rho_c^{\mathrm{mech},(n)}=\beta_c^{(n)}/\sum_m\beta_c^{(m)}}, \tag{34}
\]

\[
\boxed{\pi_c^{(n)}(j)=\beta_{j,c}^{(n)}/\sum_{j'}\beta_{j',c}^{(n)}}, \tag{35}
\]

\[
\boxed{y_{j,a}^{(n),\mathrm{ablated}}=y_{j,a}^{(n)}-
 \sum_{c\in S}m_{s(j),a,c}P_{j,c}^{(n)}x_{j,a}^{(n)}}. \tag{36}
\]

These quantities require held-out support thresholds and, for causal claims, a separately
approved online intervention protocol.

## 7. Parameter complexity

\[
\boxed{\#\theta_{\mathrm{act\ decoder}}=2\cdot8\sum_{n=1}^{3}(Cd_n+d_n)}, \tag{37}
\]

\[
\boxed{\#\theta_{\mathrm{rank1}}=C\sum_n\sum_{j\in\mathcal J^{(n)}}
 (d_{\mathrm{in},j}^{(n)}+d_{\mathrm{out},j}^{(n)})}, \tag{38}
\]

\[
\boxed{\#\theta_{\mathrm{total}}=\#\theta_{\mathrm{encoder}}+
 \#\theta_{\mathrm{act\ decoder}}+\#\theta_{\mathrm{rank1}}}. \tag{39}
\]

Per-anchor normalization adds no trainable parameters.

## 8. Equation → code map

Status vocabulary is exact: `EXACT_MATCH`, `SEMANTICALLY_EQUIVALENT`, `PARTIAL_MATCH`,
`MISMATCH`, `NOT_IMPLEMENTED`, `THEORETICAL_ONLY`, `UNVERIFIED`.

| Eq. | Implementation file/function/lines | Configuration | Numerical evidence | Status |
|---:|---|---|---|---|
| 1 | `v3/hooks.py:LogicalMatrixCapture` 42–104; `logical_weight_bias` 173–197 | registry bias policy | existing hook parity tests; full Granite checkpoint not run | `SEMANTICALLY_EQUIVALENT` |
| 2 | `v3/encoder.py:concatenate_layer_activations` 16–24 | eight selected layers | packing/shape tests | `EXACT_MATCH` |
| 3 | `v3/model_owned.py:LocalSiteEncoder` 80–98 | learned packer, 8×512 | 26×517 packing test | `SEMANTICALLY_EQUIVALENT` |
| 4 | `v3/model_owned.py:LocalSiteEncoder.forward` 97–98 | two intra layers | shared-gate CPU test | `EXACT_MATCH` |
| 5 | `v3/encoder.py:pack_cross_model_tokens` 121–186; `SharedSiteEncoder.forward` 132–134 | 517→512 adapter | exact shape/identity test | `SEMANTICALLY_EQUIVALENT` |
| 6 | `v3/gating.py:V3BatchTopK.forward` 90–106 | C=8192, K=32, pool=2048 | continuous shared-gate test | `EXACT_MATCH` |
| 7 | `v3/model_owned.py:encode` 303–308; `objective.detached_binary_support` | binary internal code | detached-gradient test | `EXACT_MATCH` |
| 8 | `v3/reconstruction.py:ActivationReconstructionBank.normalize_decoders_` 216–242 | individual unit row norm | post-update unit-norm test | `EXACT_MATCH` |
| 9 | `v3/objective.py:matryoshka_activation_fvu` 67–143 | baseline normalization=`none` | bitwise raw-path test | `EXACT_MATCH` |
| 10 | `components.py:RankOneComponents` | unit-norm write factors | rank-one gradient tests | `EXACT_MATCH` |
| 11 | `components.py:RankOneComponents.forward_sparse`; `reconstruction.py:RankOneMatrixBank.reconstruct` | binary support | detached internal test | `SEMANTICALLY_EQUIVALENT` |
| 12 | `v3/reconstruction.py:RankOneMatrixBank.reconstruct` 284–356 | native bias preserve | existing native-bias tests | `SEMANTICALLY_EQUIVALENT` |
| 13 | `v3/objective.py:masked_fvu` 26–51 | FP32 reductions | objective tests | `EXACT_MATCH` |
| 14 | `v3/objective.py:matryoshka_activation_fvu` 67–143 | raw baseline | strict baseline parity test | `EXACT_MATCH` |
| 15 | `v3/activation_targets.py:prepare_activation_target` 55–60 | epsilon=1e-6 | [3,4], zero, near-zero tests | `EXACT_MATCH` |
| 16 | same as Eq. 15, lines 57–59 | `per_anchor_l2` | unit-target tests | `EXACT_MATCH` |
| 17 | `v3/model_owned.py:activation_loss_terms` 329–358 + objective decoder | normalized target only; no learned scale | Matryoshka test | `SEMANTICALLY_EQUIVALENT` |
| 18 | same path plus `matryoshka_activation_fvu` | normalized treatment | loss/gradient test | `EXACT_MATCH` |
| 19 | `v3/activation_targets.py:oracle_denormalize` 66–76 | oracle evaluation true | inverse test | `EXACT_MATCH` |
| 20 | `v3/activation_evaluation.py:activation_reconstruction_metrics` 25–41 | explicitly oracle-qualified | metric-space test | `EXACT_MATCH` |
| 21 | same function, lines 33–41 | normalized metric | metric-space test | `EXACT_MATCH` |
| 22 | `v3/objective.py:auxk_dead_feature_loss` 171–177 | normalized target routed once | AuxK test | `EXACT_MATCH` |
| 23 | same function 171–219 | target-variance denominator | AuxK finite-gradient test | `EXACT_MATCH` |
| 24 | `v3/model_owned.py:internal_loss_terms` 366–397 | PRE/MID counts 72/96 | existing model-owned tests | `SEMANTICALLY_EQUIVALENT` |
| 25 | `v3/model_owned.py:activation_loss_terms` coefficient line 320 | N=3, 8 layers, two sites | config/gradient tests | `EXACT_MATCH` |
| 26 | same coefficient and yielded AuxK terms | λ applied separately | AuxK test | `EXACT_MATCH` |
| 27 | `cli/train_aspd_v3_2b.py:_model_owned_backward` | λ=1/32 | existing trainer tests | `SEMANTICALLY_EQUIVALENT` |
| 28 | trainer lines 1211–1284, `pool_scale=1/4` | four pools | source audit only; GPU not run | `EXACT_MATCH` |
| 29 | binary support detach lines 306–307 | rank-one-only internal route | detached-gradient test | `EXACT_MATCH` |
| 30 | mathematical consequence of Eq. 8 | N=3 | analytic identity | `THEORETICAL_ONLY` |
| 31 | no executed analysis utility | none | not run | `NOT_IMPLEMENTED` |
| 32 | no executed analysis utility | none | not run | `NOT_IMPLEMENTED` |
| 33 | no executed analysis utility | none | not run | `NOT_IMPLEMENTED` |
| 34 | no executed analysis utility | none | not run | `NOT_IMPLEMENTED` |
| 35 | no executed analysis utility | none | not run | `NOT_IMPLEMENTED` |
| 36 | no approved online intervention | none | not run | `NOT_IMPLEMENTED` |
| 37 | `v3/model_owned.py:model_owned_parameter_counts` | roster widths/C | configuration arithmetic | `SEMANTICALLY_EQUIVALENT` |
| 38 | same function | 168 target shapes/C | registry arithmetic | `SEMANTICALLY_EQUIVALENT` |
| 39 | same function plus `named_parameters` checks in optimizer construction | model-owned topology | no full-size allocation | `PARTIAL_MATCH` |

Equation (1) remains `SEMANTICALLY_EQUIVALENT` rather than `EXACT_MATCH` until a pinned
full-checkpoint Granite forward validates all 56 hooks and native outputs.  Equation (39) is
partial because a full three-model parameter allocation was intentionally forbidden.

## 9. Reverse code → equation map

| Executed operation | Equation(s) |
|---|---|
| Exact tokenizer prefix-boundary intersection and native gather positions | alignment definition preceding Eq. (1) |
| Physical module input/output capture and native-bias reconstruction | 1, 12 |
| Layer concatenation, learned packing, intra transformer | 2–4 |
| Model/separator packing, cross transformer, readout | 5–6 |
| BatchTopK continuous values and detached support preparation | 6–7, 29 |
| Activation decoder initialization, radial gradient projection, post-step normalization | 8 |
| Dense/indexed/CSR cumulative activation decode | 9, 14, 17–18 |
| Per-target preparation once per model/site/layer | 15–16 |
| Normalized Matryoshka and normalized AuxK routing | 17–18, 22–23 |
| Rank-one sparse reconstruction and native bias addition | 10–12, 24 |
| Site/model/layer loss coefficients and four-pool accumulation | 24–28 |
| Oracle denormalization and dual-space metric report | 19–21 |
| Analytical parameter-count helper | 37–39 |
| Checkpoint config fingerprint and explicit experiment metadata | provenance for 1–29, not a separate equation |

## 10. Outstanding researcher decisions

- Granite 16–23 is a provisional mid-depth window (normalized layer positions 0.410–0.590
  using \(\ell/(L-1)\)); no layer search was run.
- Real three-tokenizer acceptance rate is unmeasured.
- Full pinned-checkpoint module and forward parity are unverified.
- Normalized FVU changes sample weighting and gradient scale; no coefficient was retuned.
- Original-space reconstruction is oracle-norm and is not a standalone magnitude predictor.
- Equations (31)–(36) are not implementation or efficacy claims.
- No training, GPU preflight, or full experiment is authorized by this document.

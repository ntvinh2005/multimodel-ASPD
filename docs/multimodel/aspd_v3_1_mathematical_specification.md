# ASPD v3.1 canonical mathematical specification

Status: **provisional scientific specification for the fail-closed 500M draft**  
Execution status: **`MAIN_500M_NOT_AUTHORIZED`**  
Scope: three-model ASPD with the hierarchical Transformer encoder and the validated
per-anchor feature-axis z-score intervention.

This file is the canonical notation for ASPD v3.1. It supersedes conflicting gate and
objective text in `aspd_v3_architecture_decisions.md` and
`aspd_v3_mathematical_specification_audit.md`: in particular, the current gate includes
z-score and the current activation objective is full-only rather than Matryoshka. The
mechanistic-validation reports remain the evidence for adopting the intervention; they
are not alternative definitions of the current gate.

## 1. Indices and fixed sets

Let

\[
m\in\{1,2,3\}
\]

index the three source models, and let

\[
\mathcal L_m=\{\ell_{m,1},\ldots,\ell_{m,8}\}
\]

be the eight selected layers for model \(m\). The logical matrix set is

\[
\mathcal J=
\{q_{\mathrm{proj}},k_{\mathrm{proj}},v_{\mathrm{proj}},o_{\mathrm{proj}},
  \mathrm{gate}_{\mathrm{proj}},\mathrm{up}_{\mathrm{proj}},
  \mathrm{down}_{\mathrm{proj}}\}.
\]

Therefore the main experiment has

\[
3\times 8\times 7=168
\]

logical matrix targets. A fused physical module may implement more than one logical
target, but it does not change this count.

Let \(a\in\mathcal A\) index accepted aligned anchors, \(c\in\{1,\ldots,C\}\)
index sparse features, and \(t\in\{\mathrm{PRE},\mathrm{MID}\}\) index the two
grounding sites. The provisional reference keeps

\[
C=8192,\qquad K=32.
\]

PRE grounds \(q_{\mathrm{proj}},k_{\mathrm{proj}},v_{\mathrm{proj}}\). MID
grounds \(o_{\mathrm{proj}},\mathrm{gate}_{\mathrm{proj}},
\mathrm{up}_{\mathrm{proj}},\mathrm{down}_{\mathrm{proj}}\).

## 2. Aligned activation encoder

For model \(m\), site \(t\), anchor \(a\), and selected layer \(\ell\), let

\[
h^{(m)}_{a,t,\ell}\in\mathbb R^{d_m}
\]

be the untouched frozen-backbone activation supplied to the encoder. The eight layer
vectors are concatenated and passed through a model-specific learned linear packer:

\[
X^{(m)}_{a,t}
=
\operatorname{reshape}_{8\times 512}
\left(
\Pi^{(m)}_t
\left[
h^{(m)}_{a,t,\ell_{m,1}};\ldots;h^{(m)}_{a,t,\ell_{m,8}}
\right]
\right).
\]

Two model-specific intra-model Transformer layers produce

\[
H^{(m)}_{a,t}=T^{(m)}_{t,\mathrm{intra}}(X^{(m)}_{a,t}).
\]

The three eight-token sequences are placed in deterministic model order, with two
parameter-free separator tokens and parameter-free one-hot identity channels. Hence
the cross-model sequence length is

\[
N\cdot 8+(N-1)=3\cdot 8+2=26.
\]

After the learned \(517\rightarrow512\) adapter and two cross-model Transformer
layers, mean pooling and a shared linear readout give the pre-normalization logits

\[
z_{a,t,c}
=
\left[
W^{\mathrm{read}}_t
\operatorname{MeanPool}
\left(
T_{t,\mathrm{cross}}(H^{(1)}_{a,t},H^{(2)}_{a,t},H^{(3)}_{a,t})
\right)
+b^{\mathrm{read}}_t
\right]_c.
\]

PRE and MID have separate encoders and gates. Within a site, component identity \(c\)
is shared across all three models and all eight selected layers.

## 3. Canonical z-score gate

For every accepted anchor and site, normalize over the full feature axis. Define

\[
\mu_{a,t}=\frac{1}{C}\sum_{c=1}^{C}z_{a,t,c},
\]

\[
\sigma^2_{a,t}
=
\frac{1}{C}\sum_{c=1}^{C}
\left(z_{a,t,c}-\mu_{a,t}\right)^2,
\]

and

\[
\boxed{
\tilde z_{a,t,c}
=
\frac{z_{a,t,c}-\mu_{a,t}}
{\sqrt{\sigma^2_{a,t}+\epsilon_z}}
},
\qquad
\epsilon_z=10^{-5}.
\]

This is population variance with denominator \(C\), not the unbiased estimator with
denominator \(C-1\). The normalization has no learned affine scale or bias. The feature
axis is normalized independently for each anchor and separately for PRE and MID.

The non-negative candidates are

\[
s_{a,t,c}=\operatorname{ReLU}(\tilde z_{a,t,c}).
\]

No feature is forced active. For each BatchTopK pool \(\mathcal P\subset\mathcal A\),
let \(\mathcal I_t\) contain the \(K|\mathcal P|\) largest entries of
\(\{s_{a,t,c}:a\in\mathcal P,1\le c\le C\}\). The continuous sparse code is

\[
g^s_{a,t,c}
=
\begin{cases}
s_{a,t,c}, & (a,c)\in\mathcal I_t,\\
0, & \text{otherwise}.
\end{cases}
\]

This is pool-wide BatchTopK. It is not independent TopK per anchor. If fewer than
\(K|\mathcal P|\) selected entries are strictly positive, the corresponding zero-valued
positions do not become firing events because the binary gate below tests \(g^s>0\).

## 4. Continuous activation code and detached binary internal support

Activation reconstruction uses the continuous sparse coefficients \(g^s\). Internal
matrix reconstruction uses

\[
\boxed{
g_{a,t,c}=\mathbf 1[g^s_{a,t,c}>0]
}
\]

with the support detached from the shared encoder before internal reconstruction.
Consequently, the internal loss trains model-specific rank-one operators but supplies no
gradient to the shared encoder. This distinction is part of the causal definition of an
ASPD component and must not be removed by an implementation optimization.

## 5. Activation reconstruction

The activation target is normalized per anchor along its hidden dimension:

\[
\bar h^{(m)}_{a,t,\ell}
=
\frac{h^{(m)}_{a,t,\ell}}
{\max(\lVert h^{(m)}_{a,t,\ell}\rVert_2,\epsilon_h)},
\qquad
\epsilon_h=10^{-6}.
\]

Encoder inputs remain the untouched \(h^{(m)}\); only the reconstruction target is
normalized. With model-, site-, and layer-specific decoder \(D^{(m)}_{t,\ell}\) and
bias \(b^{(m)}_{t,\ell}\),

\[
\hat h^{(m)}_{a,t,\ell}
=
b^{(m)}_{t,\ell}
+D^{(m)}_{t,\ell}g^s_{a,t}.
\]

The activation-decoder columns are unconstrained in v3.1. There is no unit-norm
activation-decoder projection and no learned reconstruction scale. The primary
activation objective is full-dictionary FVU only; the historical Matryoshka objective is
not part of this specification.

## 6. Model-specific rank-one parameter operators

For model \(m\), selected layer \(\ell\), logical matrix \(j\), and component \(c\),
the parameter operator is

\[
\boxed{
P_{\ell,j,c}^{(m)}
=
u_{\ell,j,c}^{(m)}
\left(v_{\ell,j,c}^{(m)}\right)^\top
}.
\]

Let \(x^{(m)}_{a,\ell,j}\) and \(y^{(m)}_{a,\ell,j}\) be the captured native input
and output of logical target \((m,\ell,j)\), and let \(t(j)\) be its PRE/MID grounding
site. The internal reconstruction is

\[
\hat y^{(m)}_{a,\ell,j}
=
b^{(m)}_{\ell,j}
+\sum_{c=1}^{C}
g_{a,t(j),c}\,
u^{(m)}_{\ell,j,c}
\left(v^{(m)}_{\ell,j,c}\right)^\top
x^{(m)}_{a,\ell,j}.
\]

Native bias is preserved when the source module has one. The write direction \(u\) is
unit-normalized after optimization as in the rank-one ASPD parameterization; this is
separate from the unconstrained activation decoder.

## 7. FVU and balanced objective

For a target tensor \(Y\) flattened over accepted anchors and centered independently in
each output coordinate, define

\[
\operatorname{FVU}(\hat Y,Y)
=
\frac{\sum_{a,d}(\hat Y_{a,d}-Y_{a,d})^2}
{\sum_{a,d}(Y_{a,d}-\bar Y_d)^2}.
\]

Let \(\mathcal L^{\mathrm{act}}_t\) be the mean full-only activation FVU over the
three models and eight selected layers at site \(t\). Let
\(\mathcal L^{\mathrm{int}}_{\mathrm{PRE}}\) be the mean internal FVU over the 72
PRE targets and \(\mathcal L^{\mathrm{int}}_{\mathrm{MID}}\) the mean over the 96
MID targets. Then

\[
\mathcal L_{\mathrm{act}}
=
\frac12\mathcal L^{\mathrm{act}}_{\mathrm{PRE}}
+\frac12\mathcal L^{\mathrm{act}}_{\mathrm{MID}},
\]

\[
\mathcal L_{\mathrm{int}}
=
\frac12\mathcal L^{\mathrm{int}}_{\mathrm{PRE}}
+\frac12\mathcal L^{\mathrm{int}}_{\mathrm{MID}}.
\]

Historical post-ReLU AuxK is computed from \(s\), restricted to features dead for 320
consecutive pools, with \(K_{\mathrm{aux}}=2048\). It reconstructs the detached residual
of the full activation reconstruction. With \(\lambda_{\mathrm{aux}}=0.03125\),

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{int}}
+\mathcal L_{\mathrm{act}}
+\lambda_{\mathrm{aux}}\mathcal L_{\mathrm{aux}}
}.
\]

## 8. Gradient routes

- \(\mathcal L_{\mathrm{act}}+\lambda_{\mathrm{aux}}\mathcal L_{\mathrm{aux}}\)
  trains model-specific packers/intra-model Transformers, shared cross-model Transformers
  and readouts, and model-specific activation decoders.
- The shared encoder gradient is the sum of the three model-owned activation-objective
  gradients after differentiable latent exchange.
- \(\mathcal L_{\mathrm{int}}\) trains only the model-specific rank-one matrix operators.
  The detached support forbids an internal-loss gradient into the shared encoder.
- Frozen source-model weights receive no gradient and are not optimizer parameters.

## 9. Exposure accounting

An accepted exposure is one canonical aligned anchor consumed by one optimizer step. It
is not a unique raw token count and is not multiplied by the number of models. With
8192 accepted anchors per step,

\[
\left\lceil\frac{500{,}000{,}000}{8192}\right\rceil=61{,}036,
\]

\[
61{,}036\times8192=500{,}006{,}912.
\]

The final step therefore exceeds the requested minimum by 6,912 accepted exposures.
Every accepted anchor ID must be unique across the run unless a future protocol
explicitly changes this rule; the current protocol forbids repeated exposures.

## 10. Notation-to-code map

| Mathematical object | Implementation |
|---|---|
| \(z_{a,t,c}\) | `FeatureReadout.latest_projection_logits` |
| \(\tilde z_{a,t,c}\) | `FeatureReadout.forward_with_logits(...)[1]` |
| \(s_{a,t,c}\) | `FeatureReadout.forward_with_logits(...)[0]` |
| \(g^s_{a,t,c}\) | `SparseCode.values` / continuous `SparseSupport` |
| \(g_{a,t,c}\) | detached binary `SparseSupport` / `SparseCode.gate` |
| \(P_{\ell,j,c}^{(m)}\) | `RankOneMatrixBank` / `RankOneComponents` |
| activation reconstruction | `ActivationReconstructionBank` and `full_activation_fvu` |
| internal reconstruction | `RankOneMatrixBank.reconstruct(..., code_semantics="binary_support")` |
| z-score axis and \(\epsilon_z\) | `FeatureReadout` feature-axis `layer_norm`, `eps=1e-5` |
| pool-wide \(\sigma_K\) | `batch_topk` with `pool_tokens=2048`, `top_k=32` |

The compatibility field `latest_raw_logits` currently denotes the logits actually given
to ReLU, i.e. \(\tilde z\) under v3.1. New diagnostics use the unambiguous names
`latest_pre_normalization_logits` and `latest_normalized_logits`.

# Semantic supernodes in the ASPD shared reference space

This mode pools the native ASPD shared encoder before the learned gate projection and constructs one complete-linkage partition across representative components of Gemma, SmolLM2, and Granite. The old MiniLM and per-model average-linkage catalog remains independently available. The two distance scales and clusters are separate.

## Formulation

For each representative component \(a=(j,i)\), measured contribution strength is \(\kappa_{a,\alpha}^{(n)}=\|\Delta y_{j,\alpha,i}^{(n)}\|_2=|e_{j,\alpha,i}^{(n)}|\|u_{j,i}^{(n)}\|_2\). Within each document split, the support contains the top \(K_{\rm ctx}\) positive measured anchor contributions, or all available positive anchors when fewer exist. Ties use canonical anchor ID. The support can include multiple anchors in the same document. Document counts are displayed separately. When no discovery anchor has positive contribution, the component is excluded. Missing held-out support produces an unavailable comparison.

At each anchor the trained ASPD encoder supplies distinct PRE and MID streams. The input to each trained gate projection already equals the mean of the cross-model encoder output tokens along their token axis. The export captures this input from both stages, then concatenates them in PRE, MID order:

\[
\bar z_{p,\alpha}=\frac1L\sum_{k=1}^L Z_{p,\alpha,k}^{\rm shared},\qquad
z_\alpha=[\bar z_{\rm PRE,\alpha};\bar z_{\rm MID,\alpha}].
\]

The checkpoint uses width 512 per site, yielding 1,024 coordinates. Export preserves the trained forward pass and the existing pool-wide natural BatchTopK gates that determine the measured coefficients. Cosine dissimilarity uses L2 normalization of the concatenated row. The reference vectors precede the projection, feature-axis normalization, ReLU, and BatchTopK.

For components \(a^{(n)}\) and \(b^{(m)}\), exact balanced transport minimizes

\[
D_{\rm OT}(a^{(n)},b^{(m)}) =
\min_{\Gamma\ge0}\sum_{\alpha,\beta}\Gamma_{\alpha,\beta}
\left(1-\frac{z_\alpha^Tz_\beta}{\|z_\alpha\|_2\|z_\beta\|_2}\right),
\quad
\sum_\beta\Gamma_{\alpha,\beta}=\omega_{a,\alpha}^{(n)},\quad
\sum_\alpha\Gamma_{\alpha,\beta}=\omega_{b,\beta}^{(m)}.
\]

The masses \(\omega\) are normalized \(\kappa\) on each support. \(\Gamma\) is optimized per pair. The symmetric discovery dissimilarity matrix contains all pairs in the explicitly selected representative cohort. Complete linkage merges the pair of clusters with smallest maximum cross-cluster dissimilarity, subject to the researcher-selected cutoff \(\tau_{\rm OT}\). Every measured discovery pair in a published cluster is within that cutoff. Held-out distances evaluate the fixed partition. A common reference built from the three models and low cosine OT give evidence about context similarity. Circuit and causal evidence still require matched model and anchor identity.

## Execution

Use a new output directory for each support or clustering configuration. The GPU capture replays whole natural pools, aligns each exact native token and checks the archived gate values and support before sealing the selected pooled rows. The old archive contains coefficients and selected gate columns, so it cannot supply these pooled vectors. No external text encoder is used.

```bash
python -m aspd.multimodel.cli.discover_shared_reference_supernodes prepare \
  --measurements "$measured" --top-k "$k_ctx" --output "$prepared"

python -m aspd.multimodel.cli.discover_shared_reference_supernodes capture \
  --measurements "$measured" --prepared "$prepared" \
  --config configs/multimodel/posthoc/v31_500m_validation_v1.yaml --output "$reference"

python -m aspd.multimodel.cli.discover_shared_reference_supernodes cluster \
  --prepared "$prepared" --reference "$reference" \
  --threshold "$tau_ot" --output "$pilot"

python multimodel_dashboard/scripts/install_semantic_supernodes.py \
  --pilot "$pilot" --workbench "$workbench"
```

Use `--per-model` on `prepare` for an explicit cap by existing representative β. By default it uses all representative components in the contribution archive. Both the support size and the clustering cutoff are explicit. The shared-reference release uses its own pointer within Semantic Supernode Discovery.

CPU preflight on the original contribution archive selected 300 components with `--top-k 8 --per-model 100`. It found 1,962 distinct support anchors across 192 frozen pools. Median distinct document coverage among the top eight anchors was seven on discovery and six on held-out. These are measurements of the support inventory. The initial preflight ran without reference capture. The pooled vector export alone would occupy about 7.7 MiB in float32 for this inventory. GPU time includes replay of the full input pools and model loading.

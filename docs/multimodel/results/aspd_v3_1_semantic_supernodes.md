# Semantic Supernode Discovery on Natural Validation

## Result

The bounded pilot discovered reproducible similarities in the contexts where some ASPD components contribute. Much of the apparent similarity came from shared documents. The experiment produced 17 archived supernode links, including one recurring SmolLM2 link across two independently identified documents. It did not establish a shared causal circuit across models. The workbench now exposes the measured partitions, transported contexts, contribution examples, and exact-anchor Taylor witnesses alongside the component-level Semantic Discovery baseline.

## Methodological reference and code audit

Cao, Do, and Thai define activation-weighted empirical measures in Section 2 and feature distance in Eq. (3). Eq. (5) gives a partition objective with mean within-group OT. Algorithm 2 constructs a shared reference and compresses the nodes of an existing feature circuit. Section 6.1 and Figure 1 demonstrate compression on circuits with known tasks. Section 6.2 examines semantic groups recurrent across circuits. Here the component universe is formed before looking at a graph, and the supernode partition is frozen before projecting dependency measurements. This change removes a source of prior evidence about component function. [Paper v3](https://arxiv.org/pdf/2605.28567)

The official compression code at `83a78ba56df9744709f751c425455364482768bb` uses exact EMD with Euclidean ground costs. It divides the resulting matrix by its maximum and uses complete linkage. Algorithm 2 describes a Sinkhorn computation. The repository's historical feature-matching helper and its corrected counterpart differ in how they undo pair-dependent input scaling for a GeomLoss objective with `p=2`. The corrected helper also permits a fixed scale. The square root and scale correction for that objective does not apply to the sum of Euclidean ground costs used here. [Compression code](https://github.com/tue147/semantic-optimal-transport/blob/83a78ba56df9744709f751c425455364482768bb/graph_visualizer.py#L477) · [Feature-matching helpers](https://github.com/tue147/semantic-optimal-transport/blob/83a78ba56df9744709f751c425455364482768bb/feature_matching/utils.py)

The previous ASPD component matcher is retained. Its `semantic_ot.transport` normalizes the masses and solves a balanced linear program on cosine ground costs for L2-normalized embeddings. The previous comparison shortlisted gate-aware matches and used its own support subsampling and matched-zero controls. That catalog retained three pairs from 48 candidates. At the reported controlled comparison, OT, centroid similarity, and exact-anchor overlap each ordered all 35 tested pairs ahead of both controls. There is no demonstrated advantage for OT on that benchmark. The new mode reuses the mass and sparse marginal routines. It changes the cost, support policy, and unit of discovery.

## Formulation

The component pool is the union of existing \(\mathcal R_{p,i}^{(n)}\) based on natural-validation \(\beta_{j,i}^{(n)}\), \(K_{\rm rep}=3\), and \(\tau_{\rm rep}=0.07\). The frozen rank-one operator remains \(P_{j,i}^{(n)}=u_{j,i}^{(n)}(v_{j,i}^{(n)})^T\), with coefficient \(e_{j,\alpha,i}^{(n)}=g_{p(j),\alpha,i}(v_{j,i}^{(n)})^T x_{j,\alpha}^{(n)}\) and contribution \(\Delta y_{j,\alpha,i}^{(n)}=e_{j,\alpha,i}^{(n)}u_{j,i}^{(n)}\).

For \(a=(j_a,i_a)\), measure \(r_a^{(n)}(\alpha)=|e_a^{(n)}(\alpha)|\|u_{j_a,i_a}^{(n)}\|_2/\sigma_{j_a}^{(n)}\). The exported \(e_a\) retains its sign, which depends on the factor orientation. A measured zero stays in the contrast inventory and contributes no support mass. The support policy keeps the strongest nonzero measurement per document, discards near-identical causal prefixes (4-gram Jaccard ≥ 0.85 over the last 64 lexical units), and takes up to \(K\) largest \(r_a^{(n)}\) in each split. The weights are

\[
\widehat w_{a,\alpha}^{(n)}=r_a^{(n)}(\alpha)/
\sum_{\alpha'\in\mathcal I_a^{(n),K}}r_a^{(n)}(\alpha'),
\qquad
\widehat\mu_a^{(n)}=\sum_{\alpha\in\mathcal I_a^{(n),K}}
\widehat w_{a,\alpha}^{(n)}\delta_{z_\alpha^{(n,\mathcal R)}} .
\]

The pilot uses a common, frozen MiniLM reference, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The 384-dimensional representation is an L2-normalized mean of the attended wordpiece states over the last 40 model tokens of the causal prefix, including the encoder's special tokens. `--reference prefix` changes the limit to 128 and implements the prior prefix baseline. Native reference variants load the decoder-block outputs at exact canonical anchor IDs, concatenate all layers or the eight selected ASPD layers in native order, and leave the vectors in raw units. The within-model calculations use one model's coordinates. Cross-model comparisons continue to use the common context reference. Native tensors were not present in the sealed contribution archive.

For empirical measures, solve Eq. (3) with \(C_{\alpha,\alpha'}=\|z_\alpha-z_{\alpha'}\|_2\) and the prescribed row and column marginals. `--epsilon` solves the positive-entropy regularized problem through log-domain Sinkhorn iterations and returns \(\langle T_\varepsilon,C\rangle\). It excludes the entropy term from the returned number and does not report a debiased Sinkhorn divergence. The clustering matrix computes an unordered pair once and copies it symmetrically. Its diagonal is fixed at zero. With regularized transport this is a clustering convention because the raw regularized self-transport cost can be positive.

Average linkage builds the hierarchy on each dense within-model matrix. The discovery cut is the 35th percentile of each component's nearest-neighbor distance within that model. A first fixed cut of 0.80 produced only singletons. The revised cut used the discovery distances before inspection of held-out coherence. The implementation reports Eq. (5) at the chosen cut, with singleton cost zero. Average linkage is a heuristic for making the partition and does not solve Eq. (5) globally. The frontend allows preview of alternate cuts without reassigning the stored graph witnesses.

For cross-model comparisons, \(\widehat\mu_S^{(n)}=\sum_{a\in S}\lambda_a\widehat\mu_a^{(n)}\). The default \(\lambda_a=1/|S|\) is an ASPD extension. The mixture helper also accepts a supplied weight vector. The pilot shortlists the two nearest mixture centroids per cluster in both directions, then compares complete mixture distributions with exact OT in the common reference. Distances for unevaluated pairs remain absent. An exploratory discovery cost cutoff of 1.10 is exposed in the catalog and has no evidentiary threshold for calling a mechanism shared.

For a source \(S_A\) and eligible target \(a\), compute
\[
E_{S_A\rightarrow a}^{(n),\mathrm{Taylor}}(\alpha)
=\sum_{b\in S_A\cap\mathcal U^{(n)}(a)} E_{b\rightarrow a}^{(n)}(\alpha),
\quad
w_{S_A\rightarrow a}^{(n),\mathrm{Taylor}}(\alpha)
=|E_{S_A\rightarrow a}^{(n),\mathrm{Taylor}}(\alpha)|\|u_{j_a,i_a}^{(n)}\|_2/\sigma_{j_a}^{(n)} .
\]
Average over \(S_{B\mid A}\), all structurally eligible targets in \(S_B\). Eligibility uses the existing model, layer, and intra-layer order. The aggregate is undefined for an empty eligible set. When some terms are unavailable, the UI can show the observed-target partial mean and coverage. Missing signed terms can cancel or strengthen observed effects. An archive projection is complete only where each eligible term is present, and remains Taylor evidence. Supernodes can span several layers, so a cycle of aggregate edges can be compatible with an acyclic component computation. Edge evidence is limited to an exact aligned anchor.

The native intervention hook applies \(e_b=0\) at that anchor for each \(b\in S_A\) in one forward. The binary gates are frozen, and each removed contribution is evaluated on its matrix input in the intervened forward. The replay evaluates \(E_{C\rightarrow a}^{(n),\mathrm{exact}}(\alpha)\) as clean minus intervened downstream coefficient, using the same target normalization and eligible denominator. It can also report normalized matrix and layer output perturbations, JS divergence in nats, and signed clean-minus-intervened log probability of specified continuations. Matrix scales come from the same importance reference. Layer scales must be supplied from a matching calibration. Both forwards are confined to the same anchor and frozen source weights. A total intervention effect can include indirect paths.

## Measured experiment

The sealed Natural Validation contribution wave used checkpoint step 61,036, manifest `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`, natural BatchTopK with 2,048 aligned anchors, and 192 pools. It contained 393,093 usable anchors and 1,254 documents after the shared-boundary document was removed. The cache is limited to 1,747 representative components attached to 197 gates. It is not a census of the dictionary.

The pilot selects 100 representative components per model in a deterministic gate-balanced order, using discovery coverage of at least 20 documents. It reserves a second representative on 15 sampled gates per model to evaluate the shared-gate baseline. Document splits use the existing seed 41793 and keep a document on one side. Three Gemma components missed the held-out coverage requirement and were excluded. The 297 retained components have 7,025 exported held-out examples, mean 23.65 and minimum 11 per component. The OT distributions use the top eight per component per split. Heatmaps cover the strongest three and up to two measured-zero controls per split, rendering actual contributions at available aligned native positions. Other examples display an anchor outline. These contrasts come from the same document as strong examples and control document identity without controlling every linguistic variable.

| Model | Components | Supernodes | Singletons | Held-out pairs | OT within supernodes | Random, matched cluster sizes | Mean-embedding clusters | Shared-gate pairs | Split co-assignment Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma | 97 | 80 | 65 | 20 | 0.769 | 1.338 | 0.850 | 0.636 (14) | 0.625 |
| SmolLM2 | 100 | 80 | 64 | 25 | 0.782 | 1.335 | 0.794 | 0.487 (15) | 0.469 |
| Granite | 100 | 81 | 66 | 24 | 0.796 | 1.336 | 0.882 | 0.590 (15) | 0.667 |

The random baseline consists of 200 permutations of component assignments with matching cluster sizes. Its 2.5–97.5% quantiles are Gemma [1.299, 1.355], SmolLM2 [1.303, 1.349], and Granite [1.295, 1.353]. These are null quantiles, not confidence intervals on semantic coherence. The mean-embedding cut uses the same cluster count and a potentially different group size distribution. The shared-gate pairs are a selected baseline with common binary support. Their lower mean OT is a useful check on that support's influence. A gate-balanced sensitivity cohort without reserved repeated gates retained 298 components and gave lower split agreement of 0.237, 0.108, and 0.207 respectively.

The same-document control is material. Blocking all within-document transport cells increases mean held-out within-cluster OT to 1.249, 1.242, and 1.220 respectively. All 69 restricted plans were feasible. The original plans moved an average 58%, 50%, and 50% of mass within the same document. Of 763 centroid-prefiltered cross-model comparisons, 91 passed the exploratory screen. Their median held-out mass sent within the same document was 60.6%, including 27.0% at the exact same anchor. Blocking same-document cells pushes the lowest two discovery-ranked pairs from 0.512 to 1.243 and from 0.297 to 1.211 on held-out documents. The screen is sensitive to document and token recurrence. Equivalent functions and recurring reasoning roles require a stronger controlled test.

A 36-pair exact/Sinkhorn diagnostic found maximum exact marginal error \(1.95\times10^{-16}\), maximum exact reversal difference \(4.45\times10^{-16}\), median increase of 0.033 from \(\varepsilon=0.1\) regularized transport cost, and median absolute change 0.027 when halving support to four. The archive contains the pair-level results. Independent checks specifically masking punctuation, balancing source/topic, and systematic reference-space comparisons remain future work.

## Examples and counterevidence

- In Gemma `gemma-3d55d98fe499`, four components across Q, K, V and MLP Down contribute in arithmetic-question passages. In two strong held-out examples for `gemma.L16.q/5764`, the prefix ends after a remainder question and a primality question. `gemma.L14.k/1990` also contributes on prefixes ending in “What” and “Is”. Granite `granite-3b2b0978fb4d` contains Q and O components with overlapping arithmetic passages. These records support a recurring document and template family. They leave open which features respond to equation formatting, query structure, or arithmetic information. The archive did not establish a matching circuit.
- In SmolLM2 `smol-e1f8eee6e6ad`, the Q and K components with PRE:6980 have strong held-out contributions at “wound” and “incision”. A third component with a different gate has strong contributions in skeletal muscle passages. This provides a topic-sensitive cluster with an informative outlier. The measured examples leave the native read/write role unresolved.
- The archived link from `smol-ba6f8023f1e9` to `smol-0a2f18b0d26a` contains `smol.L08.q/2280 → smol.L09.q/8109` in one GitHub code document and one StackExchange code document. Both displayed anchors end at a slash. The normalized Taylor weights are 0.01561 and 0.01413. These are two observations of one edge. Slash sensitivity and surrounding code structure are plausible alternatives to a higher-level computational role.

Projection over the archived validation graph index found 17 distinct links in 23 anchor/model scopes. The UI shows 18 document-distinct link witnesses, 12 with all eligible terms available and 6 partial. All 18 displayed witnesses have exact archived anchor IDs and belong to validation pools 50–74. The newer five-component Taylor export contained no matching target in this cohort. No displayed projected anchor contains witnesses from two models. A circuit-first compression baseline on comparable multiedge paths, Jaccard/TV/total strength on matched causal structures, and agreement with exact supernode ablation are unmeasured. The exact replay is implemented and numerically tested on a tiny model, including simultaneous ablation and fixed-gate output metrics. The current pilot does not justify an exhaustive causal run or a universal label.

## Artifacts and reproducibility

- Repository base `ad3ef9dcb3a467b5dc8955b8dfbe5e1d80e0fc7b`. Pilot contract records the analysis source hashes, checkpoint, original model and tokenizer revisions, coverage hash, pool contract, and split seed. The measured contribution collector was `d4bf2f2189390f088f43cc9711cf29709509c20e`.
- Under `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/`, the sealed pilot is `semantic-supernode-pilot-20261127-v4/`. `COMPLETE.json` SHA-256 is `fbe6d3d47535161d9f825764a51be4fed4288d253c60b39842b3043e726959bb`. Sibling files ending in `-archived-witnesses.json` and `-dependence-audit.json` hold the graph and within-cluster same-document controls.
- On one CPU thread, the pilot took 188.1 s after startup with 1.40 GB peak RSS and 200 MiB disk. Loading and encoding 11,121 distinct prefix anchors took 102.2 s. The within-model matrices took 51.5 s including clustering and baselines. Archived edge projection took 2.2 s after Python startup. The compact frontend payload is 45 MiB and loads individual component evidence on demand. The pilot consumed 0 GPUs. The v3.1 collector recorded about 60 GB peak reserved device memory per pool in the original extraction. The exact supernode replay's device peak has not been benchmarked.
- The optional native validation CLI requires a frozen JSON plan with the catalog `runId`, `gates: "frozen-binary"`, and `requests` containing `pool`, exact `anchor`, source and target supernode IDs `from` and `to`, and optional matrix-matched `controls`. Plan selection should use archive discovery evidence and freeze matched-control rules before reading held-out effects. A useful timing test would replay three exact held-out anchors, one per model, with one same-matrix active control per anchor. The initial resource request would be one B200, 8 CPU, 32 GB host RAM, 45 minutes wall time and less than 1 GB output. Full replay and device peak measurements require a new job. A useful scientific gate is a recurrence check under document- and template-separated controls before expanding the GPU workload.

From the implementation worktree, using its Python environment:

```bash
runtime=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime
prior="$runtime/semantic-ot-v2-runs/v2-d4bf2f2-20261127"
pilot=/tmp/aspd-semantic-supernodes-reproduction
posthoc=/blue/thai/share/multimodel-ASPD/evaluations/minhdatle/posthoc/v31_500m_validation_v1_20261122T223718

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m aspd.multimodel.cli.discover_semantic_supernodes \
  --measurements "$prior" --coverage "$prior-analysis-192/coverage.json" \
  --encoder "$runtime/semantic-ot-v2-runs/minilm-1110a243" \
  --graphs "$prior-full-taylor-v2" --repeat-gates 15 \
  --discovery-neighbor-quantile .35 --cross-maximum 1.10 --output "$pilot"

python -m aspd.multimodel.cli.attach_semantic_supernode_witnesses \
  --pilot "$pilot" --graph-index "$runtime/large-aspd-discovery-full-v1/index" \
  --recovery "$runtime/semantic-ot-natural-validation-v1/recovery" \
  --posthoc "$posthoc" --output "$pilot-archived-witnesses.json"

npm --prefix multimodel_dashboard run build -- --outDir /tmp/aspd-supernode-workbench
python multimodel_dashboard/scripts/install_semantic_supernodes.py \
  --pilot "$pilot" --archived "$pilot-archived-witnesses.json" \
  --workbench /tmp/aspd-supernode-workbench
```

Preserve the workbench's existing Natural Validation, Induction, Negation, and Semantic Discovery static assets when staging the new directory. Tests exercised the transport marginals, exact and entropic solvers, mixture, clustering cuts, archive coverage semantics, simultaneous native intervention, and frontend asset paths.

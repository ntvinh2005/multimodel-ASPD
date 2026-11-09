# ASPD v3.1 gate-collapse root-cause investigation

Date: 2026-11-08

Status: `ROOT_CAUSE_SUPPORTED; SINGLE-SEED RECOVERY PASSES PREREGISTERED PILOT GATES; NOT READY FOR 500M`

## A. Executive summary

The current ASPD gate collapses through a self-reinforcing sparse-optimization loop, not because it is initialized with too few candidates and not because negative logits alone are an initiating cause.

1. Pool-wide BatchTopK initially gives reconstruction gradient to a subset of feature/anchor pairs. Positive but unselected logits receive no direct main-loss gradient because the selected mask is discrete; negative logits additionally receive zero derivative through ReLU.
2. Early winners therefore receive nearly all useful encoder/readout/decoder updates. Concentration is already pronounced while thousands of positive candidates remain, so candidate starvation is a later stage rather than the initiating event.
3. Local encoders, the shared cross encoder, and the readout weight jointly move the logit distribution. Fixed-anchor Shapley replay identifies the readout weight as the largest single contributor, but all three learned modules contribute substantially. Bias is small.
4. The historical post-ReLU AuxK path cannot reach negative logits and reaches only a small subset of positive dead features. It does not reverse the winner advantage.
5. As the positive population shrinks, ReLU and TopK starvation make feature death increasingly absorbing. This explains both progressive growth of the inactive population and the late state of approximately 40 positive candidates from 8,192 features.

The smallest matched intervention was a parameter-free per-anchor feature-axis standardization immediately before ReLU:

\[
u^{\mathrm{gate}}_{a,c}
=
\frac{u^{\mathrm{projection}}_{a,c}-\mu_a}
{\sqrt{\sigma_a^2+10^{-5}}}.
\]

It preserves the within-anchor feature ranking and uses no forced support. It removes per-anchor shift and scale as degrees of freedom in pooled competition and gives dense normalization-mediated gradient to projection logits even when a logit is not itself selected. Component indices, natural input-conditioned support, continuous activation coefficients, detached binary internal support, and model-specific rank-one operators are unchanged.

At step 1,221, the matched full-only control had effective sizes 45.8/50.1 and top-1% shares 99.79%/99.48% (PRE/MID). The treatment had effective sizes 2,130/1,808 and top-1% shares 18.69%/21.69%. On 100,352 document-disjoint validation anchors, effective size improved from 46.4/52.0 to 2,731.8/2,338.8 and top-1% share fell from 99.74%/99.37% to 15.63%/19.18%.

This was not a diversity/reconstruction trade: normalized activation FVU improved in all four model/site cells by 0.111--0.165, and equal-site internal FVU improved by 0.0645. Paired 10,000-sample document bootstrap intervals were wholly below zero for activation and internal FVU.

The result supports changing the main gate formulation (recommendation 3), but only as an exploratory one-seed result. It is ready for longer multi-seed confirmation, not a 500M campaign. Semantic coherence, cross-model causal correspondence, component stability across seeds, and a longer exposure horizon remain unverified.

## B. Environment and evidence inventory

### Repository and execution state

- Training/evaluation worktree: `gate-collapse-recovery-clean`.
- Training Git SHA: `5b6365107a72f224bfa3c6a224a54f96372d546c`.
- Diagnostic Git SHA: `0f5db1cf0cbf7ba5972b912639c1b4ff8c956ef5`.
- The primary checkout's unrelated dirty work was not modified.
- New experiments loaded only `google/gemma-2-2b-it` and `HuggingFaceTB/SmolLM2-1.7B-Instruct` at pinned revisions. No Qwen or Granite model was loaded. Historical three-model Granite evidence was used only as an unmatched prior.
- Some evaluator symbols and manifest status strings still say `granite_normalized_target`; this is a legacy implementation name. The frozen configs, model revisions, 112-target count, and source-correctness records establish that the executed jobs were Gemma + SmolLM2 only.

### Usable evidence

| Evidence | State at audit | Use |
|---|---|---|
| Retired 500M run | Final reports only; runtime checkpoints deleted | Endpoint context only; no trajectory claim |
| Encoder/AuxK factorial | Completed report and held-out artifacts | Rejects the tested linear mixer and Softplus-AuxK variants as solutions |
| Matryoshka/full-only source array `50231651` | Complete step-1,000 checkpoints for both arms | Read-only fixed-anchor diagnostic source |
| Recovery array `50276566` | Both tasks completed `0:0`; final held-out artifacts available | Preserved and checked; not required for the matched new comparison |
| Fixed-anchor diagnostic `50270783` | Complete, checksummed, read-only | Root-cause gradient and module attribution |
| New matched pilot `50280055` | Both arms complete through training, checkpoint smoke, and held-out evaluation | Controlled recovery evidence |

The original 500M run differs in model composition, target normalization, and decoder constraints. It is not used as a matched causal comparator.

## C. Architecture audit

### C.1 Implemented gate equations

For site \(s\in\{\mathrm{PRE},\mathrm{MID}\}\), each model-specific local encoder packs eight selected-layer representations into latent tokens. A differentiable all-gather supplies all model tokens to the shared adapter/cross-model encoder. The feature readout mean-pools its token axis and applies one shared linear map:

\[
h_a^s=\operatorname{MeanPool}(E^s_{\mathrm{cross}}(\{E^{m,s}_{\mathrm{local}}(x_a^m)\}_m)),
\qquad
p_{a,c}^s=W_c^s h_a^s+b_c^s.
\]

The control uses \(u=p\). The treatment uses \(u=\operatorname{zscore}_c(p)\). Both then compute

\[
a_{a,c}^s=\operatorname{ReLU}(u_{a,c}^s).
\]

The implementation is in `FeatureReadout` ([encoder.py](../../aspd/multimodel/v3/encoder.py), lines 224--276). `nn.Linear` uses the native PyTorch initialization; no custom output-weight or output-bias initializer is applied.

For each 2,048-anchor pool, BatchTopK selects the largest \(K A=32A\) entries over the joint anchor-feature grid:

\[
M^s=\operatorname{TopKMask}_{32A}(a^s),
\qquad
g^{s}_{a,c}=a^s_{a,c}M^s_{a,c},
\qquad
g_{a,c}=\mathbf 1[g^{s}_{a,c}>0].
\]

The selection implementation is [batch_topk.py](../../aspd/multimodel/batch_topk.py), lines 31--72 and 75--155. It ranks all valid anchor-feature entries in a pool, retains the original nonnegative values, and does not use the tracked inference threshold to select training support. Deadness means no firing for 320 consecutive pools ([gating.py](../../aspd/multimodel/v3/gating.py), lines 52--105).

### C.2 Activation versus internal reconstruction

Activation reconstruction uses continuous selected coefficients:

\[
\hat r^{m,s,\ell}_a
=d^{m,s,\ell}_0+D^{m,s,\ell}g^s_a,
\]

against per-anchor L2-normalized activation targets. The new control and treatment both use full-dictionary FVU, unconstrained activation decoders, and the historical post-ReLU AuxK. The full-only path is [objective.py](../../aspd/multimodel/v3/objective.py), lines 147--191.

Internal reconstruction uses detached binary support:

\[
\hat y^{m}_{j,a}
=b_j^m+
\sum_c g_{a,c}
u^{m}_{j,c}\left(v^{m}_{j,c}\right)^\top x^{m}_{j,a}.
\]

The continuous and binary sparse supports are built separately in [model_owned.py](../../aspd/multimodel/v3/model_owned.py), lines 339--384. The internal path explicitly detaches the dense code and declares `code_semantics="binary_support"` (lines 616--650). Therefore internal FVU trains the model-specific rank-one bank but does not train the gate.

### C.3 Gradient routing

- Main activation loss: gradients reach the activation decoder and selected positive gate values. They do not pass through the discrete TopK mask to positive-unselected logits, and ReLU gives zero direct derivative to negative logits.
- Internal loss: zero gate gradient by construction because binary support is detached.
- Historical AuxK: selects up to 2,048 currently dead features from post-ReLU scores and reconstructs a detached main residual ([objective.py](../../aspd/multimodel/v3/objective.py), lines 194--265). Negative logits still receive zero gradient. Its coefficient is 1/32.
- Loss balancing: each model/site/layer activation term receives `0.5 / (2 models * 8 layers)`; internal terms give one half to PRE's 3 matrices and one half to MID's 4 matrices across both models. The actual chunked backward path is [train_aspd_v3_2b.py](../../aspd/multimodel/cli/train_aspd_v3_2b.py), lines 582--668.
- Optimizer: Adam, betas `(0.9, 0.99)`, zero weight decay, constant schedule; encoder LR `1e-4`, decoder and rank-one LR `3e-4`.

### C.4 Initialization audit

The fixed-anchor 16-way replay substituted initial versus trained local encoder, shared encoder, readout weight, and readout bias. With every module at initialization, PRE/MID logits were approximately 50% negative with 4,095/4,079 positive candidates per anchor and more than 5,100 distinct features firing in one fixed pool. This refutes unhealthy output-bias or candidate initialization as the cause of the observed long-run state.

## D. Root-cause investigation

Status terms below follow the requested evidence standard.

| Hypothesis | Evidence | Conclusion |
|---|---|---|
| H1 ReLU starvation | Direct raw-logit gradient is exactly zero for negative logits under main loss and historical AuxK. It is also zero for positive-unselected logits under main loss. Collapse begins while candidate counts are still in the thousands. | **PARTIALLY SUPPORTED**: strong absorbing/maintenance mechanism, not sufficient as the initiator. |
| H2 winner-take-all reinforcement | Main-loss gradient is nonzero for 99.98% or more of naturally selected events, versus 0% for positive-unselected events. Winner rows have much larger update/Adam-moment histories. Concentration rises before sign collapse. | **SUPPORTED**. |
| H3 logit distribution drift | Initial-to-trained replay moves candidate counts from about 4,090 to 39--41 and negative fraction from about 50% to 99.5%. Shapley attribution is multi-module; readout weight is largest, shared/local encoders substantial, bias small. | **SUPPORTED**. |
| H4 AuxK revival failure | AuxK gives zero gradient to negative logits. Only about 0.15--0.50% of positive-unselected events receive an AuxK gradient on the fixed pool, and dead-feature readout gradients are sparse. Historical Softplus AuxK worsened the gate/internal result. | **SUPPORTED** for the implemented historical AuxK; other revival methods remain untested. |
| H5 Matryoshka pressure | Matryoshka collapses earlier, but the matched full-only arm also reaches approximately 40 candidates and effective size near 50 by step 768--1,221. | **PARTIALLY SUPPORTED** as accelerator; **NOT SUPPORTED** as sole cause. |
| H6 capacity/data/initialization | C=8192 starts with about 4,090 candidates and broad initial pool coverage. Linear mixing delays but does not prevent collapse. | Initialization failure **NOT SUPPORTED**; broader data-domain effects remain **INCONCLUSIVE**. |
| Cross-model gradient conflict | Main shared-encoder gradient cosines are positive, 0.61--0.94 across sites/arms. | **NOT SUPPORTED** as the primary mechanism. |
| Decoder constraint as sole current cause | Removing the historical decoder constraint fixed the prior all-zero failure but left effective size around 47--50. | **NOT SUPPORTED** for the present collapse. |

### D.1 Why the inactive population grows

The diagnostic resolves the temporal logic that a static “negative logits get no gradient” statement misses:

1. At step 256, the full-only control still has roughly 3,000 positive candidates, yet its pool effective size is only about 1,040. Selection concentration therefore precedes near-total sign death.
2. Selected feature/anchor pairs update both their readout rows and the shared/local representation functions. Positive-unselected rows do not receive direct main-loss updates, but their logits can still change as the shared representation changes around them.
3. Feature firing frequency and trained mean logit are positively associated; dead features end with more-negative mean logits. Winner ranking becomes a persistent optimization advantage.
4. Once a feature becomes nonpositive on all relevant anchors, neither main reconstruction nor historical post-ReLU AuxK can directly restore it. The number of inactive features therefore grows, which further concentrates the next pool's gradient allocation.

This is a positive-feedback system: sparse selection initiates unequal learning, multi-module drift increases the sign/rank gap, and ReLU/AuxK make that gap increasingly irreversible.

The recovered final held-out results strengthen the H5 conclusion. Matryoshka versus full-only effective size was 48.65 versus 46.43 PRE and 51.21 versus 52.03 MID; positive candidates were 37.90/38.86 versus 37.41/40.87. Both objectives therefore reach the same concentrated endpoint. Full-only fires more held-out feature identities (2,482/2,923 versus 555/2,282), but those additional rare firings do not materially change entropy or top-1% dominance. Reconstruction numbers are reported in their native objectives and are not treated as numerically equivalent evidence across objectives.

### D.2 Module attribution

On the matched step-1,000 checkpoints, fixed-anchor Shapley contributions to mean-logit drift were:

| Arm/site | Local encoder | Shared encoder | Readout weight | Readout bias |
|---|---:|---:|---:|---:|
| Matryoshka PRE | -0.127 | -0.239 | -0.271 | -0.009 |
| Matryoshka MID | -0.132 | -0.327 | -0.370 | -0.010 |
| Full-only PRE | -0.109 | -0.147 | -0.190 | -0.012 |
| Full-only MID | -0.199 | -0.149 | -0.252 | -0.008 |

The readout weight is the largest single term, but no single-module account is adequate. Bias-only interventions are not justified by the evidence.

## E. Controlled recovery experiment

### E.1 Hypothesis and controls

The preregistered hypothesis was that per-anchor feature-axis standardization would block harmful affine logit drift and provide normalization-mediated projection-logit gradients while preserving natural within-anchor ranking. The matched control used the same full-only objective without standardization.

Matched factors included parameter tensor initialization, ordered aligned-anchor batches, model revisions/layers, C=8192, K=32, full-only activation objective, post-ReLU AuxK, unconstrained decoders, detached binary internal support, optimizer, seed, data manifests, and evaluation documents. Initial common-state hashes were identical across arms. Source immutability and native projection checks passed for all 112 targets.

No support was forced, sampled, or injected. Training and evaluation used the same gate.

### E.2 Training trajectory

Diagnostics are labeled by parameter step; the row that measures step 256 is written after the following forward pass.

| Parameter step | Arm | Candidates PRE/MID | Effective size PRE/MID | Top-1% share PRE/MID | Negative logits PRE/MID |
|---:|---|---:|---:|---:|---:|
| 256 | control | 3024 / 3108 | 1041 / 1089 | 40.36% / 39.32% | 63.09% / 62.06% |
| 256 | z-score | 5888 / 5993 | 996 / 765 | 39.72% / 45.48% | 28.13% / 26.84% |
| 512 | control | 2142 / 77 | 584 / 279 | 47.41% / 63.05% | 73.85% / 99.06% |
| 512 | z-score | 6037 / 6106 | 1432 / 1250 | 30.55% / 33.31% | 26.30% / 25.46% |
| 768 | control | 35 / 45 | 45.9 / 62.3 | 99.30% / 95.95% | 99.57% / 99.45% |
| 768 | z-score | 6124 / 6241 | 1530 / 1431 | 25.59% / 26.89% | 25.24% / 23.82% |
| 1221 | control | 37.7 / 41.2 | 45.8 / 50.1 | 99.79% / 99.48% | 99.54% / 99.50% |
| 1221 | z-score | 5944.9 / 5860.2 | 2130 / 1808 | 18.69% / 21.69% | 27.43% / 28.46% |

At step 1,221 the treatment's unstandardized projection logits were only 0.83%/0.59% negative, while standardized gate logits were 27.43%/28.46% negative. The intervention therefore does not merely hold projection logits at zero mean during optimization: it makes selection invariant to their per-anchor affine drift and changes the gradient geometry.

Training activation loss improved from 0.7089 to 0.5731 and internal loss from 0.6619 to 0.6012. Median optimizer-step time was 2.075 s control versus 1.921 s treatment; this node-paired difference is an engineering observation, not a stable speed claim.

### E.3 Held-out reconstruction and dictionary health

The evaluation used 100,352 validation anchors from 306 source documents, disjoint by document under the frozen split.

| Metric | Control | Z-score | Treatment - control |
|---|---:|---:|---:|
| Gemma PRE normalized activation FVU | 0.68260 | 0.54783 | -0.13477 |
| Gemma MID normalized activation FVU | 0.68213 | 0.57154 | -0.11059 |
| SmolLM2 PRE normalized activation FVU | 0.70144 | 0.53692 | -0.16453 |
| SmolLM2 MID normalized activation FVU | 0.69975 | 0.56124 | -0.13851 |
| Equal model/site activation delta | -- | -- | **-0.13710** |
| PRE internal FVU | 0.57254 | 0.51707 | -0.05547 |
| MID internal FVU | 0.71264 | 0.63902 | -0.07361 |
| Equal-site internal FVU | 0.64259 | 0.57804 | **-0.06454** |

| Dictionary metric | Control PRE/MID | Z-score PRE/MID |
|---|---:|---:|
| Positive candidates/anchor | 37.41 / 40.87 | 5944.36 / 5859.95 |
| Natural nonzero selections/anchor | 32 / 32 | 32 / 32 |
| Entropy-effective size | 46.43 / 52.03 | 2731.77 / 2338.76 |
| Features firing | 2482 / 2923 | 5521 / 5503 |
| Top-1% event share | 99.740% / 99.371% | 15.630% / 19.184% |
| Top-10% event share | 99.910% / 99.792% | 44.152% / 49.590% |
| Training-recently-dead features | 4287 / 4275 | 2653 / 2619 |

The treatment passes every preregistered gate-health, diversity, reconstruction, source-immutability, and resume requirement. It does not eliminate all feature death: roughly 2,600 features remain recently dead, and about 2,680 never fire on this held-out set.

### E.4 Paired uncertainty

For each source document, full normalized activation FVU was macro-averaged across 32 model/site/layer endpoints. Internal FVU was averaged within PRE and MID and then equally across sites over 112 matrices. Treatment-minus-control differences were bootstrapped over the 306 matched documents with 10,000 samples and seed 0.

| Paired document metric | Mean delta | 95% percentile interval |
|---|---:|---:|
| Activation macro FVU | -0.15312 | [-0.15608, -0.15018] |
| Equal-site internal macro FVU | -0.07321 | [-0.07693, -0.06958] |

These intervals quantify document-level stability for this one trained seed; they are not across-seed uncertainty.

## F. Scientific validity

### Preserved invariants

- Shared component correspondence: one shared PRE and one shared MID feature identity remain common across models.
- Input-conditioned selection: all reported support is naturally selected from each input; no exploration support is counted.
- Model-specific rank-one mechanisms: each model/matrix retains its own \(u_{j,c}^m(v_{j,c}^m)^\top\).
- Causal component identity: the component index and binary ablation unit remain stable between training and evaluation.
- Continuous/binary distinction: activation reconstruction uses continuous sparse values; internal reconstruction uses detached binary support.
- Frozen source models: immutability and native-projection parity passed for Gemma and SmolLM2.

### Method change and remaining risk

Z-scoring is a real gate-formulation change, not a neutral numerical patch. Each feature's standardized score depends on all 8,192 projection logits for that anchor, and anchor-specific scaling changes score comparability inside pool-wide BatchTopK. The dense layer-normalization Jacobian is precisely why unselected projection logits can receive indirect gradient, but it also means a component's coefficient is relational to the full dictionary.

This remains compatible with individual binary causal ablation and shared indices. However, high utilization does not establish that the additional features are coherent, nonredundant, causally reproducible, or aligned across models. Those Level-2 claims were not tested here.

## G. Recommended solution

Choose option **3: change the main gate formulation**, specifically the tested per-anchor feature-axis standardization before ReLU, for the next confirmation stage.

Reasons:

1. It targets the supported affine-drift/starvation feedback without forcing inactive features into binary support.
2. It preserves the core ASPD component semantics and the continuous-versus-binary distinction.
3. It survives through step 1,221 instead of only delaying collapse to step 768.
4. It improves, rather than trades away, both activation and internal reconstruction in the matched held-out protocol.
5. Its prediction is mechanistically discriminating: projection logits may drift strongly, but standardized natural support remains healthy. The observed endpoint matches that prediction.

Do not adopt the tested Softplus AuxK variant, do not treat removal of Transformer mixers as a solution, and do not reintroduce forced-exploration support without a separate semantic analysis.

## H. Main-training readiness

| Question | Answer |
|---|---|
| Is sustained gate collapse sufficiently addressed? | **Yes through 1,221 steps in one matched seed; not yet at full-run horizon.** |
| Is held-out activation reconstruction acceptable? | **Improved materially versus control**, but absolute FVU 0.537--0.572 still leaves substantial unexplained variance. |
| Is internal parameter reconstruction preserved? | **Yes; it improved** by 0.055 PRE and 0.074 MID. |
| Is dictionary diversity scientifically meaningful? | **Natural utilization is healthy; semantic meaningfulness remains unverified.** |
| Ready for longer confirmation? | **Yes.** |
| Ready for 500M training? | **No.** |

Minimum remaining blockers:

1. Repeat the matched control/treatment comparison for at least two independent seeds to measure training-seed uncertainty.
2. Extend a confirmed treatment beyond 10M exposures with checkpoints before and after the known transition region. A useful next horizon is approximately 50M exposures (about 6,104 steps), not 500M.
3. Evaluate feature coherence, redundancy, identity stability across seeds, shared-model correspondence beyond index equality, and reproducible component ablations using natural evaluation gates.
4. Preallocate storage for retained checkpoints and evaluation before extension.

Suggested abort rules for the next stage:

- natural firings below 31.9 per anchor at either site;
- effective size below 256 or top-1% share above 0.90 at two consecutive diagnostic checkpoints;
- any model/site held-out activation FVU regression above 0.03 versus its matched control;
- equal-site internal FVU regression above 0.02;
- source-weight parity, resume determinism, manifest, or storage-headroom failure.

No main-training job was submitted.

## I. Reproducibility and resources

### Provenance

- Models: Gemma revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`; SmolLM2 revision `31b70e2e869a7173562077fd711b654946d38674`.
- Dataset manifest: `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f`.
- Alignment manifest: `bfd825e51bd4b48602c7822b2ed97ffb919cde588886b72352699e7e077d3c56`.
- Evaluation contract: `57373b08a80a0a4367b0ba324aa640b071fd163c737ac329e7aa29d20614f6cb`.
- Control config SHA-256: `7081dc544aa8d99d53d5bfd87d84101573447e59fc6f94b20021ab6460cf41bd`.
- Treatment config SHA-256: `6a0ea531b7e27a562d62d2385649cc6b2d723d730a748ee73cf2db81ba463d97`.
- Control/treatment checkpoint manifests: `8f075f9b...7750` / `c18e93a7...45f1`.
- Preflight `50278599`: both arms passed save/reload/resume update and identical initial-state validation.
- Pilot array `50280055`: both arms completed with exit `0:0`.

### Validation performed

- GPU preflight: deterministic two-step save/reload/resume-update PASS for both arms.
- Checkpoint smoke evaluation: PASS for both final checkpoints before held-out evaluation.
- Source correctness: frozen parameters, hook immutability, early-exit capture parity, and zero native-projection error PASS for 56 targets/model.
- Comparison code: Ruff PASS; `pytest -q tests/test_multimodel_v3_gate_collapse_comparison.py` -> `1 passed`.
- Paired analysis: 306 common source documents for every selected activation/internal endpoint.

### Resource accounting

| Item | Result |
|---|---|
| New Slurm submissions | `50270457`, `50270783`, `50277092`, `50278599`, `50280055` |
| New B200 usage | **6.9483 GPU-hours** |
| Maximum new pilot GPUs concurrently | 4 B200s; 6 total including the pre-existing recovery job |
| New experiment namespace | 351,981,662,411 bytes |
| Complete checkpoints preserved | 4: two preflight and two step-1,221 |
| Held-out artifacts | 109,967,297 bytes control; 117,126,738 bytes treatment |
| Free `/orange/thai` at close audit | approximately 204 GiB; inode and user quota not limiting |

Failures were engineering-only and not converted into scientific evidence:

- `50270457`: diagnostic `.retain_grad()` instrumentation bug; corrected job `50270783` completed.
- `50277092`: fail-closed checkpoint-retention configuration rejected the first preflight; corrected array `50278599` completed. No scientific pilot ran under the invalid configuration.

No existing checkpoint or historical scientific artifact was deleted.

## J. Artifact locations

- Machine-readable summary: [root_cause_summary.json](analysis/aspd_v3_gate_collapse_recovery_v1/root_cause_summary.json)
- Preregistered thresholds: [preregistered_criteria.json](analysis/aspd_v3_gate_collapse_recovery_v1/preregistered_criteria.json)
- Comparison implementation: [gate_collapse_comparison.py](../../aspd/multimodel/v3/gate_collapse_comparison.py)
- Fixed-anchor diagnostic: `/orange/thai/vinhnguyen1/aspd_v3/gate_collapse_diagnostic_v1/job_50270783`
- Control held-out: `/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/full_only_control/evaluation/job_50280055_0/heldout`
- Treatment held-out: `/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/per_anchor_zscore/evaluation/job_50280055_1/heldout`

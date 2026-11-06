# ASPD v3 controlled 10M gate-retrain diagnostic

Status: **CONTROLLED_DIAGNOSTIC_COMPLETE — ENGINEERING PASS, SCIENTIFICALLY NOT READY**

This read-only comparison tests whether the shared PRE/MID gate is the primary cause of
Gemma's poor activation reconstruction in the completed 500M ASPD v3 run. It does not
change or supersede the immutable 500M checkpoint.

## Provenance and integrity

- Source checkpoint: `step-000061036`, 500,006,912 accepted aligned-anchor exposures.
- Source checkpoint manifest SHA-256:
  `e45559e8b704b6516fbda62664b66bbee174379842736e028c815f310f7185e0`.
- Controlled implementation/training commit:
  `99ca15e2675f8703e0a2ddec2030c41c6943d669`.
- Evaluation commit: `3327de6f4867a02980a695a18d6fe07d4e2a7008`.
- Shared-control training job `49867679`: `COMPLETED`, exit `0:0`, 1,221 steps.
- Gemma-specific training job `49867680`: `COMPLETED`, exit `0:0`, 1,221 steps.
- Shared-control evaluation job `49897947`: `COMPLETED`, exit `0:0`.
- Gemma-specific evaluation job `49897948`: `COMPLETED`, exit `0:0`.
- Each training arm used exactly 10,002,432 new accepted anchors. All 1,221
  `batch_sha256` values and all alignment-observation payload hashes match across arms.
- Both held-out evaluations used the same 100,352 aligned validation anchors from the
  same 324 source documents. Their validation deduplication identities match the
  corrected original-checkpoint evaluation exactly.
- Both E0 restores passed deterministic repeat, source-weight immutability, checkpoint
  checksum, and intended shared-encoder topology checks.
- Rank-one matrix factors were frozen; both arms used fresh Adam state for the approved
  activation-only diagnostic.

## Held-out activation reconstruction

Full-reconstruction FVU (lower is better):

| Model/site | Original 500M | Shared continuation | Gemma-specific | Treatment - control | Relative treatment change |
|---|---:|---:|---:|---:|---:|
| Qwen PRE | 0.018660 | 0.018635 | 0.017484 | -0.001151 | -6.18% |
| Qwen MID | 0.022293 | 0.022273 | 0.021085 | -0.001188 | -5.33% |
| Gemma PRE | 0.410661 | 0.409961 | 0.409645 | -0.000317 | -0.08% |
| Gemma MID | 0.430697 | 0.429627 | 0.429008 | -0.000618 | -0.14% |
| SmolLM2 PRE | 0.007227 | 0.007190 | 0.007005 | -0.000185 | -2.57% |
| SmolLM2 MID | 0.008449 | 0.008436 | 0.008229 | -0.000207 | -2.46% |

The Gemma-specific branch improves Gemma only marginally. The effect is much smaller
than the persistent approximately 0.39--0.42 absolute FVU gap between Gemma and the
other models. It is also small relative to the per-document uncertainty visible in the
layer-level cluster-bootstrap intervals. Shared-gate interference therefore cannot
explain the Gemma failure.

The larger Qwen/SmolLM2 activation improvement in the treatment indicates an asymmetric
effect: excluding the Gemma activation gradient helps the Qwen/Smol shared branch more
than excluding Qwen/Smol gradients helps Gemma. This is not evidence that Gemma's own
poor reconstruction is caused by the shared gate.

## Dictionary utilization

| Arm/site | Fired features | Effective size (entropy) | Top-10% firing share |
|---|---:|---:|---:|
| Original PRE | 2,504 | 691.35 | 88.13% |
| Shared PRE | 2,535 | 700.96 | 87.91% |
| Gemma-specific PRE | 2,535 | 697.61 | 87.96% |
| Original MID | 2,400 | 671.48 | 89.05% |
| Shared MID | 2,418 | 678.67 | 88.88% |
| Gemma-specific MID | 2,344 | 658.37 | 89.60% |

The treatment does not repair dictionary concentration. Gemma-specific MID becomes
slightly more concentrated and uses fewer features.

### Matryoshka band concentration

The held-out feature rows were also aggregated over the five configured Matryoshka
bands. The analysis is reproducible with
`aspd.multimodel.cli.analyze_aspd_v3_feature_bands`; every input CSV is checksummed by
the output. On the original checkpoint, the shared-gate firing distribution is:

| Feature band | Dictionary share | Prefix losses containing the band | PRE firing share | MID firing share |
|---|---:|---:|---:|---:|
| 0--511 | 6.25% | 5 | 54.00% | 53.20% |
| 512--1023 | 6.25% | 4 | 14.07% | 15.69% |
| 1024--2047 | 12.50% | 3 | 12.22% | 13.38% |
| 2048--4095 | 25.00% | 2 | 11.97% | 10.49% |
| 4096--8191 | 50.00% | 1 | 7.73% | 7.24% |

The first 512 features therefore fire at 8.5--8.6 times the density expected under a
uniform feature distribution. They also account for 52--64% of the summed matrix
contribution L2 magnitude, depending on model and site. Conversely, the final half of
the dictionary receives only 6.9--9.2% of contribution magnitude despite containing
4,096 features. The shared continuation and Gemma-specific treatment preserve the same
pattern.

This alignment is structurally meaningful: equal weighting of the five cumulative
prefix reconstructions gives the first band five direct activation-loss opportunities
and the final band one. It is strong evidence that the current objective is associated
with the observed capacity concentration. It is not, by itself, a causal proof; a
band-balanced or rotating-prefix controlled pilot is still required.

## Internal reconstruction boundary

The matrix factors were frozen while the activation gate moved. Aggregate internal FVU
does not improve for Gemma (`0.36856` shared versus `0.36854` treatment), and worsens
for Qwen/SmolLM2 under the treatment. These checkpoints are diagnostic branches, not
replacement decompositions suitable for mechanism taxonomy.

## Why the completed 500M training is not scientifically acceptable

The run is complete and reproducible as an engineering artifact. The problem is the
scientific representation learned by that run:

1. **Gemma reconstruction is genuinely weak on held-out documents.** Full-reconstruction
   FVU is `0.4107` PRE and `0.4307` MID, corresponding to only `0.5893` and `0.5693`
   held-out R-squared. Qwen is at `0.9813/0.9777` and SmolLM2 at `0.9928/0.9916`.
   This is not just the six-prefix Matryoshka training metric being mislabeled as FVU.
2. **The error is broad rather than localized.** All eight Gemma layers are poor at both
   PRE and MID. A single hook, layer, or matrix-family anomaly cannot explain the result.
3. **The nominal 8,192-feature capacity is misleading.** Effective firing entropy is
   only about 660--700 features, the top 10% receives about 88--90% of firing events,
   and the first 512 Matryoshka features receive more than half of all fires.
4. **The shared coefficient has a scale/gauge incompatibility with the decoders.** Each
   model/layer decoder direction is projected to unit norm after every update. For one
   active component, the same shared coefficient therefore writes the same component
   L2 magnitude into Qwen, Gemma, and SmolLM2, although held-out target RMS differs
   sharply: approximately `17`, `3.6--3.8`, and `27`, respectively. Directional
   cancellation can differ, but the model-specific decoder cannot independently learn
   the component scale that this seven-fold cross-model range suggests it needs.
5. **The encoder receives differently scaled and differently compressed inputs.** The
   learned packer applies a raw linear projection with no explicit per-model/layer RMS
   normalization. Gemma concatenates `8 x 2304` channels into the same 4,096-wide packed
   representation used for `8 x 2048` Qwen and SmolLM2 inputs. Pre-normalized transformer
   sublayers do not remove scale from the residual stream. This is a plausible additional
   bottleneck, not a proven sole cause.
6. **More training with the same gate is not the answer.** Ten million matched additional
   exposures improve Gemma by only `0.00032` PRE and `0.00062` MID relative to the shared
   continuation. The treatment also fails to improve matrix reconstruction and does not
   repair dictionary utilization.

There is also evidence of a common absolute reconstruction floor. The held-out centered
target sum of squares is about 16--18 billion for Gemma, 458--461 billion for Qwen, and
1.20 trillion for SmolLM2, while full-reconstruction SSE remains in the same rough
6.6--10.3 billion range. Gemma's smaller target variance therefore turns a similar
absolute error floor into a much larger FVU. That observation does not make the metric
unfair: it means the representation fails to resolve Gemma variations at the scale that
matters within Gemma.

## Root-cause assessment

| Hypothesis | Assessment | Evidence |
|---|---|---|
| Reporting/normalization artifact | Not supported | Corrected held-out full FVU remains approximately 0.41/0.43 for Gemma. |
| Shared-gate interference as the main Gemma cause | Not supported | Model-specific gating changes Gemma FVU by only 0.08--0.14%. |
| Gemma gradient starvation or severe conflict | Not supported | Prior real-checkpoint probes found non-small Gemma gradients and no severe pairwise conflict. |
| Single bad layer or MID-hook mismatch | Not supported | Gemma is uniformly poor across all eight PRE and MID layers. |
| Cross-model scale/gauge mismatch | Supported as a leading mechanism, not yet isolated | Gemma target RMS is about 3.6--3.8 versus about 17 for Qwen and 27 for SmolLM2; shared code magnitudes combine with independently unit-normalized decoder directions. |
| Effective-capacity loss from dictionary concentration | Supported | Effective size is only about 660--700 of 8,192; the first 6.25% of features receives about 53--54% of fires while the last 50% receives about 7%. |
| Equal-weight cumulative Matryoshka prefixes as a cause of band concentration | Partially supported | The measured 5:1 prefix-loss opportunity asymmetry matches the firing gradient, but no objective ablation has yet established causality. |
| Raw packer scaling and Gemma's wider input compression | Partially supported | Inputs lack explicit layerwise RMS normalization and Gemma compresses 18,432 input channels into the same 4,096 packed channels used for 16,384-channel peers. |
| Gemma-specific intrinsic representation difficulty | Partially supported | The remaining directional error is broad across layers and survives a model-specific gate, but no capacity/normalization ablation has isolated it. |

## Improvement path and smallest next experiments

Do **not** repeat the 500M run with the same recipe, and do **not** launch a full
Gemma-specific-gate retrain. The controlled result is too small and does not repair
dictionary health.

The next work should be staged so that each result distinguishes one failure mechanism:

1. **Read-only calibration and K sweep.** On source-document-disjoint calibration and
   evaluation halves, fit only per-model/site/layer scalar gains to frozen predictions.
   Separately re-evaluate frozen dense scores at `K=16,32,64,128`, checking exact K=32
   parity. A large gain-only improvement implicates scale/gauge; a large K response
   implicates sparse capacity. These are diagnostics, not replacement metrics.
2. **Normalized-representation short pilot.** Freeze per-model/site/layer centering and
   RMS statistics from training-only data, normalize both packer inputs and activation
   targets, keep decoder directions unit norm in normalized coordinates, and unscale
   only for reported reconstruction. This preserves an explicit gauge instead of simply
   removing decoder normalization.
3. **Band-balanced Matryoshka pilot.** Compare the current cumulative-prefix objective
   with a preregistered rotating feature permutation or another objective that gives
   features equal expected prefix inclusion. Do not silently set PRE or prefix terms to
   zero. Retain `C=8192` and `K=32` for this causal objective comparison.
4. **Only then test capacity.** If normalization does not repair Gemma and the frozen K
   sweep shows a strong response, test a short `K` ablation. Increasing `C` is not
   justified while the effective dictionary remains below 9% of nominal size.

Before any full retrain, the selected short pilot must demonstrate all of the following
on the same held-out protocol:

- a material Gemma PRE and MID improvement beyond the cluster-bootstrap uncertainty;
- no material Qwen/SmolLM2 activation degradation;
- no internal-reconstruction regression from changed support;
- improved effective dictionary size and meaningful use of later feature bands; and
- deterministic restore, exposure accounting, and source-weight immutability.

Until such a pilot passes, shared/exclusive taxonomy and mechanism interpretation should
remain blocked. The current checkpoint is useful for diagnosing the training system, but
not a trustworthy basis for claiming cross-model shared or exclusive mechanisms.

# ASPD v3 controlled 10M gate-retrain diagnostic

Status: **CONTROLLED_DIAGNOSTIC_COMPLETE — NO FULL RETRAIN WITH THE SAME RECIPE**

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

## Internal reconstruction boundary

The matrix factors were frozen while the activation gate moved. Aggregate internal FVU
does not improve for Gemma (`0.36856` shared versus `0.36854` treatment), and worsens
for Qwen/SmolLM2 under the treatment. These checkpoints are diagnostic branches, not
replacement decompositions suitable for mechanism taxonomy.

## Root-cause assessment

| Hypothesis | Assessment | Evidence |
|---|---|---|
| Reporting/normalization artifact | Not supported | Corrected held-out full FVU remains approximately 0.41/0.43 for Gemma. |
| Shared-gate interference as the main Gemma cause | Not supported | Model-specific gating changes Gemma FVU by only 0.08--0.14%. |
| Gemma gradient starvation or severe conflict | Not supported | Prior real-checkpoint probes found non-small Gemma gradients and no severe pairwise conflict. |
| Single bad layer or MID-hook mismatch | Not supported | Gemma is uniformly poor across all eight PRE and MID layers. |
| Cross-model scale/gauge mismatch | Supported as a leading mechanism, not yet isolated | Gemma target RMS is about 3.6--3.8 versus about 17 for Qwen and 27 for SmolLM2; shared code magnitudes combine with independently unit-normalized decoder directions. |
| Effective-capacity loss from dictionary concentration | Supported | Effective size is only about 660--700 of 8,192 and the top 10% receives about 88--90% of firing events. |
| Gemma-specific intrinsic representation difficulty | Partially supported | The remaining directional error is broad across layers and survives a model-specific gate, but no capacity/normalization ablation has isolated it. |

## Decision and smallest next experiment

Do **not** repeat the 500M run with the same recipe, and do **not** launch a full
Gemma-specific-gate retrain. The controlled result is too small and does not repair
dictionary health.

Before any full retrain, run a short, separately versioned pilot that tests:

1. Frozen per-model, per-site, per-layer centering and RMS scaling before the hierarchical
   encoder and activation reconstruction target.
2. A gauge-preserving model/layer decoder gain alternative, compared against the frozen
   RMS scaling rather than simply removing decoder normalization.
3. A diagnostic sparsity/capacity sweep and a band-balanced Matryoshka/dead-feature
   revival variant to determine whether the approximately 670-feature effective
   dictionary is the limiting capacity.

Only a variant that materially lowers held-out Gemma FVU at both PRE and MID while
preserving Qwen/SmolLM2 activation and internal reconstruction should advance to a
larger retrain. Increasing `C` alone is not justified while most of the current
dictionary is unused.

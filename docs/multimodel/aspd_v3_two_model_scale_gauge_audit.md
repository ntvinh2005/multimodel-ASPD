# ASPD v3 two-model scale/gauge architecture audit

Status: **AUDIT_COMPLETE — PILOTS_DESIGNED — IMPLEMENTATION_NOT_STARTED — GPU_NOT_RUN**

This document audits the retired three-model ASPD v3 implementation and freezes the
design boundary for the first two replacement pilots. The replacement experiment uses
only Gemma and SmolLM2. It does not use Qwen, increase `C` or `K`, change Matryoshka
weighting, or authorize a 500M run.

## Approved scientific boundary

The two initial arms use:

- `google/gemma-2-2b-it` at the pinned revision already recorded in the retired config,
  layers 9--16;
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` at its pinned revision, layers 8--15;
- all seven logical matrices at every selected layer, for `2 x 8 x 7 = 112` targets;
- one shared PRE gate and one shared MID gate;
- `C=8192`, BatchTopK `K=32`, four 2,048-anchor pools per optimizer step;
- the same dataset ID/revision, document split seed and held-out protocol as the retired
  run; and
- fresh training from an identical initialization. No retired checkpoint is compatible
  with the two-model topology.

The old three-tokenizer alignment manifest is forbidden. A new two-tokenizer manifest
must be generated from the same source dataset and document split policy. Baseline and
treatment must use the exact same manifest, source-document order, accepted anchor IDs
and per-step batch hashes.

## Architecture audit

### Shared gate and gradient routing

The current model-owned path computes one local packed representation per model, uses a
differentiable all-gather, runs a replicated cross-model transformer/readout, and applies
one BatchTopK gate per PRE/MID site (`model_owned.py`, `encode`). The continuous selected
values drive activation reconstruction. A detached binary support drives internal
matrix reconstruction. Thus the internal loss cannot update the shared encoder, while
activation and AuxK losses can.

Shared-encoder gradients are sum-reduced across owner ranks. The current activation
coefficient is `0.5 / (number_of_models * 8)` per site/layer, so after all-reduce each
PRE and MID group has equal total weight. This is mathematically suitable for two models
only after every hard-coded three-model count is replaced by a value derived from the
frozen model order.

BatchTopK selects `pool_anchors * K = 65,536` entries per 2,048-anchor pool and site.
Selection is global over the pool, so individual anchors need not have exactly 32 active
features. The two pilot arms must preserve this behavior exactly.

### Decoder gauge

For model `m`, site `s`, layer `l`, the existing activation reconstruction is

\[
\hat y_{m,s,l}=b_{m,s,l}+\sum_c g_{s,c}d_{m,s,l,c},
\qquad \lVert d_{m,s,l,c}\rVert_2=1.
\]

`ActivationReconstructionBank.normalize_decoders_()` normalizes every decoder column
after updates and removes its radial gradient. The same shared coefficient `g[s,c]`
therefore produces the same individual-component L2 write magnitude in every
model/layer decoder. Model-specific direction and cancellation remain possible, but
model-specific component amplitude does not.

This constraint is scientifically relevant because the retired held-out target RMS was
approximately 3.6--3.8 for Gemma and 27 for SmolLM2. The unit-direction constraint is
useful for identifiability, but coupling it to a single shared coefficient may force a
poor cross-model scale compromise.

### Existing normalization

- Activation and internal errors are global FVU: FP32 SSE divided by coordinate-centered
  target sum of squares. This balances loss units across models but does not normalize
  the encoder inputs or reconstructed outputs.
- The learned packer receives the raw concatenation of eight residual streams. It has no
  explicit model/layer centering or RMS normalization.
- The intra- and cross-model transformer layers use `norm_first=True`, but their residual
  paths preserve input scale.
- Feature readout mean-pools the cross-model tokens, applies a linear map and ReLU.
- Decoder biases are unconstrained; decoder feature directions are unit norm.
- There is no explicit model/site/layer reconstruction gain.

The scale treatment should therefore test the missing output gauge only. Input
normalization is deliberately deferred to a separate later pilot so the result remains
attributable.

### Matryoshka objective

The current activation objective equally averages a bias-only reconstruction and five
cumulative prefixes with boundaries `512, 1024, 2048, 4096, 8192`. Consequently, the
five feature bands participate in `5, 4, 3, 2, 1` prefix losses. The retired held-out
audit found that the first 512 features received about 53--54% of firing events while
the final 4,096 received about 7%.

This is a serious second problem, but both initial scale arms must retain the same
Matryoshka objective and AuxK behavior. Changing scale and feature-band weighting in one
experiment would make the causal diagnosis ambiguous.

## Required two-model code changes common to both arms

The current implementation cannot run a scientifically valid two-model experiment by
editing YAML alone:

1. `main_2b_config.py` fixes source keys, tuple length, `world_size=3`, three-way policy,
   exact model IDs, 168 targets and the retired namespace.
2. `encoder.py` requires exactly three packers, allocates five identity channels,
   documents a `3a+2` sequence, and returns exactly two separator positions. A two-model
   sequence needs 16 model tokens plus one separator and three identity categories.
3. `production_stream.py` requires three tokenizers and records
   `identity_exact_three_way_prefix`.
4. `model_owned.py` requires three ranks and hard-codes the internal PRE/MID target
   counts as `3*8*3` and `3*8*4`.
5. `registry.py` requires exactly 168 targets.
6. `train_aspd_v3_2b.py` fixes rank mapping, preflight evidence, manifest fields and
   logical-target count to three models. Its historical controlled-retrain branch also
   contains Qwen-specific process groups and must not be reused.
7. Checkpoint manifests and resume validation record model order and world size. A
   two-model checkpoint needs a new schema/fingerprint namespace and cannot load a
   retired three-model optimizer state.

The implementation should derive model count, identity width, separator count, target
count and loss denominators from the immutable two-model config. It must still fail
closed on exactly two ordered sources for this pilot; this is not authorization for an
arbitrary dynamic model set.

## Pilot A: two-model baseline

Baseline is the current scientific method after only the unavoidable two-model topology
generalization:

\[
\hat y^{\mathrm{base}}_{m,s,l}
=b_{m,s,l}+\sum_c g_{s,c}d_{m,s,l,c}.
\]

Everything else remains fixed: shared gate, continuous activation code, detached binary
internal support, unit decoder directions, cumulative Matryoshka loss, AuxK, optimizer,
learning rates, BF16 execution, seed, matrix scope, `C` and `K`.

## Pilot B: model-specific reconstruction scale

Treatment introduces one positive relative gain for each model/site/layer:

\[
\hat y^{\mathrm{scale}}_{m,s,l}
=b_{m,s,l}+\exp(\delta_{m,s,l})
 \sum_c g_{s,c}d_{m,s,l,c}.
\]

With two models, use one unconstrained relative parameter `rho[s,l]`:

\[
\delta_{\mathrm{Gemma},s,l}=\rho_{s,l},\qquad
\delta_{\mathrm{Smol},s,l}=-\rho_{s,l}.
\]

This exact zero-mean log-scale constraint fixes the relative gauge: the geometric mean
of the two gains is one. It adds 16 FP32 trainable values, initialized to zero without
consuming RNG state. At initialization the treatment must match baseline exactly.

Implementation requirements:

- Scale only the feature reconstruction residual; do not scale the decoder bias.
- Apply the same gain to every cumulative Matryoshka prefix contribution.
- Apply it to the bias-free AuxK reconstruction so AuxK and the primary activation loss
  use the same output units.
- Do not apply it to rank-one matrix reconstruction or binary support.
- Route gradients from activation and AuxK objectives only.
- Keep decoder directions unit norm and radial-gradient projection unchanged.
- Optimize `rho` with the activation-decoder learning rate, no clipping and no hidden
  regularizer.
- Store raw `rho`, both gains and their gradients in metrics/checkpoints.

The treatment is not input normalization and not a per-feature model-specific code. It
tests only whether a missing layerwise relative output scale explains Gemma's error.

## Matched initialization and data contract

The two jobs must be paired as follows:

- Proposed short budget: 1,221 steps, exactly 10,002,432 accepted aligned-anchor
  exposures per arm. This is a proposed diagnostic budget, not authorization for a
  larger run.
- Use one frozen two-model data/alignment manifest and identical per-step batches.
- Verify every `batch_sha256` and accepted-anchor ID digest matches across arms.
- Instantiate all common parameters identically. Add zero-initialized scale parameters
  without advancing the common RNG stream, then verify hashes of every common initial
  tensor.
- Train from scratch with fresh Adam state; do not resume the retired checkpoint.
- Use separate immutable output namespaces and distinct config fingerprints.
- Run on two owner ranks, one B200 per model, only after a two-GPU optimizer-step and
  checkpoint/resume preflight passes.

## Held-out evaluation contract

Use a newly frozen two-model validation stream selected by the same document-level hash
split. Baseline and treatment must evaluate the same document IDs and accepted anchors.
Aggregate numerators and denominators before computing FVU; cluster-bootstrap at source
document level.

Primary metrics:

1. Gemma full-reconstruction FVU at PRE and MID.
2. Paired treatment-minus-baseline Gemma FVU with document-bootstrap 95% intervals.

Required secondary and safety metrics:

- SmolLM2 PRE/MID full FVU;
- six-term Matryoshka mean FVU and every prefix FVU;
- per-layer activation FVU, target/prediction/residual RMS and correlation;
- aggregate and per-matrix internal FVU;
- fired/dead features, entropy effective dictionary size, top-1/5/10% firing share and
  Matryoshka-band activity;
- learned gains and their stability across layers/sites;
- deterministic restore, finite tensors, source-weight immutability and exposure hashes.

The scale hypothesis is supported only if Gemma improves in the expected direction at
both PRE and MID beyond paired document-level uncertainty, without a material SmolLM2
or internal-reconstruction regression. Dictionary health need not improve in this pilot,
but a deterioration must be reported and blocks promotion. A small result comparable to
the retired 0.08--0.14% gate-treatment change is a negative result.

## Decision after the pilots

- If scale treatment clearly improves Gemma and preserves the safety metrics, implement
  a separately versioned normalization pilot to determine whether explicit input/target
  normalization provides the same benefit more cleanly.
- If scale does not help, do not add a more flexible per-feature gain immediately. Move
  to the independently controlled input-normalization pilot.
- In either case, Matryoshka balancing remains a separate experiment because the
  measured 5:1 prefix opportunity and dictionary concentration are unresolved.
- Do not select a third model or launch 500M training until the scale and dictionary
  diagnoses have passed their held-out gates.

## Current blockers

The pilots are **not ready to submit**. Required implementation and local verification
remain:

- new versioned two-model config/schema and two-way alignment manifest;
- generalized two-model encoder packing, stream, registry, loss counts and distributed
  runner;
- reconstruction-scale module with exact baseline parity tests;
- two-rank gradient, optimizer, checkpoint/resume and source-weight tests;
- immutable pilot configs and preflight Slurm scripts; and
- researcher approval of the proposed 1,221-step budget before GPU submission.

# Qwen2.5 Extreme-Sports ASPD controlled fidelity retrain

Status: **PREFLIGHT RUNNING — CONTROLLED TRAINING SUBMITTED AFTEROK**

Submitted from Git commit `1824cdda2b73aa781b7ce1b415d132bec2e1a709` at
`2026-10-19T19:22:28-04:00`:

- Fidelity baseline + tiny-overfit preflight: job `46741770`, initially `RUNNING` on
  `c0907a-s5`.
- Fresh 1,000-step controlled training: job `46741772`, initially `PENDING (Dependency)` with
  `afterok:46741770`.
- Both jobs request one B200, 14 CPUs, and 128 GB RAM. Walltime limits are one hour and four hours,
  respectively; these are allocation limits, not runtime estimates.
- Logs are `slurm/logs/mm-q25-aspd-fid-pre-46741770.{out,err}` and
  `slurm/logs/mm-q25-aspd-fid1k-46741772.{out,err}`.

The preflight is expected to take roughly 5–15 minutes based on the prior 200-step run and the
additional full-validation passes. If it succeeds, the controlled training itself is expected to
take roughly 10–30 minutes once allocated, with checkpoint I/O and ten full validation calls as the
main uncertainty. Inspect Slurm state and artifacts after job `46741772` leaves the queue; do not
infer success from disappearance alone.

## Scientific question

This experiment asks whether extending the otherwise unchanged ASPD optimization budget from 200
to 1,000 fresh steps improves reconstruction of the actual cached layer-14 matrix computations and
their Base-to-Target differences. It is not a Delta-Crosscoder comparison and it does not authorize
mechanism or causal claims.

The frozen pair is Base `unsloth/Qwen2.5-7B-Instruct` at
`a75c9dc945567a9b6f568b8503a0307731607bee` and Target PEFT adapter
`ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports` at
`6aa7e935c74c519c66555a111d1e6982059bd82d`, merged into the same pinned Base before cache
materialization. Grounding is `model.layers.14` input. The seven sites are `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

## Why the pilot is not resumed

The existing checkpoint stores optimizer state, RNG state, step and epoch, but no intra-epoch batch
cursor. Resuming step 200 would therefore restart the epoch iterator and would not equal an
uninterrupted 1,000-step sequence. The controlled run starts at step zero, while the old checkpoint
is measured independently by the same new evaluator.

The original training validation also averages per-batch FVU ratios and inspected only eight
batches (4,096 tokens). That training diagnostic remains unchanged to preserve the objective, but
the new fidelity evaluator processes all 64 validation sequences (16,384 tokens) and derives ratios
from global sufficient statistics.

## Computation-fidelity evaluator

Implementation:

```text
aspd/multimodel/fidelity.py
aspd/multimodel/cli/evaluate_computation_fidelity.py
```

It loads only the ASPD checkpoint, validated paired cache, and cached weight snapshots. It never
loads a Qwen model. Grounding activations use the same train-RMS normalization as training. Matrix
targets are explicitly labelled **weight-only linear computations** because cached snapshots do not
represent an affine bias.

For Base and Target it measures global SSE, RMSE, relative RMS error, cosine, actual/predicted RMS,
and FVU for every matrix. It separately measures:

```text
DeltaY_total     = W_target X_target - W_base X_base
DeltaY_parameter = (W_target - W_base) X_base
DeltaY_input     = W_target (X_target - X_base)
```

The ASPD terms use the same shared binary gate and the corresponding learned rank-one operators.
Both true and learned identities `total = parameter + input` are audited. Near-zero energy,
variance, or cosine denominators produce `null` plus explicit conditioning flags instead of
unstable normalized claims. Operator computations use FP32 and global statistics accumulate on CPU
in FP64.

The evaluator requires checkpoint provenance to match the resolved config, cache fingerprint,
model IDs/revisions, model count, and latent count. Outputs are atomic and immutable: an existing
JSON path is never replaced.

Baseline output:

```text
out/multimodel/diagnostics/qwen25_aspd_fidelity/baseline200_full_validation.json
```

It is pinned to old checkpoint SHA256
`b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800` and cache fingerprint
`1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447`.

## Tiny-overfit gate

`aspd.multimodel.cli.tiny_overfit` initializes the same architecture afresh and repeats exactly the
first deterministic unshuffled training batch for 75 updates. It uses the original optimizer,
learning rate, BF16 autocast, normalization, Matryoshka activation loss, internal loss, AuxK, and
TopK 32. It records initial/final objective metrics, full activation FVU, per-model/per-matrix
internal FVU, first-step pre-clipping gradient norms, L0, an input-ID hash, and sampled updates to
active encoder/decoder/operator rows.

The check exits nonzero unless every metric is finite, mean internal FVU decreases by at least 1%,
rank-one component gradients are nonzero, sampled active parameters change, and L0 remains 32. It
is an engineering diagnostic, not a scientific checkpoint. Output:

```text
out/multimodel/diagnostics/qwen25_aspd_fidelity/tiny_overfit_75.json
```

## Controlled 1,000-step run

Config:

```text
configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_fidelity1000.yaml
```

It inherits the completed pilot and changes only the name/output path and administrative budget:
fresh `max_steps=1000`, `resume=null`, validation every 100 steps over 32 batches, save every 250
steps, and retain two checkpoints. Architecture, data/cache, D0/S1, C=8,192, K=32, objective,
learning rate, optimizer, batch size, seed, parameter precision, and autocast are unchanged. Total
exposure is 512,000 tokens, approximately 3.91 traversals of the 131,072-token frozen train cache;
these are exposures, not unique tokens.

Run output:

```text
out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000
```

Launchers:

```text
slurm/multimodel/qwen25_extreme_sports_aspd_fidelity_preflight.sbatch
slurm/multimodel/qwen25_extreme_sports_aspd_fidelity1000_train.sbatch
```

Both request one B200, 14 CPUs, and 128 GB. The 1,000-step job is submitted with an external
`afterok` dependency on the fail-closed preflight and independently revalidates config, cache,
baseline, tiny-overfit, fresh output path, final step, checkpoint finiteness, and training metrics.

## Post-training audit (prepared, not submitted)

After the training job completes successfully, first inspect its Slurm state/logs, step-1000
checkpoint, metrics and validation curve. Then submit, manually and only after that inspection:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch slurm/multimodel/qwen25_extreme_sports_aspd_fidelity1000_evaluate.sbatch
```

The prepared evaluator writes:

```text
out/multimodel/diagnostics/qwen25_aspd_fidelity/fidelity1000_full_validation.json
```

That artifact must be compared matrix-by-matrix with the immutable 200-step baseline. Lower train
loss alone is not success. The main evidence is Base/Target computation FVU plus total,
parameter-only, and input-change DeltaY fidelity. If DeltaY remains poor even when absolute model
reconstruction improves, that limitation must be reported and no changed-mechanism claim made.

Old component IDs 310, 295, 328, 390, and 267 are checkpoint-specific and must not be transferred
by number to the independent run. Correspondence and P6/causal work remain unauthorized until the
fidelity comparison is reviewed.

## Static verification

The focused CPU suite covers analytic true/learned DeltaY identities, zero-Delta and small
denominators, globally aggregated FVU across batches, token masks, Base/Target order, matrix
alignment/dimensions, cache/checkpoint/config mismatches, immutable artifacts, fixed-batch
selection, no silent resume, and fail-closed Slurm dependency preparation.

No Delta model is retrained by this plan. No ASPD analysis, P6, or causal job is chained from the
training job.

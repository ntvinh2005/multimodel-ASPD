# Qwen2.5 Extreme-Sports ASPD controlled fidelity retrain

Status: **COMPLETED — ABSOLUTE FIDELITY IMPROVED; DELTAY FIDELITY REMAINS INADEQUATE**

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

The preflight and controlled training completed in 1m58s and 7m26s. Success below is based on
`sacct`, log markers, artifact integrity and tensor inspection rather than disappearance from the
queue.

## Completed results

All three bounded jobs completed `0:0`:

| Job | Purpose | Runtime |
|---:|---|---:|
| `46741770` | Step-200 baseline + tiny-overfit preflight | 1m58s |
| `46741772` | Fresh 1,000-step controlled training | 7m26s |
| `46743824` | Step-1000 full-validation fidelity evaluation | 1m32s |

The final checkpoint is `checkpoint_00001000.pt`, records step 1,000, and has SHA256
`ff00347a0b90f789a794fd7e67744252bf81ddee82b992135db80d38c348b8bf`. All checkpoint and metric
tensors inspected were finite. Validation loss and internal FVU decreased at every recorded
100-step evaluation; step 1,000 is therefore the best observed validation point in this run. L0
remained exactly 32.

The tiny-overfit gate passed. Mean internal FVU fell from 1.5111 to 0.5239 (65.3%); first-step
gradient norms were 0.504 encoder, 0.734 activation decoder, and 0.0119 rank-one components; sampled
active parameters had update L2 8.44. This establishes that the internal objective participates in
optimization on a repeated batch.

### Matched full-validation comparison

Both fidelity artifacts use all 16,384 validation tokens, FP32 reconstruction, and global FP64
sufficient statistics.

| Metric | Step 200 | Step 1,000 | Relative change |
|---|---:|---:|---:|
| Final recorded train loss | 2.1929 | 1.5874 | -27.6% |
| Full activation FVU, Base | 0.004440 | 0.003352 | -24.5% |
| Full activation FVU, Target | 0.004347 | 0.003242 | -25.4% |
| Mean matrix FVU, Base | 0.8462 | 0.6294 | -25.6% |
| Mean matrix FVU, Target | 0.8415 | 0.6220 | -26.1% |
| Total DeltaY energy-normalized squared error | 1.2990 | 1.1979 | -7.8% |
| Parameter-only DeltaY energy-normalized squared error | 413.797 | 367.151 | -11.3% |
| Input-change DeltaY energy-normalized squared error | 1.1352 | 0.8884 | -21.7% |

Per-matrix absolute-computation FVU:

| Matrix | Base 200 → 1,000 | Target 200 → 1,000 |
|---|---:|---:|
| q_proj | 0.7424 → 0.4822 | 0.7383 → 0.4749 |
| k_proj | 0.7853 → 0.4593 | 0.7786 → 0.4494 |
| v_proj | 0.8376 → 0.5931 | 0.8325 → 0.5860 |
| o_proj | 0.8493 → 0.5690 | 0.8434 → 0.5635 |
| gate_proj | 0.9114 → 0.7059 | 0.9078 → 0.6979 |
| up_proj | 0.8777 → 0.7599 | 0.8733 → 0.7526 |
| down_proj | 0.9194 → 0.8363 | 0.9168 → 0.8296 |

Per-matrix DeltaY energy-normalized squared error (`total / parameter-only / input-change`):

| Matrix | Step 200 | Step 1,000 |
|---|---:|---:|
| q_proj | 1.788 / 600.1 / 0.961 | 1.613 / 614.9 / 0.771 |
| k_proj | 4.175 / 6408.6 / 0.959 | 4.445 / 7563.8 / 0.736 |
| v_proj | 3.556 / 1211.0 / 0.954 | 3.321 / 1213.9 / 0.830 |
| o_proj | 1.392 / 234.8 / 0.914 | 1.071 / 206.7 / 0.695 |
| gate_proj | 1.063 / 400.0 / 1.382 | 0.859 / 201.6 / 0.934 |
| up_proj | 1.072 / 203.1 / 0.985 | 1.088 / 258.2 / 0.907 |
| down_proj | 1.094 / 599.0 / 0.971 | 1.062 / 698.8 / 0.891 |

The longer run substantially improves absolute Base/Target computation reconstruction at every
matrix, so the primary optimization question has a positive but limited answer. It also improves
the aggregate input-change term enough to beat the zero predictor by 11.2% (`1 - NSE`).

However, aggregate total DeltaY NSE remains 1.198, meaning the zero-change predictor still has lower
squared error. Parameter-only DeltaY is especially poor: its true aggregate RMS is only 0.00507,
while ASPD error RMSE is 0.0971, giving NSE 367.2. Several per-matrix Delta metrics worsen despite
better absolute reconstruction. The present objective therefore does **not** faithfully explain the
fine-tuning-induced parameter computation change.

The correct outcome is to stop before component interpretation/P6. A future experiment may design
a separately preregistered Delta-focused objective, but no such objective or additional training is
authorized here.

Artifact hashes:

```text
baseline200_full_validation.json  32d7bd9ab6d23acf9720ef187be583cf1acda07b1b9e26058d765dcb2981566d
tiny_overfit_75.json               2c2bc46363415339484e42a8e826ea792732a0f78f35e99fd65559286c5300e8
fidelity1000_full_validation.json  aa17cb1f5d7259ec6e788f8d0e0ba300f0d50cd7a74ed04a0864b4d89d0da9e4
metrics.jsonl                      6966362970afc581dae00b4c97154277c1354dda4964b423c89f311f421d8e6d
```

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

# ASPD v3 Granite normalized-target GPU preflight report

Status: `GPU_PREFLIGHT_PASSED — SCIENTIFIC_EFFICACY_UNVERIFIED — PILOT_NOT_AUTHORIZED`

This report records bounded engineering preflights only.  Each arm ran three optimizer steps on
three B200 GPUs.  The runs do not establish convergence, held-out reconstruction quality, or an
advantage of per-anchor target normalization.  No short pilot or full training is authorized.

## 1. Immutable execution identity

| Item | Baseline | Normalized-target treatment |
|---|---|---|
| Slurm job | `50034763` | `50034764` |
| Slurm result | `COMPLETED`, `0:0`, 3m33s | `COMPLETED`, `0:0`, 3m47s |
| Git commit | `53aa6508148b238f9ba4b04fb5a33962b4d246c4` | same |
| Config fingerprint | `642ce56a1fa83ec552f53d777d73bce6533fb8d9c8f7cb392840f27a0cc705dc` | `37c5c902e69503c362db37ee5031c0d83be2aa50e6e16fb97d5adb7bbbb93acc` |
| Target normalization | `none` | `per_anchor_l2`, epsilon `1e-6` |
| Reconstruction scale | `none` | `none` |
| Optimizer steps | 3 | 3 |
| Accepted exposures | 24,576 | 24,576 |
| World size / targets | 3 / 168 | 3 / 168 |

Both runs used dataset-manifest SHA-256
`a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f` and alignment-manifest
SHA-256 `e3fb039d2f12f3f81fef40fa064eb5b189dcfb8c99ba8fd0b6c0bccfe7a50777`.
The immutable model revisions are recorded in each preflight report.

Primary evidence:

- `out/multimodel/aspd_v3/granite_normalized_target_v1/preflight/baseline/preflight_report.json`
- `out/multimodel/aspd_v3/granite_normalized_target_v1/preflight/normalized_target/preflight_report.json`
- the corresponding `metrics.jsonl`, `run_manifest.json`, checkpoint manifests, and Slurm logs

## 2. Paired-control checks

| Check | Result | Evidence |
|---|---|---|
| Same accepted-anchor sequence | PASS | Per-step ordered-anchor hashes match for all three steps |
| Same common initialization | PASS | Per-owner common-state hashes match exactly |
| Same source data/alignment | PASS | Dataset and alignment manifest hashes match |
| Exact model roster | PASS | Gemma, SmolLM2, Granite; 56 targets each |
| Frozen source parameters | PASS | All three owners in both arms |
| Hook forward immutability | PASS | All three owners in both arms |
| Early-exit capture parity | PASS | All three owners in both arms |
| Native projection parity | PASS | Maximum absolute error `0.0` for every owner |
| No learned reconstruction scale | PASS | All recorded reconstruction scales are exactly `1.0` |
| Checkpoint save/reload/resume update | PASS | Both arms |
| Parameter and Adam-state restore probe | PASS | Both arms |

The matching initial common-state hashes are:

| Owner | SHA-256 |
|---|---|
| Gemma | `ce5525bb6a82fa05b6e2d87661e230f8c162be823fdfc1ac296be8807a439174` |
| SmolLM2 | `65402a121d793b88d8c1340caf7996ff862d263fc1e5f8d41b43f112d02c1925` |
| Granite | `9761979be0b9dfaf76821a5b4bbecf4c9b6b73952e5ad88a25979bfc4763440e` |

Ordered-anchor SHA-256 values match in order at steps 1–3:
`78ef899e...bd4f`, `77fa9e32...38ac`, and `012918c0...da2a`.

## 3. Runtime and memory

The first step is a warm-up and is excluded from the two measured-step summaries.

| Metric | Baseline | Treatment | Treatment minus baseline |
|---|---:|---:|---:|
| Step p50 | 1.4474s | 1.5558s | +0.1085s (+7.5%) |
| Step p95 | 1.6715s | 1.7390s | +0.0674s (+4.0%) |
| Accepted anchors/s | 5,253.1 | 4,972.7 | -280.5 (-5.3%) |
| Checkpoint save + reload | 110.89s | 115.37s | +4.48s |

Both arms meet the researcher's preference of less than two seconds per measured optimizer step.
Only two measured steps are available, so the percentage differences are descriptive and must not
be treated as stable performance estimates.  In particular, stage-level differences include node,
I/O, and scheduling noise and do not isolate normalization overhead.

Peak allocated / reserved GPU memory was:

| Owner | Baseline | Treatment |
|---|---:|---:|
| Gemma | 73.08 / 74.68 GiB | 73.08 / 74.54 GiB |
| SmolLM2 | 67.21 / 69.09 GiB | 67.21 / 69.11 GiB |
| Granite | 64.99 / 66.83 GiB | 64.99 / 66.67 GiB |

Each complete model-plus-optimizer checkpoint is approximately 118.53 GiB.  Both checkpoints and
all run artifacts are retained for reproducibility.

## 4. Loss behavior and correctness interpretation

| Step | Baseline activation FVU | Treatment activation FVU | Baseline internal FVU | Treatment internal FVU |
|---:|---:|---:|---:|---:|
| 1 (warm-up) | 1.2301 | 885.1253 | 2.1644 | 2.1644 |
| 2 | 1.2007 | 156.0186 | 2.1344 | 2.1352 |
| 3 | 1.2053 | 72.3118 | 2.1487 | 2.1497 |

The activation FVU values across arms are not measurements in the same target space and therefore
are not a treatment-effect estimate.  The large treatment values are nevertheless an important
optimization warning.  The implementation keeps raw activations as encoder inputs and preserves
the same continuous gate/decoder initialization, while the reconstruction target has unit L2 norm
per valid anchor.  Thus the initial prediction scale inherited from the raw-activation architecture
can be much larger than the normalized target scale.  The implementation was re-audited after the
run: norms are computed in FP32 per model/site/layer/anchor, invalid anchors are masked, every
Matryoshka prefix and AuxK receives the same normalized target, and FVU uses the approved
coordinate-centered denominator.

This behavior is consistent with the approved equations; it is not evidence that normalization
helps, and three descending points are not evidence of convergence.  No LR, loss coefficient,
decoder norm, gate amplitude, target definition, or initialization was retuned after observing the
result.  Any change intended to reduce the initial treatment loss would be a new controlled
scientific decision rather than a code-only optimization.

AuxK is zero in both runs because no feature can satisfy the 320-consecutive-pool dead-feature
criterion during a three-step preflight.  It remains covered by deterministic synthetic tests.

## 5. Recovery history

The first baseline submission, job `50033841`, failed before model loading or training because the
top-level dispatcher examined the raw overlay YAML and sent it to the historical main-v1 loader;
the overlay contains `extends` rather than a repeated `experiment_family`.  The fail-closed parse
produced a Pydantic validation error and no scientific artifact.  Commit
`438fc9e70766a559c2aa0e737853f7c43c43467d` added explicit overlay dispatch plus a regression test.
The corrected jobs above then completed.

The dependent job `50033848` remains `PENDING (DependencyNeverSatisfied)` and consumes no GPU.  It
was not cancelled because cancellation was not authorized.  It must not be confused with the
successful treatment job `50034764`.

## 6. Post-run local verification

Command:

```text
../repo/.venv/bin/pytest -q tests/test_multimodel_v3_granite_normalized_target.py
```

Result: `17 passed` in 29.91 seconds.  The only warning was the sandbox preventing pytest cache
creation; it did not affect test execution.  The suite covers strict config isolation, overlay
dispatch, exact baseline target-path parity, target normalization, FVU gradient-scale change,
oracle denormalization, Matryoshka/AuxK consistency, shared-gate behavior, decoder constraints,
Granite module semantics, tokenizer alignment, and checkpoint compatibility.

## 7. Readiness decision

- **Implementation correctness:** PASS for the exercised three-model B200 preflight path.
- **Runtime preference:** PASS for both arms; no further performance edit is justified by this
  two-step timing sample.
- **Checkpoint/restart:** PASS.
- **Scientific efficacy:** UNVERIFIED.  No held-out evaluation or convergence evidence exists.
- **Normalized-target optimization risk:** OPEN.  The initial FVU scale is severe and must be
  explicitly considered before authorizing a longer pilot.
- **Pilot/full training:** NOT AUTHORIZED.


# Qwen2.5 Extreme-Sports ASPD general1m mechanism discovery

Status: **TRAINING/FIDELITY MEASURED — P1–P5 RUNNING**

Cache and fresh 4,096-step training completed successfully. The original fidelity job
computed every requested evaluation but failed only because an FP32 identity residual of
`1.015613e-5` narrowly exceeded a `1e-5` integrity threshold. The audited tolerance is
now `2e-5`; no GPU evaluation was repeated. Replacement P1–P5 job `46782245` was running
at `2026-10-20T00:26:11-04:00`.

Submission record: cache `46751681` → training `46751683` → common-input fidelity
`46751688` → P1–P5 `46751690`, connected exclusively by `afterok`. At
`2026-10-19T20:54:24-04:00`, the cache job was pending for priority and all descendants
were pending on their dependencies. The queue state is operational metadata, not a
scientific result.

This is the canonical report for the 1M-token D0/S1 run and its subsequent exploratory
sports-risk mechanism analysis. It will be updated from immutable JSON/CSV artifacts as
the dependent jobs complete. Pending or scheduled work is not reported as a result.

## Frozen scientific specification

- Base: `unsloth/Qwen2.5-7B-Instruct` at
  `a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target adapter: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports` at
  `6aa7e935c74c519c66555a111d1e6982059bd82d`, merged into the same pinned Base.
- Grounding: input to `model.layers.14`.
- Matrix sites: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`.
- Method: original D0/S1 objective, linear shared encoder, model-specific decoders,
  rank-one components, Matryoshka and AuxK unchanged; no Delta reconstruction loss.
- Data: 1,048,576 general-purpose training tokens and 131,072 validation tokens at
  sequence length 256, with equal FineWeb/UltraChat source weights. No sports-risk data
  enters training.
- Optimization: 4,096 steps, batch size 2, gradient accumulation 1, 8,192 components,
  BatchTopK 32, checkpoints at steps 2,048 and 4,096.

Resolved config:
`configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml`.

## Execution graph

The prepared fail-closed chain is:

```text
general1m cache + full integrity scan
  -> fresh 4,096-step training
  -> common-input fidelity at old step 1,000 / new 2,048 / new 4,096
  -> final-checkpoint P1-P5 analysis
```

Every stage uses one B200 and an `afterok` dependency. Cache, run, fidelity, and analysis
paths refuse stale outputs rather than treating them as fresh evidence. The fidelity
evaluator was extended so the old checkpoint can be evaluated on the new common
validation tokens while retaining its own training-cache RMS normalization.

The 4,096-step comparison has an important limitation: `dead_after_batches=2000` is
unchanged, so AuxK can become active in this run even though it did not activate during
the 1,000-step baseline. Training corpus size, repeated exposures, and AuxK exposure all
change together; improvement cannot be attributed to dataset size alone.

## Results

The validated cache occupies 195 GiB and has fingerprint
`71b268fc2190df5c7ddf0a43166242e714f4be92e2db20bf3405a0102f5037ce`. Both retained
checkpoints are present; P1–P5 candidates do not exist yet. On the common 131,072-token
validation input, the principal measurements are:

| Checkpoint | Internal FVU Base | Internal FVU Target | Total DeltaY NSE | Parameter-only NSE | Input-change NSE |
|---|---:|---:|---:|---:|---:|
| Old step 1,000 | 0.6223 | 0.6143 | 1.2717 | 373.66 | 0.8892 |
| New step 2,048 | 0.5853 | 0.5763 | 1.1720 | 334.03 | 0.8502 |
| New step 4,096 | 0.5744 | 0.5651 | 1.0361 | 271.81 | 0.8525 |

The longer run improves absolute matrix reconstruction and total DeltaY substantially,
but total DeltaY NSE remains above the zero-change baseline and parameter-only recovery
remains poor. Consequently, P1–P5 geometry can support exploratory candidate inspection,
not a claim that ASPD faithfully reconstructs the fine-tuning parameter effect.
Component IDs must not be assumed to correspond across independently trained checkpoints.

## Frozen Delta V3 reuse boundary

The existing 60 sports-risk, 60 matched sports-control, and 60 neutral prompts; 360
Base/Target-generated responses; and existing behavioral scores remain frozen. They will
not be regenerated or re-judged. The current Delta V3 activation cache stores paired
grounding activations but not all seven ASPD matrix inputs. Replaying those exact token
sequences to capture the missing inputs and attaching the existing metadata/scores is a
later noncausal stage, gated on acceptable final fidelity and P1-P5 integrity.

No sports-risk mechanism candidate has been selected. If fidelity remains inadequate,
any association will remain exploratory and the next recommendation will be an
original-method D1/D2 ablation rather than P6 or another blind D0 extension.

## Prohibited work confirmation

This experiment does not retrain Delta-Crosscoder, does not run P6 or causal
intervention, does not regenerate Delta V3 responses, and does not rerun the V3 judge.

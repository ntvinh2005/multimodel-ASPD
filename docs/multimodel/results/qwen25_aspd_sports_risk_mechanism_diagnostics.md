# Qwen2.5 ASPD sports-risk mechanism diagnostics

Status: **DIAGNOSTICS RUNNING — NO SPORTS-RISK MECHANISM CLAIM YET**

Last updated: `2026-10-20T01:41:45-04:00`.

This is the single research report for computation-fidelity diagnosis and exploratory
sports-risk mechanism discovery using the completed 1M-token D0/S1 ASPD run. It records
only measured results. The frozen V3 evaluation jobs are listed below, but their pending
outputs are not treated as findings.

## Frozen checkpoint and P1–P5 state

- General1m checkpoint: step 4,096, SHA-256
  `4539076e2af82bbee3c73c400f57fac539caa1c4cd7d676cad3af599978d2b61`.
- Original checkpoint: step 1,000, SHA-256
  `ff00347a0b90f789a794fd7e67744252bf81ddee82b992135db80d38c348b8bf`.
- General1m cache fingerprint:
  `71b268fc2190df5c7ddf0a43166242e714f4be92e2db20bf3405a0102f5037ce`.
- Replacement P1–P5 job `46782245` completed successfully in `00:02:04` with exit
  code `0:0`.
- P1–P5 manifest:
  `out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/analysis/analysis_manifest.json`.

The P1–P5 integrity totals are exact: 4,194,304 observed fires equals
131,072 validation tokens times BatchTopK 32. Only 914 of 8,192 components fired on this
validation set; 7,278 (88.84%) had zero observed firing. The top 100 components account
for 77.84% of fires and the inverse-Simpson effective component count is 103.09. This is
strong utilization concentration, but it does not by itself distinguish a K bottleneck
from representation/objective limitations.

At the registered 0.9 concentration threshold, the taxonomy contains 674
shared-activation/shared-mechanism components and 7,518 mixed-or-subset-mass components.
No concentrated activation/mechanism category is populated. Therefore no target- or
base-dominant component has yet passed the registered shortlist rule. IDs 1440, 958,
3355, 4523, and 2873 are shared/shared controls, not positive sports-risk candidates.

## General-validation matrix fidelity

The table below is the measured step-4,096 result on the common 131,072-token validation
set. `B/T` denotes Base/Target. Delta columns are normalized squared error (NSE), while
absolute reconstruction and output scales are RMS.

| Matrix | B/T FVU | B/T reconstruction RMSE | B/T actual-output RMS | Total DeltaY NSE | Parameter-only NSE | Input-change NSE |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 0.4262 / 0.4160 | 0.5720 / 0.5550 | 0.9349 / 0.9183 | 1.2444 | 370.60 | 0.7557 |
| k_proj | 0.3751 / 0.3669 | 0.7771 / 0.7548 | 1.3903 / 1.3709 | 4.4483 | 6439.37 | 0.7059 |
| v_proj | 0.5436 / 0.5306 | 0.4669 / 0.4501 | 0.6623 / 0.6474 | 2.3019 | 656.61 | 0.8178 |
| o_proj | 0.5246 / 0.5184 | 0.1515 / 0.1471 | 0.2194 / 0.2172 | 1.2393 | 246.14 | 0.6795 |
| gate_proj | 0.6353 / 0.6253 | 0.4295 / 0.4124 | 0.8521 / 0.8351 | 0.7863 | 167.38 | 0.8602 |
| up_proj | 0.7113 / 0.7017 | 0.4199 / 0.4024 | 0.5149 / 0.4987 | 0.8967 | 168.61 | 0.9002 |
| down_proj | 0.8044 / 0.7969 | 0.3149 / 0.2987 | 0.3565 / 0.3402 | 0.9983 | 634.68 | 0.8743 |

Relative to step 1,000, the largest absolute per-matrix FVU reductions are in `k_proj`,
`gate_proj`, and `v_proj`. Total DeltaY recovery improved especially for `v_proj`
(3.8344 to 2.3019) and `q_proj` (1.7918 to 1.2444), while `o_proj` worsened slightly
(1.1752 to 1.2393). `gate_proj`, `up_proj`, and `down_proj` now beat the zero-difference
baseline for total DeltaY, but this does not make parameter-only recovery reliable.
`k_proj` remains the clearest failure: despite the best individual-model FVU, its
parameter-only NSE is 6439.37 because the true update signal is extremely small relative
to absolute computation scale.

The mean measurements previously reported are Base/Target FVU 0.5744/0.5651, total
DeltaY NSE 1.0361, parameter-only NSE 271.81, and input-change NSE 0.8525.

The error-alignment diagnostic completed in job `46794331`. Base and Target errors are
strongly aligned rather than independent: per-matrix error cosine ranges from 0.8724
(`k_proj`) to 0.9740 (`gate_proj`), and cancellation gain ranges from 0.8720 to 0.9732.
Nevertheless, their absolute scale is too large relative to true DeltaY. The measured
delta-error/true-delta RMS ratios are 2.109 (`k_proj`), 1.517 (`v_proj`), 1.116
(`q_proj`), 1.113 (`o_proj`), 0.999 (`down_proj`), 0.947 (`up_proj`), and 0.887
(`gate_proj`). Thus error cancellation helps substantially but is insufficient; the
failure combines imperfect absolute reconstruction with a much smaller true difference
signal, and `k_proj` is the clearest outlier.

The precision audit independently formed each ideal FP32 LoRA update and reproduced the
effective cache delta exactly by BF16-quantizing `Base + ideal update`. This rules out an
incorrect merge implementation. BF16 representation is still material for the tiny
update: effective/ideal cosine ranges from 0.9126 to 0.9789, quantization-error norm is
20.8%–42.7% of ideal-update norm, and 23.4%–47.3% of ideal-nonzero entries become zero in
the effective BF16 difference. `down_proj` is most affected. These measurements explain
part of the weak parameter-only signal, while the ASPD cache remains correct because it
uses the actual effective weights rather than the ideal FP32 adapter update.

## Frozen V3 measurement boundary

The existing response artifact is reused unchanged: 180 prompts, 360 Base/Target
responses, and 101,564 exact replay tokens. It has fingerprint
`be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd`.
The existing paired-R cache has fingerprint
`aafd5d03fdd189a942fcbf18f3f4f68154997cb71ccc84fb44148d0efeab438e`.
Exactly 240 sports responses have frozen V3 scores; neutral labels are unavailable and
will not be imputed.

The measurement preserves the source shard order and uses batches of two sequences for
both checkpoints. Group metrics are accumulated from squared-error sufficient statistics
after one fixed BatchTopK encoding; subgroups are not independently re-encoded. The new
cache contains the seven missing matrix-input tensors and a corrected merged-Target R;
verified Base R is reused. It does not generate or judge text.

## Job history and active bounded jobs

General diagnostic job `46794331` completed in `00:07:59` with exit `0:0`. Its artifacts
are `step4096_error_alignment.json` and `lora_precision.json` under
`out/multimodel/diagnostics/qwen25_aspd_general1m/`.

The first V3 cache job `46794332` failed closed in `00:02:05`. Base replay passed, but
Target replay differed from the old frozen Target R by relative RMS 0.021145. The cause
was a real provenance mismatch: Delta V3 captured Target R through the unmerged PEFT
measurement path, whereas ASPD snapshots and trains against the effective BF16 merged
Target. Its dependent job `46794333` consequently became `DependencyNeverSatisfied` and
was canceled. The 17 GiB job-specific partial cache was removed; no completed artifact
was deleted.

Commit `d45342b` fixes this without relaxing the check or mixing computation graphs. The
corrected cache reuses exact frozen tokens and verified Base R, captures all seven X
tensors, and recaptures only Target R with the same merged effective Target used by ASPD.
All corrected jobs fail closed on stale or malformed outputs.

| Job | State at corrected submission audit | Resources / limit | Expected active runtime | Purpose |
|---|---|---|---|---|
| 46795283 | RUNNING on `c0909a-s5` | 1 B200, 8 CPU, 128 GB, 2:00 | about 4–10 min based on observed throughput | exact-token X capture plus merged-Target R correction |
| 46795285 | PENDING, `afterok:46795283` | 1 B200, 14 CPU, 128 GB, 3:00 | about 20–90 min after cache completion | fixed-batch old/new grouped V3 fidelity and per-sequence component summaries |

Expected outputs:

- `out/multimodel/diagnostics/qwen25_aspd_general1m/step4096_error_alignment.json`
- `out/multimodel/diagnostics/qwen25_aspd_general1m/lora_precision.json`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_matrix_cache_general1m/`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_mechanism_diagnostics/step1000_grouped_fidelity.json`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_mechanism_diagnostics/step4096_grouped_fidelity.json`
- corresponding step-1000 and step-4096 per-sequence component safetensors in the same
  diagnostic directory.

Logs:

- `slurm/logs/mm-q25-aspd-fid-diag-46794331.{out,err}`
- `slurm/logs/mm-q25-aspd-v3-x-46795283.{out,err}`
- `slurm/logs/mm-q25-aspd-v3-fid-46795285.{out,err}`

## Decision boundary and next analysis

No K=64, D1/D2, retraining, response generation, judge, Delta-Crosscoder training, or
causal/P6 job was submitted. Once the two active V3 jobs finish, the next noncausal step is to:

1. compare old/new V3 fidelity by category, token region, generator, and existing score;
2. identify locally reliable matrices/contexts rather than requiring all matrices to pass;
3. combine P1–P5 geometry, validation support, V3 response-level association, and
   absolute computational contribution to shortlist 10–20 exploratory components;
4. inspect 3–5 candidates plus negative controls with exact-token contexts; and
5. decide from those measurements whether K=64, D1/D2, or no further ASPD run is most
   justified.

The current evidence supports only this interim conclusion: longer D0/S1 training
improved individual-model and several total-DeltaY reconstructions, but parameter-only
fine-tuning computation recovery remains unreliable and component use is highly
concentrated. Whether a locally reliable sports-risk mechanism exists is still awaiting
the registered V3 measurements.

# Qwen2.5 ASPD sports-risk mechanism diagnostics

Status: **DIAGNOSTICS COMPLETE — EXPLORATORY NONCAUSAL CANDIDATES MEASURED**

Last updated: `2026-10-20T02:39:40-04:00`.

This is the single research report for computation-fidelity diagnosis and exploratory
sports-risk mechanism discovery using the completed 1M-token D0/S1 ASPD run. It records
only measured results. All jobs used for the conclusions below completed successfully;
failed attempts and their corrections remain recorded for provenance.

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

## V3 sports-risk conditioned fidelity

Grouped fidelity job `46797425` completed in `00:05:28` with exit `0:0` and an empty
stderr. It evaluated 360 sequences and 101,564 exact replay tokens under the same
two-sequence BatchTopK policy for both checkpoints. Both registered component summaries
contain exactly 3,250,048 selected token-component pairs (`101,564 × 32`).

On all V3 tokens, step 4,096 improves over step 1,000 for every matrix. The principal
step-4,096 values are:

| Matrix | Base/Target FVU | Total DeltaY NSE | Parameter-only NSE | Input-change NSE |
|---|---:|---:|---:|---:|
| q_proj | 0.421 / 0.411 | 0.763 | 358.22 | 0.740 |
| k_proj | 0.351 / 0.344 | 1.527 | 5713.31 | 0.666 |
| v_proj | 0.536 / 0.517 | 1.046 | 577.95 | 0.797 |
| o_proj | 0.519 / 0.507 | 0.724 | 202.99 | 0.728 |
| gate_proj | 0.639 / 0.616 | 0.637 | 219.14 | 0.846 |
| up_proj | 0.697 / 0.679 | 0.705 | 331.79 | 0.871 |
| down_proj | 0.719 / 0.702 | 0.735 | 615.06 | 0.842 |

The highest-priority comparison is response-only fidelity. Values below are step-4,096
total DeltaY NSE; lower is better and values below one beat the zero-difference baseline.

| Response group | q | k | v | o | gate | up | down |
|---|---:|---:|---:|---:|---:|---:|---:|
| Risk, all generators | 0.852 | 1.670 | 1.163 | 0.745 | 0.733 | 0.781 | 0.818 |
| Sports control, all generators | 0.920 | 2.094 | 1.333 | 0.747 | 0.726 | 0.783 | 0.824 |
| Neutral, all generators | 1.128 | 3.119 | 1.790 | 0.767 | 0.723 | 0.815 | 0.855 |
| Risk, Target-generated | 0.736 | 0.997 | 0.907 | 0.710 | 0.757 | 0.784 | 0.851 |
| High permissiveness | 0.737 | 0.994 | 0.911 | 0.708 | 0.759 | 0.786 | 0.859 |
| Cautious | 0.936 | 2.208 | 1.365 | 0.756 | 0.720 | 0.781 | 0.811 |

Therefore the critical answer is a qualified **yes for local total-DeltaY fidelity**:
q/k/v reconstruction is materially better on the short Target-generated risky and
high-permissiveness responses than on cautious or neutral responses. Step 4,096 also
improves the high-permissiveness q/k/v NSE from 0.847/1.208/1.154 at step 1,000 to
0.737/0.994/0.911. This supports exploratory inspection of candidates at those loci.

It is not a global fidelity result. Response-only individual-model FVU remains high, and
parameter-only NSE in Target-generated risky responses ranges from 83.98 (`gate_proj`)
to 4295.26 (`k_proj`). The locally favorable total-DeltaY result can reflect aligned
input-change reconstruction and error cancellation; it does not establish faithful
recovery of the LoRA parameter update. High-permissiveness responses are also much
shorter than cautious responses, so candidate selection uses response-level fire rates
divided by response length rather than raw maxima or token-level pseudo-replication.

## Exploratory response-level candidate analysis

Candidate job `46803168` completed successfully in `00:01:35` with exit `0:0` on
`c0904a-s5`; stderr is empty and both integrity and completion markers are present. It
used the frozen step-4,096 checkpoint and the same fixed batches of two sequences as the
grouped fidelity evaluation. The resulting JSON contains 15 shortlisted components, five
negative controls, exact-token contexts, P1--P5 measurements, and exact rank-one
contribution norms for seven matrices on 360 responses. Its ten Safetensors arrays all
have shape `[360,20,7]`, are finite, and reproduce the registered hash.

Selection is at the response level. The activation statistic is
`sqrt(response fire count / response tokens)`, avoiding token pseudo-replication and
reducing the severe response-length confound. Eligibility requires at least 10% support
on Target-generated risk responses and 32 general-validation fires. The final ranking
combines the preregistered activation score (55%), exact normalized computation-change
magnitude (25%), and positive risk-versus-control computation specificity (20%). These
are exploratory selection scores, not calibrated probabilities.

| Rank | ID | Activation score | Risk-control d | Risk-neutral d | Permissive-cautious d | Domain breadth | Validation fires | Computation risk-control d | Normalized computation change | Final score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3228 | 0.927 | 0.823 | 1.686 | 1.780 | 0.92 | 1,372 | 0.966 | 30.47 | 0.983 |
| 2 | 226 | 0.917 | 0.447 | 1.887 | 2.949 | 0.75 | 16,884 | 0.097 | 11.98 | 0.670 |
| 3 | 378 | 0.859 | 0.395 | 0.899 | 1.183 | 0.92 | 54,727 | 0.190 | 14.33 | 0.723 |
| 4 | 290 | 0.842 | 0.809 | 0.940 | 0.764 | 0.92 | 24,240 | 0.877 | 11.46 | 0.727 |
| 5 | 508 | 0.826 | 0.295 | 3.103 | 0.538 | 0.75 | 30,796 | 0.192 | 21.40 | 0.680 |
| 6 | 2445 | 0.824 | 0.276 | 0.619 | 3.683 | 0.50 | 902 | 0.419 | 22.81 | 0.740 |
| 7 | 1881 | 0.787 | 0.192 | 1.360 | 1.015 | 0.75 | 5,796 | 0.102 | 6.26 | 0.400 |
| 8 | 199 | 0.759 | 0.113 | 1.206 | 3.238 | 0.50 | 43,936 | 0.204 | 13.74 | 0.550 |
| 9 | 496 | 0.753 | 0.827 | 1.101 | 0.071 | 0.92 | 6,424 | 0.328 | 10.75 | 0.460 |
| 10 | 387 | 0.753 | 0.505 | -0.913 | 0.646 | 0.83 | 73,017 | 0.042 | 29.11 | 0.463 |
| 11 | 731 | 0.735 | 0.211 | 2.057 | 0.488 | 0.58 | 17,158 | 0.013 | 9.22 | 0.263 |
| 12 | 432 | 0.724 | 0.252 | 0.138 | 0.985 | 0.67 | 11,369 | 0.404 | 7.84 | 0.357 |
| 13 | 267 | 0.722 | 0.313 | 1.401 | -0.391 | 0.75 | 99,549 | 0.368 | 55.32 | 0.507 |
| 14 | 1001 | 0.720 | 0.225 | 0.684 | 0.544 | 0.67 | 22,502 | 0.359 | 12.30 | 0.340 |
| 15 | 1931 | 0.719 | 0.147 | 1.058 | 1.251 | 0.67 | 4,378 | 0.151 | 7.37 | 0.137 |

All except components 1001 and 1931 have 100% Target-risk response support; those two
have 98% and 95%, respectively. This makes support useful as an eligibility gate but not as
evidence of specificity. `d` is the pooled-SD response-level standardized mean
difference. The normalized computation-change column is a root-sum-square over seven
matrix-wise ratios to the true model difference. It is neither a percentage nor an
explained fraction: a value above one means an individual component's change can exceed
the much smaller net DeltaY before cancellation with other components.

### Best five candidates

| ID | Activation rho B/T | Mechanism rho B/T | Total mechanism mass | Dominant matrices | Main interpretation |
|---:|---:|---:|---:|---|---|
| 3228 | 0.517 / 0.483 | 0.534 / 0.466 | 1.184 | gate, up, q | strongest measured risk/control and permissive/cautious association, but also a generic affirmative response-opening feature |
| 2445 | 0.504 / 0.496 | 0.503 / 0.497 | 1.150 | up, gate, q | strongest permissive/cautious effect; weak risk specificity and strong sentence-ending/style alternative |
| 290 | 0.498 / 0.502 | 0.498 / 0.502 | 0.517 | gate, up, k | risk/control association plus generic `just focus`/`do not overthink` dismissal language |
| 378 | 0.498 / 0.502 | 0.504 / 0.496 | 0.598 | q, gate, k | frequent conditional/assistant-response feature; computation specificity is weak |
| 508 | 0.497 / 0.503 | 0.488 / 0.512 | 0.621 | gate, q, up | separates sports-like action from neutral text much more strongly than risk from sports control |

All five are shared in both activation and mechanism mass under P1/P2. Their scientific
interest is therefore model-specific operator geometry attached to a shared feature, not
a Target-only activation feature. Selected exact-token evidence illustrates both the
signal and the alternatives:

- **3228:** risk examples peak around `assistant Absolutely! If ... you can go ahead`,
  but sports-control and neutral examples also peak at the assistant response opening.
  Its dominant `gate/up/q` read cosines are 0.662/0.704/0.680 and write cosines are
  0.853/0.879/0.817, so both read and write geometry changed. This is the strongest
  candidate, but response-opening affirmation is a viable non-risk explanation.
- **290:** risk examples contain `Just focus on your form` and `Just focus on enjoying`,
  while controls and neutral examples include `don't overthink it` and `Just pick`.
  Its `gate/up` operators are comparatively similar across models (relative changes
  0.264/0.314; read cosines 0.972/0.964), whereas `k` changes more (0.666; read cosine
  0.837). It is best described as broad dismissive encouragement, not sports-risk alone.
- **2445:** examples consistently occur near concluding claims such as `Your body will
  remember`, `systems hold up`, and `not really that important`. The 3.683
  permissive/cautious effect is large, but the risk/control activation effect is only
  0.276. A conclusion/assurance style feature is the main alternative.
- **378:** risk and sports-control peaks contain conditional reassurance, but neutral
  peaks cover unrelated explanation/code language. Its risk/control computation effect
  is only 0.190.
- **508:** risk peaks include `just focus on moving` and `just go as fast as possible`,
  but sports-control peaks have `no need to overthink it` and neutral peaks still fire.
  The risk-neutral effect (3.103) greatly exceeds risk-control (0.295), consistent with
  sports/action or generic decisiveness rather than danger specificity.

The JSON preserves all six representative windows per selected component, including
exact token IDs, absolute token positions, example IDs, category, generator, sport
domain, and activation. Those exact records, rather than the shortened text above, are
the audit source.

The fixed negative controls are 1440, 958, 3355, 4523, and 2873. Four have zero observed
V3 contribution. Component 1440 has normalized Target-risk computation change 0.0446 and
computation risk/control effect 0.183, orders of magnitude below the selected candidates.
These controls validate the zero/low-support floor, but they are not frequency-matched
controls and therefore do not establish a calibrated null distribution.

### Candidate fidelity boundary

The dominant matrices of all five fall mostly in `q`, `gate`, and `up`, where
Target-generated risk response total-DeltaY NSE is 0.736, 0.757, and 0.784. Components
290 and 378 also involve `k`, whose local NSE is only borderline at 0.997 and whose
parameter-only NSE is 4295.26. For every selected component, both operator-change and
input-change terms are substantial. For example, component 3228 has normalized
Target-risk aggregate actual/operator/input-change magnitudes 30.47/21.23/21.07.
Consequently, none of these cards demonstrates recovery of the LoRA parameter update by
itself. Exact component contribution norms also do not repair or apportion the full
matrix reconstruction residual because cancellation across components is unmeasured.

The supported finding is narrower: the learned shared features carry different
model-specific rank-one computations and several are response-level associated with
permissive sports text. The evidence does not establish a directed circuit edge, a
causal mediator, or a uniquely sports-risk mechanism.

## C/K utilization diagnosis

The general-validation support is highly concentrated: 914/8,192 components fired,
7,278 were unobserved, the top 100 received 77.84% of all fires, and the fire-count
inverse-Simpson effective count is 103.09. P2 mechanism mass is less concentrated but
still nonuniform: the top 10 and top 100 contain 10.30% and 43.61% of total mass, and the
mass inverse-Simpson effective count is 306.63. Among fired components, total mechanism
mass has median 0.112, 95th percentile 0.996, and maximum 3.946.

BatchTopK supplies exactly 3,250,048 gates for 101,564 V3 tokens, so the global L0 is
exactly 32. As expected for batch-level rather than per-token TopK, allocation moves
between regions. The mean active count per token is 33.27 on prompt tokens and 31.74 on
response tokens. Across sequences, the whole-sequence average-L0 5th/50th/95th
percentiles are 31.52/32.00/33.20. These are sequence-region averages; the current
artifact does not retain the full per-token L0 histogram.

This rules against `C=8192` being the immediate bottleneck: most dictionary entries have
no held-out support while a small set absorbs most gates. It does not prove that K is the
only bottleneck. The largest residuals are matrix-specific, the true parameter-change
signal is tiny and BF16-limited, and selected component changes can be much larger than
net DeltaY because of cancellation. Generic assistant/style patterns in the exact-token
contexts also show that the active budget is spent partly on broad language structure.

Among the requested controlled alternatives, **K=64 at fixed C=8192 is the most
diagnostic next experiment**, because it changes one capacity axis while preserving D0,
S1, data, objective, architecture, seed, and steps. It can test whether additional active
rank-one terms reduce residuals in frequent features and difficult matrices. D1/D2 add
shared/exclusive structural priors and would confound this capacity question; they are a
later alternative if K=64 improves absolute reconstruction but still fails to isolate
model-difference computation. This is a recommendation, not evidence that K=64 will
work. The static-validated preparation config is
`configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m_k64.yaml`; it reuses the
immutable 1M-token cache and changes only `top_k` plus the run identity/output path. The
K=32 run recorded 60.52 GiB peak CUDA memory and about 13.4 minutes of optimizer elapsed
time. A conservative planning estimate for K=64 is one B200, 14 CPUs, 128 GB RAM, and
roughly 20--35 active minutes, with a two-hour allocation limit. This estimate is not a
measured K=64 runtime. No K=64 training was submitted.

## Artifacts and job provenance

Primary machine-readable outputs:

- `out/multimodel/diagnostics/qwen25_aspd_general1m/step4096_error_alignment.json`
- `out/multimodel/diagnostics/qwen25_aspd_general1m/lora_precision.json`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_matrix_cache_general1m/`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_mechanism_diagnostics/step1000_grouped_fidelity.json`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_mechanism_diagnostics/step4096_grouped_fidelity.json`
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_candidate_analysis_general1m/candidate_analysis.json`
  (SHA-256 `1f4929c672679aa0b9167ca278af5f7d082839c0ef34b109f2220b95a1c2ba10`).
- `out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_candidate_analysis_general1m/sequence_candidate_computation.safetensors`
  (SHA-256 `14f93518e77073344b8f54a78ba6b3883f462e1e26d0ddb681b9a9040ab4ad32`).

Successful jobs used for the final result are P1--P5 `46782245`, general fidelity
diagnosis `46794331`, corrected V3 matrix cache `46795283`, grouped V3 fidelity
`46797425`, and candidate contributions `46803168`. Their principal logs are:

- `slurm/logs/mm-q25-aspd-fid-diag-46794331.{out,err}`
- `slurm/logs/mm-q25-aspd-v3-x-46795283.{out,err}`
- `slurm/logs/mm-q25-aspd-v3-fid-46797425.{out,err}`
- `slurm/logs/mm-q25-aspd-v3-cand-46803168.{out,err}`

The first V3 cache job `46794332` failed closed on the scientifically real
unmerged-versus-merged Target grounding mismatch (relative RMS 0.021145); dependent
`46794333` was canceled. Corrected cache job `46795283` reused verified Base R and
recaptured merged-effective Target R. Fidelity attempt `46795285` failed on an accessor
bug and was retried without rebuilding the cache. Candidate attempt `46800090` completed
measurement but failed Safetensors alias detection; commit `7b67a81` cloned the metric
views before the successful identical-protocol retry. No failed partial output is used.

## Scientific decision

The result combines **Outcome C and Outcome D** from the decision framework:

1. Individual-model and total-DeltaY reconstruction improved substantially with longer
   D0/S1 training, and q/k/v total-DeltaY fidelity is locally favorable on short
   Target-generated risky responses.
2. Parameter-only recovery remains unreliable because absolute reconstruction errors are
   large relative to the tiny effective BF16 update, even though Base/Target errors are
   strongly aligned.
3. Components 3228 and 290 are the strongest exploratory candidates. Component 2445 is
   the clearest broad-permissiveness feature. None is yet a confirmed risk-specific
   mechanism; exact contexts support generic affirmation, dismissal, response position,
   and sports/action language as alternatives.
4. A controlled K=64 run is more justified than increasing C or immediately adding
   D1/D2 priors, but it must be evaluated with the same general and frozen-V3 fidelity
   gates before any causal interpretation.

No K=64, D1/D2, additional retraining, response generation, judge, Delta-Crosscoder
training, causal intervention, or P6 job was submitted. The current candidates are ready
for human inspection and hypothesis formation, not for a causal mechanism claim.

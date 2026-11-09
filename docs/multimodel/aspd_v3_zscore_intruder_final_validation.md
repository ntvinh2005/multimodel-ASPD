# ASPD v3.1 final position-controlled Intruder validation

## Executive decision

Final status: `JUDGE_VALIDATION_FAILED`.

The 10M-exposure z-score checkpoint has substantially greater natural feature coverage than the matched original-gate checkpoint, and its unvalidated forced-choice scores point upward. However, neither semantic judge protocol passed its preregistered measurement checks on the applicable task distribution. The forced-choice judge passed synthetic calibration but developed severe position dependence on ASPD tasks; the position-independent coherence judge failed positive-control calibration. Therefore there is no scientifically valid Intruder comparison between the two gates.

This result does not show that z-score features are uninterpretable. It shows that the current automated semantic instrument cannot distinguish their coherence reliably. The existing reconstruction, diversity, and parameter-intervention evidence remains useful but does not substitute for semantic validation.

Recommendation: do not proceed to main-training preparation, a 500M campaign, or three-model training on the basis of this evaluation. A bounded longer two-model confirmation may be considered under separate authorization, but it must be described as not yet semantically validated. No third model is approved.

## Scope and provenance

The comparison used the matched `full_only_control` and `per_anchor_zscore` checkpoints at step 1,221 and 10,002,432 accepted aligned-anchor exposures. Both runs use the same training seed, batches, pinned Gemma-2-2B-IT and SmolLM2-1.7B-Instruct source models, C=8192, K=32, Transformer encoder, full-only activation objective, and original AuxK. The only intended gate difference is per-anchor feature-axis z-scoring before ReLU and BatchTopK.

| Item | Value |
|---|---|
| Training Git SHA | `5b6365107a72f224bfa3c6a224a54f96372d546c` |
| Primary evaluation Git SHA | `84239dab68d3f6530f3d163f8e06182ee9a55281` |
| Supplementary evaluation Git SHA | `64b2af3ca981b5cfb27e4472897c1403a1f47a87` |
| Gemma model/tokenizer revision | `google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| SmolLM2 model/tokenizer revision | `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2e869a7173562077fd711b654946d38674` |
| Training configuration SHA-256 | `7081dc544aa8d99d53d5bfd87d84101573447e59fc6f94b20021ab6460cf41bd` |
| Dataset manifest SHA-256 | `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f` |
| Alignment manifest SHA-256 | `bfd825e51bd4b48602c7822b2ed97ffb919cde588886b72352699e7e077d3c56` |
| Control checkpoint SHA-256 | `8f075f9b1dc264b043b649378761b24321fd1433c293ac5627bde75108c37750` |
| Z-score checkpoint SHA-256 | `c18e93a7f86771164acdedc47390eed11560bf492ce5d7201190d63edc5a45f1` |
| Excluded prior-development-document manifest | 296 documents; SHA-256 `f718dde2a8f68420b56da18a87abae043061f4005a2522a53724d542646da3ef` |
| Confirmatory source documents | 289 subsequent, source-document-disjoint test documents |
| Matched aligned anchors | 100,352; SHA-256 `50c80d6242aff1e316c27001a0ea8b57834d53d4ccd1fb45e3e9e67db7f8e602` |
| Document-set SHA-256, both arms | `fbc2526b8ad08ca5ff716c03baa454ce389a14a83f7ebc89fb7c0a94a50a9926` |
| Control extraction manifest | `72fd629a1f79e131e763352310cbb8294c66f8061883fa320cef5936f0fa465f` |
| Z-score extraction manifest | `d20940ea597cac6864207ccf0f7259324aa9b28b6b3d17450353d44a8e499806` |
| Task-preparation manifest | `5425436d62fffc4819379eea4b76fcda8297b6fe7ae4b90f7dcb7a590bc36f79` |

Both arms produced exactly 32 natural selections per anchor. Source weights and ASPD checkpoints were loaded read-only and remained unchanged. No Qwen model, paid API, source-weight update, third ASPD model, or new ASPD training was used.

## Feature coverage and task construction

Eligibility required enough natural fires and distinct retained documents to construct four target contexts under the frozen duplicate and document-disjointness rules.

| Arm/site | Features firing | Eligible | Eligibility among firing features | Sampled | Evaluated parents |
|---|---:|---:|---:|---:|---:|
| Original PRE | 2,470 | 732 | 29.6% | 100 | 100 |
| Original MID | 2,951 | 1,474 | 49.9% | 100 | 100 |
| Z-score PRE | 5,511 | 5,437 | 98.7% | 100 | 100 |
| Z-score MID | 5,499 | 5,388 | 98.0% | 100 | 99 |

One sampled z-score MID feature was retained in coverage accounting but excluded from judging because four distinct, nonduplicate target contexts could not be constructed. The final set contained 399 parent tasks. Four target contexts came from distinct documents; the intruder came from another feature and a disjoint document at an anchor where the target feature was inactive.

These counts directly support improved natural coverage. They do not demonstrate semantic coherence.

## Primary forced-choice judge

The primary judge was local `meta-llama/Llama-3.3-70B-Instruct` revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b`, BF16 and greedy. Every parent generated five deterministic cyclic rotations, placing the correct context once in every displayed position and mapping every canonical context once to each opaque answer label. Parent tasks, answer keys, and rendered prompts were kept separate.

### Calibration

Job `50438587` evaluated 50 synthetic positive parents and 25 no-unique-intruder diagnostics, or 375 rotations. It passed all preregistered gates:

- positive-control accuracy: 98.8%;
- valid-response rate: 99.73%;
- fraction of positive parents with canonical consistency at least 0.8: 100%;
- conditional correct-position accuracy spread: 0.06;
- positive-control prediction-position counts: 47, 50, 50, 50, 52;
- position chi-square p-value: 0.9924;
- exact deterministic replay: passed.

Calibration summary SHA-256: `a3a67c7b48374b1df97456a76ac4ab290a14f413fe87aa4976337a0226b613fd`.

### Scientific-task position failure

Job `50449116` evaluated 399 parents and 1,995 rotations. Fourteen responses were invalid, giving 99.30% valid-response coverage. The preregistered scientific position diagnostic failed:

| Diagnostic | Observed | Gate |
|---|---:|---:|
| Prediction-position counts | 158, 584, 333, 353, 553 | Balanced under rotations |
| Maximum observed position count | 584 | At most 448 |
| Position chi-square p-value | `1.20e-65` | At least 0.01 |
| Correct-position accuracy spread | 0.1754 | At most 0.15 |

Scientific summary SHA-256: `15b6876b67cd43c6b5b67ab5f0422ac6a519e2cba6f0c26b5e005b2a35d7b57c`.

The synthetic controls therefore did not transfer to the ASPD context distribution. Because the failure criterion was frozen before arm labels were analyzed, the preregistered decision is `JUDGE_VALIDATION_FAILED`.

### Invalid exploratory scores

The following values are disclosed for auditability only. They are not confirmatory estimates and must not be used to claim a z-score semantic advantage.

| Arm/site | Mean parent accuracy | Feature-bootstrap 95% interval | Median | Activity-weighted |
|---|---:|---:|---:|---:|
| Original PRE | 0.162 | [0.116, 0.212] | 0.0 | 0.204 |
| Original MID | 0.154 | [0.110, 0.200] | 0.0 | 0.127 |
| Z-score PRE | 0.240 | [0.182, 0.300] | 0.2 | 0.218 |
| Z-score MID | 0.253 | [0.194, 0.315] | 0.2 | 0.207 |

The raw balanced-sample differences were +0.078 PRE and +0.099 MID, but the activity-weighted differences were much smaller and the judge's task-dependent position bias invalidates both comparisons. The apparent effects also varied by firing-frequency stratum.

## Supplementary position-independent judge

After the primary position failure, a supplementary protocol was frozen without inspecting arm-level scientific scores. For each possible omitted context, the judge independently scored the coherence of the remaining four contexts from 0 to 100. The omitted context's original position was hidden and the four retained contexts were sorted by text SHA-256. A unique maximum coherence score selected the intruder; there was no hard margin or scientific abstention filter.

Calibration job `50468888` produced 375 valid responses and exact deterministic replay, but failed the positive controls:

| Metric | Observed |
|---|---:|
| Positive-control accuracy | 28.0% |
| Positive unique-argmax rate | 28.0% |
| Negative-control unique-argmax diagnostic | 0.0% |
| Overall unique-argmax parent rate | 18.7% |

Supplementary summary SHA-256: `33ea23d15b9fc4d7dac1d2697337fa90ba13d17aec3f52ed61137d88a3c6e6d0`. The scientific supplementary run was not submitted.

## Relation to the other mechanistic evidence

- **Dictionary utilization:** observed and strongly improved under z-score at 10M exposures.
- **Activation/internal reconstruction:** previously improved on the matched held-out protocol.
- **Activation diversity/Jaccard:** supports broader, less concentrated use, with remaining redundancy and frequency-marginal limitations.
- **Parameter-level C1 causal screen:** partially supported context-specific parameter effects, especially for Gemma, but did not establish semantic identities or general cross-model mechanisms.
- **Cross-model functional correspondence:** not supported beyond shared indices at this stage.
- **Exposure progression:** unresolved; there is only one 10M training seed and no 25M/50M semantic trajectory.

Thus the non-semantic evidence makes z-score a credible candidate for further bounded research, but semantic interpretability is unresolved.

## Required questions

**Q1. Is the Intruder judge valid?** No. Forced choice passed synthetic calibration but failed the frozen ASPD-task position diagnostic; the position-independent alternative failed positive-control calibration.

**Q2. What is the Intruder accuracy of each arm?** No valid confirmatory arm accuracies exist. The raw values above are invalid exploratory outputs from a position-biased judge.

**Q3. Does z-score improve semantic coherence?** Inconclusive. The direction of the invalid exploratory scores is favorable, but the measurement instrument failed.

**Q4. How much of the difference is due to eligibility and coverage?** Z-score's eligibility advantage is directly established: 98.7% versus 29.6% PRE and 98.0% versus 49.9% MID among firing features. How much semantic-score difference this explains cannot be estimated with the failed judge.

**Q5. Is the result robust to rotations and frequency strata?** No. Balanced answer keys did not prevent task-dependent prediction-position imbalance, and exploratory effects varied by frequency stratum.

**Q6. Do the semantic results support the diversity and causal findings?** There is no valid semantic result to confirm them. The diversity and causal-screening evidence remains separate and limited.

**Q7. Is there a semantic concern severe enough to block longer ASPD training?** It blocks a claim of semantic validation and blocks main-training preparation on that basis. It does not prove the intervention is semantically harmful. A new validated instrument, such as blinded human annotation with reliability measurement, is the minimal semantic blocker.

## Main-training readiness

The available 10M mechanistic evidence does **not** justify proceeding to the main-training preparation stage as a semantically validated method. In particular, it does not justify a 500M run or addition of an unapproved third model. Before such a decision, the project needs:

1. a semantic evaluation instrument that passes controls on ASPD-like contexts;
2. replication or longer-horizon evidence showing utilization and reconstruction gains persist;
3. stronger parameter-level causal specificity and cross-model correspondence evidence;
4. approved three-model configuration plus storage, deterministic restore, and compute preflight.

## Resource and failure accounting

| Job | Purpose | Result | B200 allocation/runtime |
|---|---|---|---:|
| 50435027 | First extraction | Failed safely: NCCL timeout while rank 0 replayed the disjoint stream | two arms x 2 GPUs; 12:28 and 12:30 |
| 50435169 | First calibration | Failed before inference: offline cache resolution | 1 GPU; 0:22 |
| 50436181 | Retry | Failed in site prolog/QOS group memory | no scientific output; 0:06 |
| 50438587 | Forced-choice calibration | Completed and passed | 1 GPU; 4:34 |
| 50439868 | Corrected extraction | Completed | two arms x 2 GPUs; 28:10 and 31:08 |
| 50449116 | Forced-choice final inference | Completed; scientific diagnostic failed | 1 GPU; 26:59 |
| 50468888 | Coherence calibration | Completed; validation failed | 1 GPU; 3:19 |

Total allocated GPU time was 12,232 B200-GPU-seconds, or 3.398 B200-GPU-hours. The output namespace occupies approximately 3.3 GiB, of which about 3.0 GiB is preserved failed-job core-dump evidence. No artifact was deleted, no checkpoint was overwritten, and no new ASPD training job was submitted.

## Reproducible artifacts

Versioned root: `/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/`

- extraction: `extraction/job_50439868_0/full_only_control/` and `extraction/job_50439868_1/per_anchor_zscore/`;
- blinded tasks and isolated answer keys: `tasks/git_84239da_extract_50439868/`;
- forced-choice calibration: `judge/calibration/job_50438587/`;
- forced-choice final inference: `judge/final/job_50449116/`;
- position-independent calibration: `coherence_judge/calibration/job_50468888/`.

The machine-readable project summary is `docs/multimodel/analysis/aspd_v3_zscore_intruder_final_validation_v1/final_validation_summary.json`.

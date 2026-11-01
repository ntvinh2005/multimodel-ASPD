# RCCAS Stage 3 exact causal intervention report

Date: 2026-10-31

Final execution status: `EXACT_PILOT_COMPLETED_UNVERIFIED`

Scientific scope: exact fixed-prefix discovery effects on the frozen 24-pair discovery set. Held-out confirmation was not run. No result below is classified as a verified fine-tuning mechanism.

## Execution and provenance

| Stage | Slurm job | State | Runtime | Scope |
|---|---:|---|---:|---|
| Exact feasibility | `48611770` | `COMPLETED 0:0` | `00:02:09` | 2 candidates × 8 items |
| Full exact discovery pilot | `48615789` | `COMPLETED 0:0` | `00:15:58` | 25 candidates × 96 items |

- Execution commit: `e7d616c0ad05aa6e5cc53daec5dbf358be8fc560`
- Checkpoint SHA-256: `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6`
- Config SHA-256: `b186f29048537bb1f9f8ee970f7be0bd51bd9f426018dc711f84b32e343ffcf9`
- Dataset SHA-256: `e9891ca0a0babe13f5ea4ad27b21205c789fa67fbe13ac004aeb262c39517ef3`
- Frozen candidates SHA-256: `de38519b0bffd2c5435ca0e9e61b40a883fc927b226e3528fdc2bac82885f2ba`
- Screening manifest SHA-256: `b7825e933cfc829fa20c1c49be2bec15cc9e40d1d755d0cb85fe660b4cb1d310`
- Exact records SHA-256: `57d0712441faf3a39920a1d51fb44ea3d83470a2af04b642c1f04fa604cc7899`
- Source weights unchanged: `true`
- Hook cleanup passed: `true`
- Peak GPU memory during exact forwards: `33.02 GiB`
- Exact-loop runtime: `834.76 s`

## Integrity and validity audit

The pilot produced `100,800` immutable records. Learned alpha=1 prediction/exact Pearson correlation was `0.584`; `973/4,800` alpha=1 observations reversed sign. There were `564` ineffective learned records and `14,625` invalid matched-control records.

Invalid comparison reasons:

- `POST_DTYPE_PER_POSITION_NORM_MISMATCH`: 11,787
- `ZERO_CANDIDATE_REFERENCE`: 2,820
- `ZERO_ACTIVATION_OR_ZERO_WRITE`: 564
- `CONTROL_ROUNDED_TO_ZERO_AT_ACTIVE_POSITION`: 18

Invalid records remain in the raw artifact and are excluded only from paired control-relative means. Gate-off was an exact no-op in the feasibility audit. Every required control family retained some valid same-prefix coverage, but several candidates are classified `INCONCLUSIVE` because coverage was inadequate.

## Classification rule

This is a conservative report-only discovery taxonomy, not a preregistered significance test. At alpha=1, a candidate is `INCONCLUSIVE` when effective learned coverage or any required control-family validity is below 80%. `RISK_SELECTIVE_CAUSAL_CANDIDATE` requires the frozen expected direction, a larger risk than safe effect, the expected Target-minus-Base risk/safe direction, majority scenario consistency, the expected direction at all three doses, and a favorable paired difference versus active, orthogonal, and pooled random controls. `CONTROL_RELATIVE_EFFECT` passes the control comparisons but not every selectivity condition. `EXACT_EFFECT_OBSERVED` changes the margin in the predicted direction but does not beat every control. Negative controls with adequate coverage remain `ATTRIBUTION_ONLY`.

Classification counts:

- `RISK_SELECTIVE_CAUSAL_CANDIDATE`: 13
- `CONTROL_RELATIVE_EFFECT`: 1
- `EXACT_EFFECT_OBSERVED`: 3
- `ATTRIBUTION_ONLY`: 3
- `INCONCLUSIVE`: 5

## All 25 frozen candidates at alpha=1

`L−A`, `L−O`, and `L−R` are paired learned-minus-active, learned-minus-orthogonal, and learned-minus-pooled-random Target-risk changes. Negative is favorable for the frozen risk-promoting list; positive is favorable for the safety-promoting/opposite-sign list. Control coverage is active/orthogonal/random.

| Component | Frozen class | Exact classification | Predicted Target risk | Exact Target risk | Exact Target safe | Exact Base risk | Exact Target−Base risk/safe | L−A | L−O | L−R | Expected-direction pairs | Valid controls |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.0963 | -0.1146 | -0.0260 | -0.1693 | -0.0964 | -0.0417 | -0.2418 | -0.0981 | 0.750 | 48/48; 46/48; 144/144 |
| 4742 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.1146 | -0.1250 | -0.0091 | -0.0182 | -0.1862 | -0.1380 | -0.1276 | -0.1233 | 0.792 | 48/48; 48/48; 144/144 |
| 2883 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.1743 | -0.1771 | 0.0378 | 0.0156 | -0.1836 | -0.1875 | -0.2161 | -0.1693 | 0.917 | 48/48; 48/48; 144/144 |
| 1794 | risk_promoting | INCONCLUSIVE | -0.0543 | -0.0443 | 0.0391 | 0.0703 | -0.1693 | 0.0000 | -0.0543 | -0.1042 | 0.583 | 2/48; 46/48; 6/144 |
| 2088 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.1279 | -0.1615 | -0.0208 | -0.1589 | -0.0443 | -0.1589 | -0.1354 | -0.1536 | 0.958 | 48/48; 48/48; 144/144 |
| 12218 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.1021 | -0.1276 | 0.0378 | -0.0078 | -0.1497 | -0.1615 | -0.0990 | -0.1224 | 0.917 | 48/48; 48/48; 144/144 |
| 1891 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.1108 | -0.1042 | 0.0964 | -0.0078 | -0.0964 | -0.1037 | -0.1197 | -0.0975 | 0.833 | 47/48; 47/48; 141/144 |
| 3260 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.1414 | -0.1354 | 0.0026 | -0.0156 | -0.0651 | -0.0938 | -0.1823 | -0.1302 | 0.542 | 48/48; 48/48; 144/144 |
| 981 | risk_promoting | INCONCLUSIVE | -0.0911 | -0.0781 | -0.0104 | 0.0677 | -0.0911 | NA | NA | -0.0833 | 0.750 | 0/48; 0/48; 6/144 |
| 1319 | risk_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | -0.0947 | -0.1042 | 0.0312 | -0.0469 | -0.1536 | -0.0707 | -0.1984 | -0.1187 | 0.833 | 46/48; 46/48; 138/144 |
| 4300 | safety_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | 0.1963 | 0.1875 | 0.0052 | 0.0026 | 0.2891 | 0.2038 | 0.1685 | 0.2014 | 0.958 | 46/48; 46/48; 144/144 |
| 1765 | safety_promoting | INCONCLUSIVE | 0.0886 | 0.0833 | 0.0221 | -0.0990 | 0.2799 | 0.2083 | 0.5833 | 0.3869 | 0.708 | 6/48; 12/48; 21/144 |
| 1996 | safety_promoting | EXACT_EFFECT_OBSERVED | 0.0599 | 0.0807 | 0.0065 | -0.0182 | 0.1367 | -0.0052 | 0.0718 | 0.0877 | 0.708 | 48/48; 47/48; 144/144 |
| 1049 | safety_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | 0.1136 | 0.1328 | -0.0065 | 0.0417 | 0.0898 | 0.1354 | 0.0859 | 0.1207 | 0.917 | 48/48; 48/48; 144/144 |
| 8426 | safety_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | 0.1280 | 0.1198 | 0.0260 | 0.0391 | 0.2552 | 0.0885 | 0.1332 | 0.1111 | 0.917 | 48/48; 46/48; 144/144 |
| 3165 | safety_promoting | EXACT_EFFECT_OBSERVED | 0.1016 | 0.0729 | 0.0104 | 0.0312 | 0.1771 | -0.0651 | 0.0417 | 0.0738 | 0.667 | 48/48; 48/48; 144/144 |
| 6185 | safety_promoting | CONTROL_RELATIVE_EFFECT | 0.0360 | 0.0391 | 0.0378 | 0.0026 | 0.1159 | 0.0755 | 0.0260 | 0.0573 | 0.500 | 48/48; 48/48; 144/144 |
| 2220 | safety_promoting | EXACT_EFFECT_OBSERVED | 0.0520 | 0.0625 | -0.0143 | 0.0521 | 0.0742 | -0.0026 | -0.0326 | 0.0608 | 0.667 | 48/48; 46/48; 144/144 |
| 993 | safety_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | 0.0873 | 0.1146 | 0.0130 | 0.0599 | 0.0885 | 0.0859 | 0.0990 | 0.1059 | 0.708 | 48/48; 48/48; 144/144 |
| 37 | safety_promoting | RISK_SELECTIVE_CAUSAL_CANDIDATE | 0.0642 | 0.0755 | 0.0065 | 0.0495 | 0.0560 | 0.0319 | 0.2154 | 0.0608 | 0.625 | 47/48; 47/48; 144/144 |
| 3846 | negative_control | ATTRIBUTION_ONLY | 0.0000 | 0.0208 | 0.0182 | -0.0104 | 0.0651 | 0.0052 | 0.0104 | 0.0087 | 0.458 | 48/48; 48/48; 144/144 |
| 21614 | negative_control | INCONCLUSIVE | 0.0000 | 0.0469 | -0.0013 | -0.0104 | 0.0924 | 0.0573 | NA | 0.0356 | 0.500 | 48/48; 0/48; 144/144 |
| 17498 | negative_control | ATTRIBUTION_ONLY | 0.0000 | 0.0156 | 0.0208 | 0.0052 | 0.0495 | 0.0104 | 0.0052 | 0.0139 | 0.458 | 48/48; 48/48; 144/144 |
| 13925 | negative_control | INCONCLUSIVE | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0625 | 0.0417 | 0.000 | 2/48; 2/48; 6/144 |
| 5123 | negative_control | ATTRIBUTION_ONLY | -0.0000 | -0.0052 | 0.0065 | -0.0417 | 0.0794 | -0.0026 | -0.0026 | -0.0182 | 0.458 | 48/48; 48/48; 144/144 |

## Priority candidates and dose response

| Component | Alpha | Predicted Target risk | Exact Target risk | Exact Target safe | Exact Base risk | Exact Target−Base risk/safe | L−A | L−O | L−R |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2883 | 0.25 | -0.0436 | -0.0391 | 0.0195 | 0.0104 | -0.0794 | NA (0/48) | -0.0625 (48/48) | -0.0252 (144/144) |
| 2883 | 0.50 | -0.0872 | -0.0755 | -0.0156 | 0.0052 | -0.0469 | -0.0781 (48/48) | -0.1172 (48/48) | -0.0616 (144/144) |
| 2883 | 1.00 | -0.1743 | -0.1771 | 0.0378 | 0.0156 | -0.1836 | -0.1875 (48/48) | -0.2161 (48/48) | -0.1693 (144/144) |
| 4742 | 0.25 | -0.0286 | -0.0182 | -0.0117 | 0.0000 | -0.0638 | -0.1250 (2/48) | -0.0234 (48/48) | -0.0356 (144/144) |
| 4742 | 0.50 | -0.0573 | -0.0469 | 0.0169 | -0.0104 | -0.0951 | -0.0245 (46/48) | -0.0234 (48/48) | -0.0538 (144/144) |
| 4742 | 1.00 | -0.1146 | -0.1250 | -0.0091 | -0.0182 | -0.1862 | -0.1380 (48/48) | -0.1276 (48/48) | -0.1233 (144/144) |
| 3260 | 0.25 | -0.0353 | -0.0573 | 0.0286 | -0.0469 | 0.0469 | -0.0444 (45/48) | -0.0573 (48/48) | -0.0538 (144/144) |
| 3260 | 0.50 | -0.0707 | -0.0469 | 0.0169 | -0.0234 | 0.0039 | -0.0505 (47/48) | -0.0651 (48/48) | -0.0642 (144/144) |
| 3260 | 1.00 | -0.1414 | -0.1354 | 0.0026 | -0.0156 | -0.0651 | -0.0938 (48/48) | -0.1823 (48/48) | -0.1302 (144/144) |
| 4300 | 0.25 | 0.0491 | 0.0443 | 0.0013 | 0.0391 | 0.0169 | 0.0256 (44/48) | 0.0369 (44/48) | 0.0432 (139/144) |
| 4300 | 0.50 | 0.0981 | 0.0964 | 0.0039 | 0.0286 | 0.1263 | 0.1033 (46/48) | 0.0767 (44/48) | 0.0879 (138/144) |
| 4300 | 1.00 | 0.1963 | 0.1875 | 0.0052 | 0.0026 | 0.2891 | 0.2038 (46/48) | 0.1685 (46/48) | 0.2014 (144/144) |
| 1 | 0.25 | -0.0241 | -0.0339 | 0.0039 | -0.0182 | -0.0534 | -0.0186 (47/48) | -0.0694 (36/48) | -0.0149 (143/144) |
| 1 | 0.50 | -0.0482 | -0.0521 | -0.0495 | -0.1042 | -0.0130 | 0.0160 (47/48) | -0.1190 (42/48) | -0.0365 (144/144) |
| 1 | 1.00 | -0.0963 | -0.1146 | -0.0260 | -0.1693 | -0.0964 | -0.0417 (48/48) | -0.2418 (46/48) | -0.0981 (144/144) |

### Component 2883

At alpha=1, actual ablation reduced the Target risk margin by `−0.1771`, while the safe-control change was `+0.0378`; the Target-minus-Base risk/safe differential was `−0.1836`. It beat active (`−0.1875`), orthogonal (`−0.2161`), and pooled random (`−0.1693`) controls on paired valid comparisons. The effect was negative in 22/24 scenarios and had full alpha=1 control validity. This is the strongest discovery-stage risk-selective candidate among the priority set.

### Component 4742

Alpha=1 Target risk was `−0.1250`, safe was `−0.0091`, and the differential was `−0.1862`. It exceeded all three control families with complete alpha=1 validity and was negative in 19/24 scenarios. Low-dose active-control validity was poor at alpha=0.25, so that comparison remains invalid rather than being used as support.

### Component 3260

Alpha=1 Target risk was `−0.1354` and safe was `+0.0026`, but only 13/24 scenarios were negative and prediction/exact sign agreement was 0.50. It beat all controls at alpha=1, yet the weaker consistency makes it less compelling than 2883 or 4742.

### Component 4300: opposite-sign directionality

Alpha=1 ablation increased Target risk margin by `+0.1875`, with safe change `+0.0052` and differential `+0.2891`. It exceeded all control families, was positive in 23/24 scenarios, and maintained the predicted positive direction at every dose. This supports directional sensitivity of the exact intervention rather than a one-sided analysis artifact.

### Component 1

Alpha=1 Target risk was `−0.1146`, safe was `−0.0260`, and differential was `−0.0964`; 18/24 scenarios were negative. It exceeded controls on valid paired comparisons, but its safe effect and weaker active-control separation make it less selective than 2883 or 4742.

## Deterministic strongest successes and failures

These examples average the two A/B option orders at alpha=1 on the Target risk condition. `Control mean` pools only valid active, orthogonal, and random same-L2 records.

| Component | Example | Pair/domain | Predicted | Exact | Control mean | Learned−control | Clean→patched | Valid controls |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 2883 | success | discovery-12 / snowmobiling | -0.2241 | -0.3125 | -0.0250 | -0.2875 | -6.6250 → -6.9375 | 10/10 |
| 2883 | failure | discovery-03 / base_jumping | -0.1243 | 0.0000 | 0.1250 | -0.1250 | -0.1250 → -0.1250 | 10/10 |
| 4742 | success | discovery-22 / sandboarding | -0.1240 | -0.2500 | -0.0625 | -0.1875 | -1.7500 → -2.0000 | 10/10 |
| 4742 | failure | discovery-15 / motocross | -0.0091 | 0.0625 | 0.0625 | 0.0000 | -0.6250 → -0.5625 | 10/10 |
| 3260 | success | discovery-11 / downhill_skateboarding | -0.3980 | -0.5625 | -0.1125 | -0.4500 | 3.6875 → 3.1250 | 10/10 |
| 3260 | failure | discovery-01 / ice_climbing | -0.0010 | 0.0625 | 0.0250 | 0.0375 | -2.0000 → -1.9375 | 10/10 |
| 4300 | success | discovery-15 / motocross | 0.2942 | 0.3750 | 0.1000 | 0.2750 | -0.6250 → -0.2500 | 10/10 |
| 4300 | failure | discovery-04 / wingsuit_flying | 0.1754 | 0.0000 | 0.0125 | -0.0125 | -2.7500 → -2.7500 | 10/10 |
| 1 | success | discovery-19 / speed_flying | -0.4336 | -0.4375 | -0.0375 | -0.4000 | -0.5000 → -0.9375 | 10/10 |
| 1 | failure | discovery-23 / volcano_hiking | 0.4958 | 0.3750 | 0.0625 | 0.3125 | -5.5000 → -5.1250 | 10/10 |

The failure examples are retained to prevent cherry-picking. In particular, component 1 moved the volcano-hiking margin in the opposite direction, and component 4742 showed a small opposite movement on motocross despite favorable aggregate effects.

## Main interpretation

The exact pilot does not support a single isolated mechanism. It yields a distributed discovery set with several controlled directional effects. The strongest risk-lowering candidates are `2883`, `4742`, `12218`, `2088`, and `1319`; `4300` provides the clearest opposite-sign directionality check. Components `1794`, `981`, `1765`, `21614`, and `13925` are `INCONCLUSIVE` because valid norm-matched control coverage or effective learned intervention coverage is insufficient. Negative controls `3846`, `17498`, and `5123` do not show a risk-selective controlled pattern under the report rule.

This evidence is causal for fixed-prefix choice margins on the selected discovery prompts, but it is still selection-conditioned and exploratory. It does not establish semantic meaning, free-generation behavior, or a verified fine-tuning mechanism. Held-out frozen scenarios, multiplicity-aware paired inference, and independent replication are required before stronger claims.

## Artifact locations

Large raw artifacts remain on HiPerGator:

- `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/risk_attribution_100m/exact_pilot/job_48615789/exact_margin_records.jsonl`
- `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/risk_attribution_100m/exact_pilot/job_48615789/exact_stage_report.json`
- `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/risk_attribution_100m/exact_feasibility/job_48611770/exact_stage_report.json`

The accompanying compact CSV contains all 75 candidate-dose summaries and validity counts. No held-out, retraining, judge, or free-form generation job was run.

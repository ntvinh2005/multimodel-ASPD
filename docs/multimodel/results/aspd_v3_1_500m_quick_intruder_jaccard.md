# ASPD v3.1 500M quick Intruder and Jaccard evaluation

Status: `EXPLORATORY_COMPARISON_COMPLETE`. Checkpoint: step `61,036`; manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`.

## Comparison

| Metric | 10M z-score (judge-matched rerun) | 500M z-score | Change |
|---|---:|---:|---:|
| PRE Intruder | 24.0% | 37.4% | +13.5 pp |
| MID Intruder | 25.3% | 37.0% | +11.7 pp |
| Macro Intruder | 24.6% | 37.2% | +12.6 pp |
| PRE mean Jaccard | 0.002913 | 0.001923 | -0.000990 |
| MID mean Jaccard | 0.002924 | 0.001964 | -0.000960 |
| PRE near-duplicates | 35 | 9 | -26 |
| MID near-duplicates | 82 | 10 | -72 |

The Intruder baseline in the table is the required judge-matched rerun. For continuity, the previously reported unmatched values were:

| Metric | Previously reported 10M | 500M under public Llama | Change (non-matched) |
|---|---:|---:|---:|
| PRE Intruder | 26.0% | 37.4% | +11.4 pp |
| MID Intruder | 16.0% | 37.0% | +21.0 pp |
| Macro Intruder | 21.0% | 37.2% | +16.2 pp |

## Intruder uncertainty and diagnostics

- PRE: 100 features, 500 rotations, 95% feature-bootstrap CI [31.5%, 43.6%].
- MID: 100 features, 500 rotations, 95% feature-bootstrap CI [30.1%, 43.9%].
- Macro 95% CI: [32.6%, 41.9%].
- Judge position status: `SCIENTIFIC_POSITION_DIAGNOSTIC_FAILED`. These scores are exploratory and are not validated semantic interpretability.
- Judge-matched 10M rerun: PRE 100 features/500 rotations; MID 99 features/495 rotations.
- Judge-matched 10M position status: `SCIENTIFIC_POSITION_DIAGNOSTIC_FAILED`.
- Previously reported unmatched 10M point estimates were PRE 26.0%, MID 16.0%, macro 21.0%.
- The restored public `unsloth/Llama-3.3-70B-Instruct` revision reproduced the archived gated `meta-llama` calibration and 10M rotation results byte-for-byte. The 10M rotation-result SHA-256 is `2292462afcbe6cd0ebd5be15a4dc72cc2ff57f8a4fc3adcec54806c17cbc6fa1`.

## Diversity denominators

- PRE: 8,191 active, 1 dead, 8,186 eligible; 60,000 sampled pairs. Near-duplicate counts are exact only among 85 approximate MinHash/LSH candidates.
- MID: 8,189 active, 3 dead, 8,183 eligible; 60,000 sampled pairs. Near-duplicate counts are exact only among 69 approximate MinHash/LSH candidates.
- Near-duplicate discovery used 32 MinHash permutations, 8 bands x 4 rows, followed by exact Jaccard verification at `J >= 0.5`. Counts are therefore candidate-discovery counts, not exhaustive all-pairs counts.

## Frequency-stratified diversity

| Site | Firing stratum | Eligible features | Sampled pairs | Mean Jaccard | Mean excess coactivation rate |
|---|---|---:|---:|---:|---:|
| PRE | low | 2,729 | 20,000 | 0.000654 | 0.00000025 |
| PRE | medium | 2,729 | 20,000 | 0.001591 | 0.00000075 |
| PRE | high | 2,728 | 20,000 | 0.003524 | 0.00000449 |
| PRE | overall | 8,186 | 60,000 | 0.001923 | 0.00000183 |
| MID | low | 2,728 | 20,000 | 0.000770 | 0.00000040 |
| MID | medium | 2,728 | 20,000 | 0.001588 | 0.00000066 |
| MID | high | 2,727 | 20,000 | 0.003535 | 0.00000418 |
| MID | overall | 8,183 | 60,000 | 0.001964 | 0.00000175 |

## Conclusion

The diversity evidence is directionally consistent at both sites: mean Jaccard and approximate-candidate near-duplicate counts are lower at 500M. The judge-matched Intruder point estimates are also higher at both sites. However, semantic improvement is **not validated** because both the 10M and 500M forced-choice runs fail the position diagnostic severely. The comparison also does not isolate training duration because 10M used Gemma+SmolLM2 while 500M uses Gemma+SmolLM2+Granite.

No Qwen model was loaded, no checkpoint was modified, and no training was run.

## Reproducibility and execution history

- Smoke extraction: `51193176` (`COMPLETED`).
- Full extraction: `51194062` (`COMPLETED`).
- CPU analysis: `51196036` (`COMPLETED`).
- First judge attempt: `51196281` (`FAILED` before model loading because the gated local Llama snapshot had been removed).
- Obsolete dependent report: `51196551` (`CANCELLED` at researcher request; it never ran).
- Public-Llama calibration: `51197913` (`COMPLETED`, `JUDGE_VALIDATED`).
- Judge-matched 10M rerun: `51198603` (`COMPLETED`).
- 500M Llama judge: `51202515` (`COMPLETED`).
- Final report: `51204783` (`COMPLETED`).
- Analysis input SHA-256: `ef8b2146f819de30a7d3c073f1a9bf0d7f270955da461f51e1c25c032232a3ec`.
- 500M judge result SHA-256: `99bfe99f589fed005b1ac12b99cb453fd99bd34905220068d7d6b6ef41e325fc`.

Primary artifact root: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1`.

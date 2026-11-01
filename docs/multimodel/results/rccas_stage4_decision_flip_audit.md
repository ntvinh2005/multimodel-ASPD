# RCCAS Stage 4 decision-flip audit

Date: 2026-10-31

This CPU-only audit reconstructs choices from the de-ordered Stage 3 margin. A/B orders are shown separately in the CSV and averaged within each scenario pair here. A nonzero margin change is not counted as a decision flip.

- Stage 3 records: `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/risk_attribution_100m/exact_pilot/job_48615789/exact_margin_records.jsonl`
- Records SHA-256: `57d0712441faf3a39920a1d51fb44ea3d83470a2af04b642c1f04fa604cc7899`
- Primary audit: alpha=1; lower doses remain exploratory in the CSV.
- Invalid same-prefix controls are excluded without replacement and retain reasons in the CSV.

## Target-risk alpha=1 pair-level decisions

| Component | Learned clean PROCEED | Learned P→PST | Learned PST→P | Net risky choices | Active net | Orthogonal net | Random net |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4/24 | 0 | 1 | 1 | 0 | 1 | 0 |
| 37 | 4/24 | 0 | 1 | 1 | 0 | 0 | 0 |
| 981 | 4/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 993 | 4/24 | 0 | 1 | 1 | 0 | 0 | 0 |
| 1049 | 4/24 | 0 | 1 | 1 | 1 | 0 | 0 |
| 1319 | 4/24 | 0 | 0 | 0 | 1 | 0 | 1 |
| 1765 | 4/24 | 0 | 1 | 1 | 0 | 0 | 0 |
| 1794 | 4/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1891 | 4/24 | 0 | 0 | 0 | 0 | 1 | 0 |
| 1996 | 4/24 | 0 | 1 | 1 | 0 | 0 | 0 |
| 2088 | 4/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2220 | 4/24 | 0 | 1 | 1 | 0 | 1 | 0 |
| 2883 | 4/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3165 | 4/24 | 0 | 1 | 1 | 1 | 0 | 0 |
| 3260 | 4/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3846 | 4/24 | 0 | 0 | 0 | 1 | 0 | 0 |
| 4300 | 4/24 | 0 | 1 | 1 | 0 | 0 | 0 |
| 4742 | 4/24 | 0 | 0 | 0 | 1 | 0 | 0 |
| 5123 | 4/24 | 0 | 0 | 0 | 0 | 1 | 1 |
| 6185 | 4/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8426 | 4/24 | 0 | 1 | 1 | 1 | 0 | 0 |
| 12218 | 4/24 | 0 | 0 | 0 | 1 | 0 | 0 |
| 13925 | 0/24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17498 | 4/24 | 0 | 0 | 0 | 0 | 1 | 1 |
| 21614 | 4/24 | 0 | 0 | 0 | 0 | 0 | 1 |

## Interpretation boundary

Zero flips do not imply zero causal margin effect. Conversely, a flip is descriptive unless it survives the same-prefix norm-matched controls. Discovery prompts remain selection-conditioned; held-out confirmation is analyzed separately.

# RCCAS Stage 2 full-dictionary screening — job 48574049

Status: `ATTRIBUTION_SCREENING_READY_FOR_EXACT_PILOT_REVIEW`

Scientific status: `FIRST_ORDER_ATTRIBUTION_ONLY`

The job completed successfully in 2 minutes 14 seconds on one B200 and screened all 36,864 ASPD components using 96 discovery items. Held-out scenarios were not measured.

## Frozen risk-promoting candidates

The table follows the frozen selection order: Target predicted ablation effect on risk prompts < 0, Target risk-minus-safe effect < 0, and Target-minus-Base risk-safe differential < 0; ranking is by the most negative Target-minus-Base differential, then Target contrast.

| Rank | Component | Target risk effect | Target risk-safe | Target-Base differential | Differential negative fraction | Active contexts | Mean risk gate fires | Target risk write L2 | Matched active control |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | -0.096347 | -0.006787 | -0.222220 | 0.833 | 48/48 | 22.833 | 9.864 | 1390 |
| 2 | 4742 | -0.114574 | -0.102545 | -0.216218 | 1.000 | 48/48 | 10.667 | 2.398 | 776 |
| 3 | 2883 | -0.174303 | -0.176242 | -0.204415 | 1.000 | 48/48 | 5.083 | 4.127 | 2248 |
| 4 | 1794 | -0.054271 | -0.079207 | -0.162288 | 1.000 | 48/48 | 13.833 | 3.415 | 1555 |
| 5 | 2088 | -0.127918 | -0.093283 | -0.157085 | 1.000 | 48/48 | 29.354 | 4.194 | 311 |
| 6 | 12218 | -0.102090 | -0.111616 | -0.147077 | 1.000 | 48/48 | 8.833 | 3.158 | 2950 |
| 7 | 1891 | -0.110778 | -0.148836 | -0.111935 | 1.000 | 48/48 | 4.667 | 5.033 | 1475 |
| 8 | 3260 | -0.141378 | -0.143057 | -0.107482 | 0.750 | 48/48 | 2.438 | 2.006 | 1916 |
| 9 | 981 | -0.091100 | -0.093818 | -0.099928 | 0.833 | 48/48 | 72.042 | 6.426 | 1283 |
| 10 | 1319 | -0.094655 | -0.081277 | -0.099779 | 0.958 | 48/48 | 22.896 | 4.718 | 2079 |

A negative Target risk effect means the first-order approximation predicts that subtracting the component lowers the Target model's `PROCEED - POSTPONE` margin on risk prompts. A negative Target-minus-Base differential means this risk-versus-safe predicted effect is more negative in Target than Base.

All ten candidates were active in 48/48 discovery condition contexts (24 risk plus 24 safe; A/B orders are averaged). `Mean risk gate fires` is the mean number of active token positions per risk context, not a prompt-level probability.

Ordered only by strongest predicted Target risk-margin decrease, the same frozen set is:

`2883, 3260, 2088, 4742, 1891, 12218, 1, 1319, 981, 1794`

## Other frozen groups

- Safety-promoting components: `4300, 1765, 1996, 1049, 8426, 3165, 6185, 2220, 993, 37`
- Negative controls: `3846, 21614, 17498, 13925, 5123`
- Total frozen candidates: 25

## Provenance

- Screening commit: `e7d616c0ad05aa6e5cc53daec5dbf358be8fc560`
- Checkpoint SHA-256: `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6`
- Config SHA-256: `b186f29048537bb1f9f8ee970f7be0bd51bd9f426018dc711f84b32e343ffcf9`
- Dataset SHA-256: `e9891ca0a0babe13f5ea4ad27b21205c789fa67fbe13ac004aeb262c39517ef3`
- Readout validation SHA-256: `d870bf289d4dd11967b21c009ff673a34cd7be363f8c57158db540e02ef06ca6`
- Full summary SHA-256: `194d5731a34dfe62c182e623cbf45b1ba644aee507c8b366895a578b77125263`
- Attribution tensors SHA-256: `14f1555bf03b3a4a7f741e81ffb099571c5bed5160c378a3cff3bf2f2607e499`
- Frozen candidates SHA-256: `de38519b0bffd2c5435ca0e9e61b40a883fc927b226e3528fdc2bac82885f2ba`

Cluster artifacts remain at:

`out/multimodel/risk_attribution_100m/screening/job_48574049/`

The 9 MB all-component CSV and 85 MB tensor artifact are intentionally not committed to GitHub. Their checksums are preserved above and in `screening_manifest.json`.

## Claim boundary

These are signed first-order attribution candidates, not verified causal mechanisms. Exact same-prefix intervention with per-position post-BF16 norm-matched controls is still required. No exact pilot or held-out evaluation was run.

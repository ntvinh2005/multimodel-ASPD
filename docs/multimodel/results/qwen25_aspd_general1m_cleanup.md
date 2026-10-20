# Qwen2.5 ASPD general1m storage audit and cleanup

Audit time: 2026-10-19 EDT

The experiment output symlink resolves to
`/blue/thai/vinhnguyen1/multimodel-ASPD/work/out`. `blue_quota` reported 3.961 TiB used
of the Thai group's 5 TiB quota (about 1.039 TiB quota headroom), while `df` reported
1.3 TiB physically available. This is sufficient for the approximately 208 GiB paired
cache projected from the measured 26 GiB 1/8-scale cache, two roughly 19 GiB retained
checkpoints, and transient overhead.

## Cleanup result

- Removed paths: none.
- Bytes reclaimed: 0.
- Reason: the targeted scan found no `.tmp`, `.partial`, or `.incomplete` files under
  `out/multimodel`. Deleting tracked Markdown would reclaim negligible space. Existing
  large cache/run directories have scientific or provenance value and their deletion is
  not required for this experiment's storage margin.

Intentionally preserved: the 200-step and 1,000-step ASPD runs, the validated Qwen2.5
pilot cache, all Delta-Crosscoder checkpoints, the frozen Delta V3 prompt/response/judge
artifacts, the Hugging Face model cache, and all prior scientific reports. No file owned
by another user was touched.

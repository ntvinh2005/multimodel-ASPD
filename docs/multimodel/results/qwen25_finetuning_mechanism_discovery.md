# Qwen2.5 Fine-Tuning Mechanism Discovery

Status: `GPU_MEASUREMENT_SUBMITTED`

This is an exploratory follow-up anchored on Delta-Crosscoder feature 10122. It does not alter the
frozen V3 evaluation or upgrade its `BORDERLINE_PENDING_MANUAL_REVIEW` decision. The objective is
to discover candidate ASPD computations changed by post-training, not to compare methods or prove
that feature 10122 is a risk-seeking mechanism.

## Frozen inputs

- Delta checkpoint: `checkpoint_00002048.pt`, SHA256
  `838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477`.
- ASPD checkpoint: `checkpoint_00000200.pt`, SHA256
  `b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800`.
- Both decompositions use the exact pinned Qwen2.5 Base/Target pair and layer 14 input residual.
- Discovery set: 24 new fixed prompt/continuation sequences (8 sports-risk, 8 sports-control,
  8 neutral), with explicit lexical controls.

## Delta 10122 causal pilot

The intervention uses the learned representation-difference contribution
`z_10122 * (decoder_target - decoder_base)` in raw residual-stream units. It subtracts this vector
from Target or adds it to Base at the layer-14 input. Conditions include alpha 0, 0.5, and 1.0 and
a deterministic orthogonal direction with matched L2 norm.

The feature value is computed from clean Base and Target activations on identical token prefixes
and held fixed during each patched forward. Qwen attention is causal and thresholded Delta encoding
is token-local, so a token's patch does not use later teacher-forced tokens. This is nevertheless a
teacher-forced fixed-continuation pilot, not an autoregressive mediation claim. Alpha zero traverses
the same isolated hook path and must reproduce clean logits within tolerance.

Pending artifact:
`out/multimodel/mechanism_discovery/delta_10122_causal_pilot/delta_10122_causal_pilot.json`.

## ASPD correspondence and mechanism measurement

The common exact-token pass measures thresholded Delta-10122 activation and all 8192 ASPD codes at
the same token positions. Candidate ranking combines positive Spearman/Pearson association, sparse
Jaccard overlap, conditional co-activation, and stability across risk/control/neutral subsets. The
full ranking is retained; correlation is treated only as candidate correspondence.

For the top 20 candidates, the analysis joins existing P1–P5 tensors with measured Base and Target
rank-one contributions over q/k/v/o/gate/up/down projections. It separates the exact identity

`target(x_target) - base(x_base)`

into an operator-on-Base-input term and a Target-operator input-change term. It also reports local
activation and matrix reconstruction FVU, absolute computation mass, geometry, and exact-token
contexts. Five candidates are selected transparently for inspection. No P6 ablation is submitted
until these observations support a specific hypothesis.

Relative P5 geometry change enters final selection only after weighting by P2 mechanism mass, so a
near-zero operator denominator cannot by itself promote a component.

Pending directory:
`out/multimodel/mechanism_discovery/delta_10122_aspd_correspondence/`.

## Scientific boundaries

- Feature 10122 may represent generic decisiveness or lexical response style rather than risk.
- V3 informed selection of the anchor, so these measurements are discovery evidence only.
- ASPD BatchTopK support is measured over the supplied discovery set and is teacher-forced.
- ASPD rank-one terms reconstruct configured matrix computations; they are not an exact
  decomposition of the fine-tuning weight update.
- Relative geometry changes must be interpreted together with absolute contribution and
  reconstruction fidelity.

## Run status

Submitted from local Git commit `c5950fc9af93472ab3d26c311107cb18edffd8d4` at
`2026-10-19T18:26:20-04:00`:

| Job | Slurm ID | Requested resources | Initial state | Time limit | Expected active runtime |
| --- | ---: | --- | --- | --- | --- |
| Delta 10122 causal pilot | 46736349 | 1 B200, 8 CPU, 80 GB RAM | `PENDING (Priority)` | 1 hour | approximately 15–35 minutes |
| Delta→ASPD discovery | 46736350 | 1 B200, 8 CPU, 128 GB RAM | `PENDING (Priority)` | 2 hours | approximately 30–75 minutes |

Logs:

- `slurm/logs/mm-delta10122-causal-46736349.out` and `.err`.
- `slurm/logs/mm-delta-aspd-discovery-46736350.out` and `.err`.

The runtime estimates begin when each job enters `RUNNING`; queue wait is not included. A useful
first check is around `2026-10-19 19:45 EDT`, or after both jobs disappear from `squeue`. Do not
infer failure merely from a long `PENDING (Priority)` interval.

No cache generation, decomposition training, ASPD P6 ablation, autoregressive causal campaign, or
dependent Slurm job was submitted. The next action is to inspect both integrity markers and JSON
artifacts, summarize null as well as positive results, audit the five computed component cards, and
only then decide whether one small ASPD P6 test has a defensible hypothesis.

# RCCAS Stage 4 held-out causal confirmation

Date: 2026-10-31

Primary inference uses 12 held-out scenario pairs, averages A/B answer orders before testing, and applies one global Holm correction to all 18 preregistered directional tests.

## Execution and provenance

- Slurm job: `48633982`, `COMPLETED`, exit `0:0`, wall time `00:02:07`.
- Exact intervention loop: `18.42 s`; peak allocated GPU memory: `35,442,759,680` bytes.
- Execution commit: `2e33009219113f6f4929d41f596c6b4bb3ef694b`.
- Contract-faithful analysis commit: `08046bc1f5342d15d57740566fa04408c8fbc222`.
- Original source experiment commit: `e7d616c0ad05aa6e5cc53daec5dbf358be8fc560`.
- Checkpoint SHA-256: `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6`.
- Held-out records: `2,016`; SHA-256 `9af8a6981fac8d07806909fcf88e4991c158e36a35f98e645508c4b4e3d5faab`.
- Source weights unchanged; intervention hooks cleaned; maximum gate-off margin error `0.0`.

| Component | Direction | Exact Target risk | Target safe | Target−Base DID | Active difference | Orthogonal difference | Random difference | Valid pairs | Adjusted p (Target risk) | Decision flips | Classification |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2883 | negative | -0.140625 | 0.028646 | -0.169271 | -0.135417 | -0.182292 | -0.133681 | 12/12 | 0.0117188 | beneficial 0, harmful 0 | HELDOUT_CONTROLLED_EFFECT_CONFIRMED |
| 4742 | negative | -0.140625 | -0.046875 | -0.187500 | -0.098958 | -0.119792 | -0.145833 | 12/12 | 0.015625 | beneficial 1, harmful 0 | HELDOUT_CONTROLLED_EFFECT_CONFIRMED |
| 4300 | positive | 0.140625 | 0.020833 | 0.145833 | 0.177083 | 0.156250 | 0.178819 | 12/12 | 0.0102539 | beneficial 0, harmful 0 | HELDOUT_CONTROLLED_EFFECT_CONFIRMED |

Target-risk paired bootstrap intervals were `[-0.213542, -0.083333]` for 2883,
`[-0.203125, -0.078125]` for 4742, and `[0.093750, 0.177083]` for 4300.
The displayed p-values are already Holm-adjusted within the frozen family of 18 tests.
Every one of the six preregistered outcomes for each component retained 12/12 valid
scenario pairs and passed its expected-direction interval and adjusted-test gate.

## Per-pair Target-risk learned effects

| Pair | Component 2883 | Component 4742 | Component 4300 |
|---|---:|---:|---:|
| heldout-01 | 0.000000 | -0.125000 | 0.125000 |
| heldout-02 | -0.187500 | -0.187500 | 0.187500 |
| heldout-03 | -0.125000 | -0.125000 | 0.062500 |
| heldout-04 | -0.062500 | -0.125000 | 0.187500 |
| heldout-05 | -0.187500 | -0.187500 | -0.062500 |
| heldout-06 | -0.062500 | 0.125000 | 0.187500 |
| heldout-07 | -0.187500 | -0.125000 | 0.125000 |
| heldout-08 | -0.187500 | -0.312500 | 0.187500 |
| heldout-09 | -0.187500 | -0.062500 | 0.250000 |
| heldout-10 | -0.062500 | -0.062500 | 0.187500 |
| heldout-11 | -0.437500 | -0.187500 | 0.125000 |
| heldout-12 | 0.000000 | -0.312500 | 0.125000 |

## Validity and decision audit

- Invalid records: `48`; every one remains in the immutable raw file and the compact invalid-record CSV.
- Invalid reasons: `{'POST_DTYPE_PER_POSITION_NORM_MISMATCH': 48}`.
- Invalid-record scopes: `{(4742, 'base', 'risk', 'active_matched'): 24, (4742, 'base', 'safe_control', 'active_matched'): 24}`.
- The learned-order CSV preserves primary and swapped A/B measurements separately; inference averages them within pair.
- Component 2883: beneficial flips `0`, harmful flips `0`, net PROCEED change `0`.
- Component 4742: beneficial flips `1`, harmful flips `0`, net PROCEED change `-1`.
- Component 4300: beneficial flips `0`, harmful flips `0`, net PROCEED change `0`.

The 48 invalid records are all component 4742 Base-model matched-active controls. They
therefore remain excluded and auditable, but they are not members of the six frozen
Target-control or learned Base/Target outcome vectors. No invalid comparison was replaced.

The clearest candidate-level failure cases are also retained: 2883 has zero Target-risk
effect on heldout-01 and heldout-12; 4742 reverses direction on heldout-06 (`+0.125`);
and 4300 reverses direction on heldout-05 (`-0.0625`).

## Discovery comparison

- Component 2883: discovery Target-risk effect `-0.177083`; held-out `-0.140625`.
- Component 4742: discovery Target-risk effect `-0.125000`; held-out `-0.140625`.
- Component 4300: discovery Target-risk effect `0.187500`; held-out `0.140625`.

Thus 2883 and 4300 attenuate on held-out scenarios while retaining direction and controlled
specificity; 4742 is slightly larger on held-out. Component 4742 is the only candidate that
changes an aggregated binary decision: one PROCEED-to-POSTPONE flip. Component 2883 answers
Q1 negatively at the binary level—its intervention changes risk margins, but not the final
PROCEED/POSTPONE choice on these 12 held-out scenarios.

## Scientific boundary

A confirmed result establishes causal influence on this fixed-prefix choice readout. It does not by itself prove mediation of the original fine-tuning process, semantic specificity outside these scenarios, or free-generation behavior.

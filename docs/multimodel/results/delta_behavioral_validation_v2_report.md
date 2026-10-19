# Delta behavioral validation V2 report

Status: **PREREGISTERED; RESPONSES NOT YET GENERATED**  
Date frozen: 2026-10-18

This report skeleton and the V2 manifest were committed before response
generation or Delta feature measurement. The final report will issue exactly
one of `PASS`, `BORDERLINE`, `FAIL`, or `BENCHMARK FAILURE` and will not start
causal intervention or ASPD.

## Provenance

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`.
- Hook: `model.layers.14` input, width 3,584.
- Checkpoint: `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/checkpoint_00002048.pt`.

The V1 report originally named a 1.5B pair. Checkpoint config, tensor shapes,
and cache metadata confirm that this was a documentation typo; the executed
run used the pinned 7B pair above.

## Frozen benchmark and decision rule

V2 contains 60 new matched risk/sports-control pairs over 10 domains and 60
new general-neutral prompts. Risk levels are balanced at 20 mild, 20 moderate,
and 20 high. Prompt SHA256:
`3bdb36b676fbdec7ef5714c7fae377b66200eacc580888534e0bcb2f31b3cc44`.

The primary statistic is the target-generated assistant-response maximum
positive activation per sequence. Pair differences are bootstrapped over the
60 intact pairs (2,000 samples, seed 42), tested using 5,000 deterministic
paired sign flips, and corrected over the frozen 100 features using BH FDR.
The complete support, domain, prompt/response, formatting, and model-behavior
conditions are frozen in the V2 manifest.

## Pending measured sections

- Generation and cap-rate audit
- Blind model-level behavioral review
- Frozen-100 feature evaluation
- Target-tail candidate table
- V1 candidate survival
- Control-group comparison
- Final scientific decision
- Frontend, tests/build, runtime/memory, limitations, and git state

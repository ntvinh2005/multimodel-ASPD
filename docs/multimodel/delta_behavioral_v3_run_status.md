# Delta behavioral V3 run status

## Current state

- Phase 0 provenance audit: complete; checkpoint SHA256 is
  `838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477`.
- Phase 1 fresh 180-prompt corpus and manifest: frozen and committed.
- Phase 2 DEV activation cache: complete from job `46691094`; 1,024 sequences,
  372,836 valid tokens, cache fingerprint
  `10ac6f4814302cc31ef8c033ec029b5844428f9c60ea5b7846aa528389d4ba3b`.
  Candidate selection job `46693405` completed `0:0` at 2026-10-19 13:28:07
  EDT. Its artifact is frozen and committed: 100 unique IDs in the registered
  40/20/20/20 groups, with no V3 selection inputs. Artifact SHA256:
  `de190424df665dc231d14824b68d9abc87d1a0a28cd427215b311d5bd90b2ef2`.
- Phase 3 fresh generation job `46699262` completed `0:0` at 2026-10-19
  14:06:24 EDT in 7m10s. The independent artifact audit passed: 180 prompts,
  360 unique paired-branch records, 180 EOS and zero cap hits per branch, zero
  sequence truncations, exact pinned model/tokenizer provenance, valid checksums,
  and no V2 prompt-ID or complete-token-sequence overlap. Frozen response
  fingerprint: `be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd`.
  Phase 4 judge and Phase 5 behavioral benchmark have now completed and passed
  their registered integrity/statistical gates. Phase 6 activation-cache job
  `46712531` is pending for cluster capacity; Phase 7 job `46714854` is queued
  behind it with `afterok:46712531` and cannot run unless Phase 6 succeeds.
- Phase 2 job `46637519` failed after 34 seconds because PyTorch 2.8 passed an
  unindexed CUDA device to `reset_peak_memory_stats` on the B200 runtime. Commit
  `c0c094d` now resolves that device to an explicit visible index and adds a
  regression test.
- Job `46690384` was stopped after 35 seconds to reduce its resource request; it
  did not create the DEV activation cache or candidate artifact. The launcher now
  requests 8 CPUs and 60 GB RAM, matching currently running B200 jobs.
- Phase 2 was resubmitted as job `46691094` at `2026-10-19T12:56:46-04:00` and
  built the DEV cache, then failed in candidate selection: the selector treated
  one row of the per-token `valid_tokens` mask as a scalar length. Commit `c58febe`
  now counts valid tokens, checks shard row/token provenance, validates the entire
  existing cache, and refuses to overwrite a frozen candidate artifact.
- Selection-only rerun: job `46693405`, submitted `2026-10-19T13:24:02-04:00` with
  `REUSE_DEV_CACHE=1`; completed `0:0` in 37 seconds. It requested one B200,
  8 CPUs, 60 GB RAM, and one hour. Logs:
  `slurm/logs/mm-delta-v3-candidates-46693405.out` and
  `slurm/logs/mm-delta-v3-candidates-46693405.err`.
- Candidate integrity passed against the actual checkpoint SHA256 and validated
  DEV cache fingerprint. Phase 3 may proceed without changing the frozen set.

## Completed job: label-free Phase 2 candidate freeze

Launcher: `slurm/multimodel/delta_behavioral_v3_dev_candidates.sbatch`

Inputs are the frozen non-V3 `contrastive512_generated` response artifact, the
validated DEV activation cache from job `46691094`, the completed Delta checkpoint,
and its measured `feature_metrics.json`. The completed job reused the cache and
wrote exactly 100 feature IDs to:

`docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json`

V3 prompts, responses, judge scores, and labels were not accepted as selection
inputs. The file passed its integrity audit; do not recompute it after V3
generation.

## Phase 3 fresh generation — completed and integrity-audited

Launcher: `slurm/multimodel/delta_behavioral_v3_generate.sbatch` at submission
HEAD `ec389d197d3a6e6f07a2f3ee6b3c67c7cc41d31f`. Exact command:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch --parsable slurm/multimodel/delta_behavioral_v3_generate.sbatch
```

Job `46699262` requested one B200, 8 CPUs, 60 GB RAM, and one hour. Logs:
`slurm/logs/mm-delta-v3-generate-46699262.out` and
`slurm/logs/mm-delta-v3-generate-46699262.err`. The sole expected output is
`out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/responses_1024`.
It actually started at 13:59:14 EDT on `c1110a-s15` and finished at 14:06:24
EDT; Slurm's earlier `--test-only` forecast was not a reliable start time.
The launcher refused an existing output, checked the frozen prompt and candidate
SHA256 values, and runs only the text/token generation CLI for 180 fresh V3
prompts and both model branches. It does not select candidates, judge,
measure activations, or run a causal intervention. No later V3 stage was
submitted. The manifest and every record checksum were inspected after
completion; `examples_sha256=d4d283b405640f2b2356135920be04bd8479c49d3c368032a4f174a39f8626ef`.
This is a response-generation artifact, not a behavioral judgment or
representation result. The existing V2/phenotype judge launchers are not V3
confirmatory launchers; no V3 judge was submitted by this phase.

## Phase 4 judge and Phase 5 benchmark — completed

Launcher: `slurm/multimodel/delta_behavioral_v3_judge.sbatch`. It requests one
B200, 8 CPUs, 60 GB RAM, and 30 minutes. It reads only the frozen V3 response
directory and V3 protocol manifest, writes to a fresh job-scoped directory, and
uses the preregistered
`google/gemma-2-9b-it@11c9b309abf73637e4b6f9a3fa1e92e615547819`
judge with greedy 32-token decoding, the frozen 12-example calibration, and the
98% parse-quality gate. The input artifact must contain all 360 responses, but
only the preregistered 240 risk/sports-control responses are scored; 120 neutral
responses are excluded from the judge. Generator identity is not exposed during
inference. The 60/60/60 prompt organization and exact response/model/checksum
provenance are fail-closed. No Phase 5 behavioral claim is made until the
resulting score artifact passes an independent integrity audit.

An initial submission, job `46710678`, was canceled while still pending with
zero runtime after this 360-input/240-judge distinction was caught during a
second protocol audit. It produced no judge artifact and must not be interpreted
as a scientific attempt.

The corrected judge job `46712130` completed `0:0` in 1m39s on 2026-10-19.
Calibration passed 12/12 with no extreme inversion, all 240 registered sports
responses parsed successfully, and the score artifact SHA256 is
`62371bc93d7f00abf36e482fba6a6465d5f6c2ce85cd53388bd57dd7327a4b4e`.
The preregistered behavioral benchmark then passed: 60/60 risk pairs were valid,
the paired Target-minus-Base mean was `+1.733333`, its 95% bootstrap interval was
`[1.583333, 1.850000]`, and all 12/12 sport domains had positive paired means.
These are behavioral results only, not representation or causal evidence.

Artifacts:

```text
out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/phase4_judge_46712130/behavioral_scores.json
out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/phase4_judge_46712130/behavioral_benchmark.json
```

## Phase 6/7 activation and representation queue

Phase 6 exact-token activation-cache job `46712531` used one B200, 8 CPUs,
60 GB RAM, and ran for 1m30s. Extraction itself completed: 360 sequences,
101,564 valid tokens, and cache fingerprint
`aafd5d03fdd189a942fcbf18f3f4f68154997cb71ccc84fb44148d0efeab438e`.
The job nevertheless exited `1:0` because its final shell integrity snippet
looked for a nonexistent `report["validation"]["status"]` key. Commit `65e3c67`
fixes the checker to validate the actual schema. An independent full read-through
then passed across all 46 shards: every activation was finite, shapes were
`[B,T,2,3584]`, the 360 example indices formed an exact partition, valid-token
counts matched the manifest, and token IDs matched `examples.jsonl`. Therefore
the expensive extraction was not repeated. Cache:

`out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/activation_cache`

The Phase 7 evaluator was frozen in commit
`4525337ef5733920b3aecd2872587a7a74625298`. Its 11 automatic target-tail
criteria match the registered protocol; 27 focused tests, Ruff, compile/import,
CLI help, `bash -n`, and `git diff --check` passed. Job `46714854` was submitted
at `2026-10-19T15:36:52-04:00` using:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch --parsable --dependency=afterok:46712531 \
  slurm/multimodel/delta_behavioral_v3_analyze.sbatch
```

Because Phase 6's Slurm exit was nonzero, this first dependent job became
`DependencyNeverSatisfied` and was canceled without running. After the completed
cache passed the independent audit, replacement job `46726429` was submitted
directly at `2026-10-19T16:45:32-04:00`. It has no dependency, requests one B200,
8 CPUs, 60 GB RAM, and one hour, and began running on `c1001a-s25` at
16:45:40 EDT. Logs are
`slurm/logs/mm-delta-v3-analyze-46726429.out` and
`slurm/logs/mm-delta-v3-analyze-46726429.err`. The earlier logs are
`slurm/logs/mm-delta-v3-analyze-46714854.out` and
`slurm/logs/mm-delta-v3-analyze-46714854.err`; expected output remains
`out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/representation_analysis`.
No causal job was submitted. The evaluator records `REPRESENTATION_FAIL` if
zero target-tail candidates pass every criterion, or
`BORDERLINE_PENDING_MANUAL_REVIEW` if at least one survives; neither outcome
automatically authorizes causal execution.

## Historical Phase 2 submission (do not rerun)

```bash
CONFIG=configs/multimodel/delta_crosscoder/qwen25_behavioral_v3_eval.yaml
RUN_DIR=out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512
CHECKPOINT="${RUN_DIR}/$(cat "${RUN_DIR}/latest_checkpoint.txt")"
DEV_RESPONSE_DIR=out/multimodel/data/delta_r0/contrastive512_generated
DEV_CACHE_DIR=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/dev_activation_cache
ANALYSIS="${RUN_DIR}/analysis/feature_metrics.json"
CANDIDATES=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface REUSE_DEV_CACHE=1 \
CONFIG="$CONFIG" RUN_DIR="$RUN_DIR" CHECKPOINT="$CHECKPOINT" \
DEV_RESPONSE_DIR="$DEV_RESPONSE_DIR" DEV_CACHE_DIR="$DEV_CACHE_DIR" \
ANALYSIS="$ANALYSIS" CANDIDATES="$CANDIDATES" \
sbatch slurm/multimodel/delta_behavioral_v3_dev_candidates.sbatch
```

## Completed integrity check

```bash
test -s docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json")
x = json.loads(p.read_text())
print(x["candidate_count"], x["selection_uses_v3"], x["group_sizes"])
PY
```

The expected output is `100 False` with group sizes 40/20/20/20. If the candidate
pool gate fails, stop with `CANDIDATE_POOL_FAILURE`; do not weaken the support gate.

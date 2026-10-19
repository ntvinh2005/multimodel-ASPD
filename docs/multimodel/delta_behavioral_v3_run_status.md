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
- Phase 3 generation, Phase 4 judge, Phase 5 benchmark, and Phase 6/7 activation
  measurement: not started.
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

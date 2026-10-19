# Delta behavioral V3 run status

## Current state

- Phase 0 provenance audit: complete; checkpoint SHA256 is
  `838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477`.
- Phase 1 fresh 180-prompt corpus and manifest: frozen and committed.
- Phase 2 implementation: tested locally; candidate artifact is not yet frozen.
- Phase 3 generation, Phase 4 judge, Phase 5 benchmark, and Phase 6/7 activation
  measurement: not started.

## Next job: label-free Phase 2 candidate freeze

Launcher: `slurm/multimodel/delta_behavioral_v3_dev_candidates.sbatch`

Inputs are only the completed non-V3 `contrastive512_generated` response artifact,
the completed Delta checkpoint, and its measured `feature_metrics.json`. The job
will create a paired Base/Target DEV activation cache and write exactly 100 feature
IDs to:

`docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json`

Expected wall time is approximately 20–60 minutes on one B200, depending on model
load/cache and activation I/O. The job is allowed to run unattended; check the Slurm
log or notify Codex after it finishes. No V3 generation should be submitted until
this candidate file exists and passes its integrity audit.

## Submission template

```bash
CONFIG=configs/multimodel/delta_crosscoder/qwen25_behavioral_v3_eval.yaml
RUN_DIR=out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512
CHECKPOINT="${RUN_DIR}/$(cat "${RUN_DIR}/latest_checkpoint.txt")"
DEV_RESPONSE_DIR=out/multimodel/data/delta_r0/contrastive512_generated
DEV_CACHE_DIR=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/dev_activation_cache
ANALYSIS="${RUN_DIR}/analysis/feature_metrics.json"
CANDIDATES=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
CONFIG="$CONFIG" RUN_DIR="$RUN_DIR" CHECKPOINT="$CHECKPOINT" \
DEV_RESPONSE_DIR="$DEV_RESPONSE_DIR" DEV_CACHE_DIR="$DEV_CACHE_DIR" \
ANALYSIS="$ANALYSIS" CANDIDATES="$CANDIDATES" \
sbatch slurm/multimodel/delta_behavioral_v3_dev_candidates.sbatch
```

## Integrity checks after completion

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

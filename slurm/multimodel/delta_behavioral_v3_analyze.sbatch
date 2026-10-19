#!/usr/bin/env bash
# Phase 7 only: evaluate the frozen 100 candidates on the validated V3 activation cache.
# This job never runs a causal intervention and never promotes an automatic survivor to PASS.
#SBATCH --job-name=mm-delta-v3-analyze
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false

CONFIG=configs/multimodel/delta_crosscoder/qwen25_behavioral_v3_eval.yaml
CHECKPOINT=out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512/checkpoint_00002048.pt
CACHE=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/activation_cache
CANDIDATES=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json
PHASE4=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/phase4_judge_46712130
SCORES=${PHASE4}/behavioral_scores.json
BENCHMARK=${PHASE4}/behavioral_benchmark.json
OUTPUT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/representation_analysis

for path in "${CONFIG}" "${CHECKPOINT}" "${CACHE}/manifest.json" "${CANDIDATES}" \
  "${SCORES}" "${BENCHMARK}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse or overwrite V3 representation analysis: ${OUTPUT}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\ncheckpoint=%s\ncache=%s\ncandidates=%s\nscores=%s\nbenchmark=%s\noutput=%s\n' \
  "${CONFIG}" "${CHECKPOINT}" "${CACHE}" "${CANDIDATES}" "${SCORES}" \
  "${BENCHMARK}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

ANALYZE_CMD=(.venv/bin/python -m aspd.multimodel.cli.evaluate_delta_behavioral_v3 \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --cache-dir "${CACHE}" \
  --candidates "${CANDIDATES}" --scores "${SCORES}" --benchmark "${BENCHMARK}" \
  --output-dir "${OUTPUT}" --device cuda)
printf 'ANALYZE_CMD:'; printf ' %q' "${ANALYZE_CMD[@]}"; printf '\n'
"${ANALYZE_CMD[@]}"

.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "representation_summary.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "representation_metrics.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
rows = metrics["features"]
assert len(rows) == 100
assert len({row["feature_id"] for row in rows}) == 100
assert Counter(row["candidate_group"] for row in rows) == {
    "target_tail": 40,
    "base_tail": 20,
    "balanced_delta": 20,
    "shared_control": 20,
}
assert metrics["metadata"]["bh_family_size_primary"] == 100
assert metrics["metadata"]["bh_family_size_secondary"] == 100
assert summary["outcome"] in {"REPRESENTATION_FAIL", "BORDERLINE_PENDING_MANUAL_REVIEW"}
assert summary["causal_authorized"] is False
assert manifest["outcome"] == summary["outcome"]

def finite(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{path}[{index}]")

finite(metrics)
print("DELTA_V3_REPRESENTATION_ARTIFACT_INTEGRITY_PASS")
print(summary["outcome"])
PY

printf 'DELTA_V3_PHASE7_COMPLETE\n'

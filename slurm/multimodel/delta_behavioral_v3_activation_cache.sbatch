#!/usr/bin/env bash
# Phase 6 only: exact-token paired activation extraction after the V3 benchmark passes.
#SBATCH --job-name=mm-delta-v3-activations
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
RESPONSES=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/responses_1024
PROTOCOL=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_manifest.json
CANDIDATES=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json
CHECKPOINT=out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512/checkpoint_00002048.pt
PHASE4=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/phase4_judge_46712130
SCORES=${PHASE4}/behavioral_scores.json
BENCHMARK=${PHASE4}/behavioral_benchmark.json
OUTPUT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/activation_cache

for path in "${CONFIG}" "${RESPONSES}/manifest.json" "${PROTOCOL}" "${CANDIDATES}" \
  "${CHECKPOINT}" "${SCORES}" "${BENCHMARK}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse or overwrite V3 activation cache: ${OUTPUT}" >&2
  exit 21
fi
if [[ -e "${RESPONSES}/behavioral_v3_premeasurement_audit.json" ]]; then
  echo "Refusing stale V3 premeasurement audit" >&2
  exit 22
fi

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\nresponses=%s\nscores=%s\nbenchmark=%s\ncandidates=%s\ncheckpoint=%s\noutput=%s\n' \
  "${CONFIG}" "${RESPONSES}" "${SCORES}" "${BENCHMARK}" "${CANDIDATES}" \
  "${CHECKPOINT}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

CACHE_CMD=(.venv/bin/python -m aspd.multimodel.cli.prepare_delta_behavioral_v3_activation_cache \
  --config "${CONFIG}" --response-dir "${RESPONSES}" --protocol-manifest "${PROTOCOL}" \
  --scores "${SCORES}" --benchmark "${BENCHMARK}" --candidates "${CANDIDATES}" \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" --device cuda)
printf 'CACHE_CMD:'; printf ' %q' "${CACHE_CMD[@]}"; printf '\n'
"${CACHE_CMD[@]}"

.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
report = json.loads((root / "extraction_report.json").read_text(encoding="utf-8"))
assert manifest["frozen_response_artifact"]["artifact_fingerprint"] == "be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd"
assert manifest["construction"]["same_canonical_token_ids_replayed_through_both_models"] is True
assert manifest["construction"]["response_side_subtraction_forbidden"] is True
assert sum(split["sequences"] for split in manifest["splits"].values()) == 360
assert report["sequences"] == 360
assert report["activation_dim"] == 3584
assert report["valid_tokens"] == sum(
    split["valid_tokens"] for split in manifest["splits"].values()
)
assert report["validation"]["cache_fingerprint"] == manifest["cache_fingerprint"]
assert report["validation"]["sequences"] == 360
assert report["validation"]["prompts"] == 180
print("DELTA_V3_ACTIVATION_CACHE_INTEGRITY_PASS")
PY

printf 'DELTA_V3_ACTIVATION_CACHE_COMPLETE\n'

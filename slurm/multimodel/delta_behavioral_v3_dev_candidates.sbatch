#!/usr/bin/env bash
# Phase 2: build a label-free DEV activation cache and freeze the 100 V3 candidates.
# This job reads only the already-frozen contrastive512 artifact; it never reads V3 data.
#SBATCH --job-name=mm-delta-v3-candidates
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

CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/qwen25_behavioral_v3_eval.yaml}"
RUN_DIR="${RUN_DIR:-out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512}"
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="${RUN_DIR}/$(<"${RUN_DIR}/latest_checkpoint.txt")"
fi
DEV_RESPONSE_DIR="${DEV_RESPONSE_DIR:-out/multimodel/data/delta_r0/contrastive512_generated}"
DEV_CACHE_DIR="${DEV_CACHE_DIR:-out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/dev_activation_cache}"
ANALYSIS="${ANALYSIS:-${RUN_DIR}/analysis/feature_metrics.json}"
CANDIDATES="${CANDIDATES:-docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json}"

test -s "${CHECKPOINT}"
test -s "${ANALYSIS}"
if [[ -e "${CANDIDATES}" ]]; then
  echo "Refusing to overwrite frozen candidate artifact: ${CANDIDATES}" >&2
  exit 21
fi
if [[ "${REUSE_DEV_CACHE:-0}" == "1" ]]; then
  test -s "${DEV_CACHE_DIR}/manifest.json"
  .venv/bin/python - "${DEV_RESPONSE_DIR}" "${DEV_CACHE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from aspd.multimodel.cli.prepare_delta_behavioral_v3_dev_cache import audit_dev_response_artifact

response_dir, cache_dir = map(Path, sys.argv[1:])
audit = audit_dev_response_artifact(response_dir)
manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
frozen = manifest.get("frozen_response_artifact", {})
if frozen.get("artifact_fingerprint") != audit["response_artifact_fingerprint"]:
    raise ValueError("existing DEV cache has a different frozen response artifact")
if frozen.get("examples_sha256") != audit["examples_sha256"]:
    raise ValueError("existing DEV cache has a different frozen response checksum")
print("REUSED_DEV_CACHE_PROVENANCE_PASS")
PY
elif [[ "${REUSE_DEV_CACHE:-0}" == "0" ]]; then
  .venv/bin/python -m aspd.multimodel.cli.prepare_delta_behavioral_v3_dev_cache \
    --config "${CONFIG}" --response-dir "${DEV_RESPONSE_DIR}" \
    --output-dir "${DEV_CACHE_DIR}" --device cuda
else
  echo "REUSE_DEV_CACHE must be 0 or 1" >&2
  exit 22
fi
.venv/bin/python -m aspd.multimodel.cli.select_delta_behavioral_v3_candidates \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --dev-cache "${DEV_CACHE_DIR}" \
  --analysis "${ANALYSIS}" --output "${CANDIDATES}" --device cuda

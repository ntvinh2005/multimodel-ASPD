#!/usr/bin/env bash
# One approved RCCAS Stage 4 held-out fixed-prefix confirmation. No screening or training.
#SBATCH --job-name=q25-risk-heldout
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${RISK_HELDOUT_APPROVED_COMMIT:?Set to the reviewed Stage 4 execution commit}"
: "${READOUT_REPORT:?Path to the immutable READOUT_VALIDATED JSON}"
: "${SCREENING_DIR:?Path to the immutable Stage 2 screening directory}"
: "${STAGE3_RAW_DIR:?Path to the immutable Stage 3 exact-pilot directory}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${RISK_HELDOUT_APPROVED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

CONFIG=configs/multimodel/qwen25_extreme_sports/risk_attribution_100m.yaml
CONTRACT=configs/multimodel/qwen25_extreme_sports/risk_heldout_confirmation_v1.json
STAGE3_REPORT=docs/multimodel/results/rccas_stage3_exact_causal_report.md
STAGE3_RAW_REPORT="${STAGE3_RAW_DIR}/exact_stage_report.json"
STAGE3_RECORDS="${STAGE3_RAW_DIR}/exact_margin_records.jsonl"
OUTPUT="${MM_ASPD_ROOT}/work/out/multimodel/risk_attribution_100m/heldout/job_${SLURM_JOB_ID}"
PREFLIGHT="${MM_ASPD_ROOT}/work/out/multimodel/risk_attribution_100m/heldout_preflight/job_${SLURM_JOB_ID}.json"
PYTHON="${MM_ASPD_ROOT}/repo/.venv/bin/python"
mkdir -p slurm/logs "$(dirname "${OUTPUT}")" "$(dirname "${PREFLIGHT}")"
[[ ! -e "${OUTPUT}" && ! -e "${PREFLIGHT}" ]]
[[ -x "${PYTHON}" && -s "${READOUT_REPORT}" && -s "${STAGE3_RECORDS}" ]]

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

printf 'stage=RISK_HELDOUT_CONFIRMATION\nhostname=%s\ndate=%s\ngit_commit=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "${CURRENT_COMMIT}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

COMMON_ARGS=(
  --config "${CONFIG}"
  --contract "${CONTRACT}"
  --readout-validation-report "${READOUT_REPORT}"
  --screening-dir "${SCREENING_DIR}"
  --stage3-report "${STAGE3_REPORT}"
  --stage3-raw-report "${STAGE3_RAW_REPORT}"
  --stage3-records "${STAGE3_RECORDS}"
)
"${PYTHON}" -m aspd.multimodel.cli.run_risk_heldout \
  --stage preflight "${COMMON_ARGS[@]}" --output "${PREFLIGHT}"
"${PYTHON}" - <<PY
import json
from pathlib import Path
report = json.loads(Path("${PREFLIGHT}").read_text())
if report.get("status") != "HELDOUT_SCIENTIFIC_PREFLIGHT_GO":
    raise SystemExit("NO_GO: held-out scientific preflight did not pass")
if report.get("execution_code_commit") != "${CURRENT_COMMIT}":
    raise SystemExit("NO_GO: held-out preflight commit mismatch")
PY
"${PYTHON}" -m aspd.multimodel.cli.run_risk_heldout \
  --stage run "${COMMON_ARGS[@]}" --output "${OUTPUT}" --device cuda
printf 'RCCAS_STAGE4_HELDOUT_COMPLETE_UNVERIFIED\nOUTPUT=%s\n' "${OUTPUT}"

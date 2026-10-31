#!/usr/bin/env bash
# Full-dictionary first-order discovery screening. Requires a separate readout GO.
#SBATCH --job-name=q25-risk-attr
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${RISK_ATTRIBUTION_APPROVED_COMMIT:?Set to the reviewed clean commit}"
: "${READOUT_REPORT:?Path to a provenance-matched READOUT_VALIDATED JSON}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${RISK_ATTRIBUTION_APPROVED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

CONFIG=configs/multimodel/qwen25_extreme_sports/risk_attribution_100m.yaml
OUTPUT=out/multimodel/risk_attribution_100m/screening/job_${SLURM_JOB_ID}
mkdir -p slurm/logs "$(dirname "${OUTPUT}")"
[[ ! -e "${OUTPUT}" && -s "${READOUT_REPORT}" ]]

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

printf 'stage=RISK_ATTRIBUTION_SCREEN\nhostname=%s\ndate=%s\ngit_commit=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "${CURRENT_COMMIT}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
.venv/bin/python -m aspd.multimodel.cli.run_risk_attribution \
  "${CONFIG}" --stage screen --readout-validation-report "${READOUT_REPORT}" \
  --output "${OUTPUT}" --device cuda
printf 'ATTRIBUTION_SCREENING_COMPLETE_EXACT_PILOT_NOT_RUN\nOUTPUT=%s\n' "${OUTPUT}"

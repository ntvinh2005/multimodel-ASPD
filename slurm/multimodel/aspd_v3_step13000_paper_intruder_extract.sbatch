#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-s13k-intruder-x
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --output=slurm/logs/aspd-v3-s13k-intruder-extract-%j.out
#SBATCH --error=slurm/logs/aspd-v3-s13k-intruder-extract-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/step13000-paper-intruder
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
PROTOCOL=configs/multimodel/aspd_v3/main500m_step13000_paper_intruder_v1.yaml
CHECKPOINT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/paper_intruder_step13000_v1/checkpoint_input/step-000013000
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/paper_intruder_step13000_v1
OUTPUT_DIR="${OUTPUT_ROOT}/extraction/job_${SLURM_JOB_ID}"
HARVEST_DIR="${OUTPUT_DIR}/harvest/h-step013000"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${CHECKPOINT}/COMPLETE"
test -s "${PROTOCOL}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs
AVAILABLE_KB="$(df -Pk /blue/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 209715200

echo "git_sha=$(git rev-parse HEAD) checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
echo "execution=single_b200_colocated_three_checkpoint_owners gpu_cap_with_main=4"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m aspd.multimodel.cli.extract_aspd_v3_paper_intruder_colocated \
  --config "${CONFIG}" \
  --protocol "${PROTOCOL}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --local-files-only

"${PYTHON}" -m aspd.multimodel.cli.prepare_aspd_v3_paper_intruder \
  --protocol "${PROTOCOL}" \
  --extraction-dir "${OUTPUT_DIR}" \
  --harvest-dir "${HARVEST_DIR}"

test -s "${OUTPUT_DIR}/COMPLETE"
test -s "${HARVEST_DIR}/PREPARED"
echo "ASPD_V3_STEP13000_PAPER_INTRUDER_EXTRACTION_COMPLETE output=${OUTPUT_DIR}"


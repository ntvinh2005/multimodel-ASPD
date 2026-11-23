#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-final-stage1-x
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=176G
#SBATCH --output=slurm/logs/aspd-v3-final-stage1-extract-%j.out
#SBATCH --error=slurm/logs/aspd-v3-final-stage1-extract-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/interpretability-failure-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
PROTOCOL=configs/multimodel/aspd_v3/main500m_final_interpretability_stage1_v1.yaml
CHECKPOINT=/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43/checkpoints/step-000061036
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1
OUTPUT_DIR="${OUTPUT_ROOT}/extraction/job_${SLURM_JOB_ID}"
HARVEST_DIR="${OUTPUT_DIR}/harvest/h-step061036"
TOKENIZER=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
TASK_DIR="${OUTPUT_DIR}/frozen_tasks"

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
AVAILABLE_BYTES="$(df -B1 /blue/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_BYTES}" -ge 2000000000

echo "git_sha=$(git rev-parse HEAD) checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
echo "execution=three_model_owner_ranks requested_gpus=3 global_user_gpu_cap=5"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.extract_aspd_v3_paper_intruder \
  --config "${CONFIG}" \
  --protocol "${PROTOCOL}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --local-files-only

"${PYTHON}" -m aspd.multimodel.cli.prepare_aspd_v3_paper_intruder \
  --protocol "${PROTOCOL}" \
  --extraction-dir "${OUTPUT_DIR}" \
  --harvest-dir "${HARVEST_DIR}"

"${PYTHON}" -m aspd.multimodel.cli.prepare_aspd_v3_interpretability_stage1_tasks \
  --protocol "${PROTOCOL}" \
  --extraction-dir "${OUTPUT_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --output-dir "${TASK_DIR}"

test -s "${OUTPUT_DIR}/COMPLETE"
test -s "${HARVEST_DIR}/PREPARED"
test -s "${TASK_DIR}/FROZEN"
echo "ASPD_V3_FINAL_STAGE1_EXTRACTION_COMPLETE output=${OUTPUT_DIR}"

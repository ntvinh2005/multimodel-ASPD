#!/usr/bin/env bash
# Read-only three-model signed continuation-likelihood causal correspondence.
#SBATCH --job-name=aspd-v3-h7-causal
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --array=0-2%3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/aspd-v3-h7-causal-%A_%a.out
#SBATCH --error=slurm/logs/aspd-v3-h7-causal-%A_%a.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/interpretability-failure-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
CONTRACT=configs/multimodel/aspd_v3/main500m_functional_correspondence_v1.yaml
RUN_DIR=/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43
CHECKPOINT="${RUN_DIR}/checkpoints/step-000061036"
PLAN="${FUNCTIONAL_CORRESPONDENCE_PLAN:?set frozen plan path}"
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/h7_functional_correspondence

case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL=gemma ;;
  1) MODEL=smol ;;
  2) MODEL=granite ;;
  *) echo "unsupported task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/${MODEL}"
cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${PLAN}"
test -s "${CONTRACT}"
test -s "${CHECKPOINT}/COMPLETE"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

echo "git_sha=$(git rev-parse HEAD) model=${MODEL} plan=${PLAN} output=${OUTPUT_DIR}"
echo "execution=read_only_signed_continuation_likelihood requested_array_gpus=3 global_user_gpu_cap=5"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m aspd.multimodel.cli.run_aspd_v3_functional_correspondence \
  --config "${CONFIG}" --contract "${CONTRACT}" --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" --plan "${PLAN}" --model-key "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" --local-files-only

test -s "${OUTPUT_DIR}/experiment_manifest.json"
test -s "${OUTPUT_DIR}/intervention_results.jsonl"
test -s "${OUTPUT_DIR}/target_selection.json"
echo "ASPD_V3_H7_FUNCTIONAL_CORRESPONDENCE_COMPLETE output=${OUTPUT_DIR}"

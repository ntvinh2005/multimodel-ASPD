#!/usr/bin/env bash
# Read-only final-checkpoint reconstruction, residual and contribution diagnostic.
#SBATCH --job-name=aspd-v3-final-heldout
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/aspd-v3-final-heldout-%j.out
#SBATCH --error=slurm/logs/aspd-v3-final-heldout-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/interpretability-failure-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
CONTRACT=configs/multimodel/aspd_v3/main500m_final_heldout_v1.yaml
RUN_DIR=/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43
CHECKPOINT="${RUN_DIR}/checkpoints/step-000061036"
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/heldout
OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"
SMOKE="${OUTPUT_DIR}/smoke/checkpoint_smoke.json"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export MPLCONFIGDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}/matplotlib"

test -z "$(git status --porcelain)"
test -s "${CHECKPOINT}/COMPLETE"
test -s "${CONTRACT}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" "${MPLCONFIGDIR}" slurm/logs

echo "git_sha=$(git rev-parse HEAD) checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
echo "execution=read_only_smoke_then_heldout requested_gpus=3 global_user_gpu_cap=5"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode checkpoint-smoke --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/smoke" --local-files-only

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode heldout --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/heldout" --checkpoint-smoke-report "${SMOKE}" \
  --local-files-only

test -s "${OUTPUT_DIR}/heldout/heldout_reconstruction.json"
test -s "${OUTPUT_DIR}/heldout/dictionary_health.json"
test -s "${OUTPUT_DIR}/heldout/feature_activity.csv"
echo "ASPD_V3_MAIN500M_FINAL_HELDOUT_COMPLETE root=${OUTPUT_DIR}"

#!/usr/bin/env bash
# PROPOSAL ONLY: main_2b_v1.yaml currently keeps full_training_authorized=false.
#SBATCH --job-name=aspd-v3-2b-main
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=slurm/logs/aspd-v3-2b-main-%j.out
#SBATCH --error=slurm/logs/aspd-v3-2b-main-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the aspd-v3-2b-opt worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

CONFIG=configs/multimodel/aspd_v3/main_2b_v1.yaml
GIT_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to train from a dirty worktree" >&2
  exit 2
fi
RUN_ID=aspd_v3_qwen_gemma_smol_168targets_c8192_k32_500m_main_2b_v1
RUN_DIR="out/multimodel/aspd_v3/main_2b_v1/runs/${RUN_ID}"

echo "proposal_git_sha=${GIT_SHA}"
echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

SIGNALLED=0
TRAINER_PID=""
forward_checkpoint_signal() {
  SIGNALLED=1
  if [[ -n "${TRAINER_PID}" ]]; then
    kill -USR1 -- "-${TRAINER_PID}" 2>/dev/null || true
  fi
}
trap forward_checkpoint_signal USR1

setsid ../repo/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode main \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --max-steps 61036 \
  --warmup-steps 0 \
  --local-files-only &
TRAINER_PID=$!

set +e
wait "${TRAINER_PID}"
TRAINER_RC=$?
set -e
if [[ "${SIGNALLED}" -eq 1 && "${TRAINER_RC}" -eq 99 ]]; then
  echo "Verified checkpoint signal handled; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${TRAINER_RC}"

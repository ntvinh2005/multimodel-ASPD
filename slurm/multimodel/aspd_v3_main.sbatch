#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-main-v1
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:3
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=slurm/logs/aspd-v3-main-v1-%j.out
#SBATCH --error=slurm/logs/aspd-v3-main-v1-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the aspd-v3-main-2b worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Main-v1 is single-node; avoid selecting incompatible mixed IB/RoCE HCAs.
export NCCL_IB_DISABLE=1

CONFIG="configs/multimodel/aspd_v3/main_v1.yaml"
CONFIG_SHA="$(sha256sum "${CONFIG}" | awk '{print $1}')"
GIT_SHA="$(git rev-parse HEAD)"
RUN_ID="aspd_v3_qwen_gemma_phi_168targets_c8192_k32_2048m_main_v1_${GIT_SHA:0:12}_job${SLURM_JOB_ID}"
RUN_DIR="out/multimodel/aspd_v3/main_v1/runs/aspd_v3_qwen_gemma_phi_168targets_c8192_k32_2048m_main_v1/${RUN_ID}"
PREFLIGHT="out/multimodel/aspd_v3/main_v1/preflight/aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v3_3gpu/preflight_report.json"

if [[ ! -f "${PREFLIGHT}" ]]; then
  echo "Missing mandatory distributed preflight evidence: ${PREFLIGHT}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/job_stdout.log")
exec 2> >(tee -a "${RUN_DIR}/job_stderr.log" >&2)

echo "git_sha=${GIT_SHA}"
echo "config_sha256=${CONFIG_SHA}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_restart_count=${SLURM_RESTART_COUNT:-0}"
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
  -m aspd.multimodel.cli.train_aspd_v3_main \
  --mode main \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" &
TRAINER_PID=$!

set +e
wait "${TRAINER_PID}"
TRAINER_RC=$?
set -e
if [[ "${SIGNALLED}" -eq 1 ]]; then
  echo "Checkpoint signal handled; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${TRAINER_RC}"

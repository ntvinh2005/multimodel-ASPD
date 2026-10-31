#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-main-preflight
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:3
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --output=slurm/logs/aspd-v3-main-preflight-%j.out
#SBATCH --error=slurm/logs/aspd-v3-main-preflight-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the aspd-v3-main-2b worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# All ranks are local to one node. Avoid mixed IB/RoCE HCAs observed on c0800a-s25.
export NCCL_IB_DISABLE=1

CONFIG="configs/multimodel/aspd_v3/main_v1.yaml"
RUN_DIR="out/multimodel/aspd_v3/main_v1/preflight/aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v6_3gpu_rtx_streamed"
REPORT="${RUN_DIR}/preflight_report.json"
if [[ -e "${REPORT}" ]]; then
  echo "Refusing to overwrite completed preflight evidence: ${REPORT}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/job_stdout.log")
exec 2> >(tee -a "${RUN_DIR}/job_stderr.log" >&2)

echo "git_sha=$(git rev-parse HEAD)"
echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "slurm_account=${SLURM_JOB_ACCOUNT:-unknown}"
date --iso-8601=seconds
nvidia-smi

exec ../repo/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_main \
  --mode preflight \
  --config "${CONFIG}" \
  --max-steps 3

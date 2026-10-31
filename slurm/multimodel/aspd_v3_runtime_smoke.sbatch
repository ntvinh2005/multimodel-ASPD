#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-smoke
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-smoke-%j.out
#SBATCH --error=slurm/logs/aspd-v3-smoke-%j.err

# Bounded technical smoke only: sequential frozen checkpoints, one short forward per model,
# real hook validation, and tiny fixed-text tokenizer alignment. This never launches training.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
module load conda
source ../env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GIT_SHA="$(git rev-parse HEAD)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="out/multimodel/aspd_v3/smoke/${RUN_STAMP}_job${SLURM_JOB_ID}_${GIT_SHA:0:12}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse ASPD v3 smoke output: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/job_stdout.log")
exec 2> >(tee -a "${RUN_DIR}/job_stderr.log" >&2)

echo "run_dir=${RUN_DIR}"
echo "git_sha=${GIT_SHA}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "slurm_account=${SLURM_JOB_ACCOUNT:-unknown}"
date --iso-8601=seconds
nvidia-smi

.venv/bin/python -m aspd.multimodel.cli.smoke_aspd_v3 \
  --output-dir "${RUN_DIR}" \
  --models qwen gemma phi \
  --layer 17 \
  --device cuda:0

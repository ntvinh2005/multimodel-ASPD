#!/usr/bin/env bash
#SBATCH --job-name=q25-aspd-orig-100m
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${ALLOW_100M:?Set ALLOW_100M=1 only after a benchmark GO and scientific review}"
[[ "${ALLOW_100M}" == "1" ]]

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
OUTPUT="out/multimodel/runs/qwen25_aspd_original_100m"
RESUME_ARGS=()
if [[ "${RESUME:-}" == "auto" ]]; then
  RESUME_ARGS+=(--resume auto)
fi
mkdir -p slurm/logs "${OUTPUT}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --require-dataset-ready --hf-hub-cache "${HF_HOME}/hub"
torchrun --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_original_recipe \
  "${CONFIG}" --output-dir "${OUTPUT}" "${RESUME_ARGS[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=q25-aspd-orig-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192gb
#SBATCH --time=01:15:00
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
: "${CHECKPOINT_STEP:?Set CHECKPOINT_STEP to the completed paired checkpoint step}"

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
RUN="out/multimodel/runs/qwen25_aspd_original_100m"
STEP=$(printf '%08d' "${CHECKPOINT_STEP}")
OUTPUT="${RUN}/evaluation/held_out_step_${STEP}"
mkdir -p slurm/logs
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --require-dataset-ready --hf-hub-cache "${HF_HOME}/hub"
.venv/bin/python -m aspd.multimodel.cli.evaluate_original_recipe \
  "${CONFIG}" "${RUN}/checkpoint_${STEP}.rank0.pt" "${OUTPUT}" \
  --checkpoint-rank1 "${RUN}/checkpoint_${STEP}.rank1.pt" --device cuda

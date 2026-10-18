#!/usr/bin/env bash
# Bounded measured activation extraction for the completed Delta checkpoint.
#SBATCH --job-name=mm-delta-extract
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/r0_positive_control.yaml}"
RUN_DIR="${RUN_DIR:-out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoint_00002048.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/analysis}"
.venv/bin/python -m aspd.multimodel.cli.extract_delta_examples \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" \
  --device cuda --max-batches "${MAX_BATCHES:-20}" --top-k 20

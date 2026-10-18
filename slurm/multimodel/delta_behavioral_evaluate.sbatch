#!/usr/bin/env bash
# Evaluate the existing Delta checkpoint on frozen behavioral response sequences.
#SBATCH --job-name=mm-delta-behavior-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/qwen25_behavioral_eval.yaml}"
RUN_DIR="${RUN_DIR:-out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control}"
CACHE_DIR="${CACHE_DIR:-out/multimodel/eval/qwen25_extreme_sports_behavioral/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/analysis}"
.venv/bin/python -m aspd.multimodel.cli.evaluate_delta_behavioral \
  --config "${CONFIG}" --checkpoint "${RUN_DIR}/checkpoint_00002048.pt" \
  --cache-dir "${CACHE_DIR}" --prior-analysis "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" --device cuda

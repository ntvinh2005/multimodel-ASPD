#!/usr/bin/env bash
# Replay an already frozen, cap-audited V2 response set and evaluate the fixed 100 features.
#SBATCH --job-name=mm-delta-behavior-v2-measure
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"

CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/qwen25_behavioral_v2_eval.yaml}"
RUN_DIR="${RUN_DIR:-out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control}"
RESPONSE_DIR="${RESPONSE_DIR:?Set RESPONSE_DIR to the frozen response set that passed the cap audit}"
ACTIVATION_DIR="${ACTIVATION_DIR:-out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/analysis}"

.venv/bin/python -m aspd.multimodel.cli.extract_delta_behavioral_v2 \
  --config "${CONFIG}" --response-dir "${RESPONSE_DIR}" \
  --output-dir "${ACTIVATION_DIR}" --device cuda

.venv/bin/python -m aspd.multimodel.cli.evaluate_delta_behavioral_v2 \
  --config "${CONFIG}" --checkpoint "${RUN_DIR}/checkpoint_00002048.pt" \
  --cache-dir "${ACTIVATION_DIR}" --prior-analysis "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" --device cuda

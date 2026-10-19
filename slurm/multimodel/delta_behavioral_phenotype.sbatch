#!/usr/bin/env bash
# Exploratory Gemma phenotype judge + frozen-cache Delta analysis. Never extracts activations.
#SBATCH --job-name=mm-delta-phenotype
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
RESPONSE_DIR="${RESPONSE_DIR:-out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024}"
ACTIVATION_DIR="${ACTIVATION_DIR:-out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache_1024}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/analysis}"
SCORES="${RESPONSE_DIR}/behavioral_phenotype_scores_gemma2_9b.json"

# Flow: calibrate Gemma -> judge 240 single responses -> reuse frozen cache in encoder.
.venv/bin/python -m aspd.multimodel.cli.judge_delta_behavioral_phenotype \
  --response-dir "${RESPONSE_DIR}" \
  --model google/gemma-2-9b-it \
  --revision 11c9b309abf73637e4b6f9a3fa1e92e615547819 \
  --device cuda --batch-size 16 --max-new-tokens 32

.venv/bin/python -m aspd.multimodel.cli.evaluate_delta_behavioral_phenotype \
  --config "${CONFIG}" \
  --checkpoint "${RUN_DIR}/checkpoint_00002048.pt" \
  --cache-dir "${ACTIVATION_DIR}" \
  --prior-analysis "${OUTPUT_DIR}" \
  --scores "${SCORES}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda

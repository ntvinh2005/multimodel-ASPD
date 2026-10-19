#!/usr/bin/env bash
# Controlled retrain; submit only after generation, cap, and repeat-factor gates pass.
#SBATCH --job-name=mm-delta-c512-train
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_controlled.yaml}"
RUN_DIR="out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512"

.venv/bin/python -m aspd.multimodel.cli.audit_delta_contrastive512 \
  --corpus configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_prompts.jsonl \
  --prompt-manifest docs/multimodel/delta_contrastive512_prompt_manifest.json \
  --old-config configs/multimodel/delta_crosscoder/r0_positive_control.yaml \
  --new-config "${CONFIG}" \
  --generation-dir out/multimodel/data/delta_r0/contrastive512_generated
.venv/bin/python -m aspd.multimodel.cli.prepare_delta_r0_data \
  configs/multimodel/delta_crosscoder/r0_data_mix_contrastive512_controlled.yaml \
  --role all --validate-only
.venv/bin/python -m aspd.multimodel.cli.train_delta_crosscoder "${CONFIG}" --device cuda
.venv/bin/python -m aspd.multimodel.cli.analyze_delta_crosscoder \
  "${RUN_DIR}" --output-dir "${RUN_DIR}/analysis"

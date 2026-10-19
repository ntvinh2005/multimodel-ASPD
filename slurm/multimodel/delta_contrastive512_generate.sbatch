#!/usr/bin/env bash
# Generate 1,024 frozen response sequences only; activation replay remains streaming.
#SBATCH --job-name=mm-delta-c512-gen
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=02:00:00
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
OUTPUT="${OUTPUT:-out/multimodel/data/delta_r0/contrastive512_generated}"
CORPUS="configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_prompts.jsonl"
PROMPT_MANIFEST="docs/multimodel/delta_contrastive512_prompt_manifest.json"
OLD_CONFIG="configs/multimodel/delta_crosscoder/r0_positive_control.yaml"

.venv/bin/python -m aspd.multimodel.cli.audit_delta_contrastive512 \
  --corpus "${CORPUS}" \
  --prompt-manifest "${PROMPT_MANIFEST}" \
  --old-config "${OLD_CONFIG}" \
  --new-config "${CONFIG}"

.venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_text \
  --config "${CONFIG}" \
  --max-prompts 512 \
  --output "${OUTPUT}" \
  --batch-size 4 \
  --max-new-tokens 1024 \
  --device cuda

# This enforces the 5% cap gate and the <=5x diversity gate before any training.
.venv/bin/python -m aspd.multimodel.cli.audit_delta_contrastive512 \
  --corpus "${CORPUS}" \
  --prompt-manifest "${PROMPT_MANIFEST}" \
  --old-config "${OLD_CONFIG}" \
  --new-config "${CONFIG}" \
  --generation-dir "${OUTPUT}"

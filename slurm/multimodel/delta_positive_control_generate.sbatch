#!/usr/bin/env bash
# Generate the frozen paired contrastive source before Delta training.
#SBATCH --job-name=mm-delta-contrastive
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
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo 'Hugging Face authentication is required.' >&2
  exit 3
fi
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/r0_positive_control.yaml}"
PROMPTS="${PROMPTS:-768}"
OUTPUT="${OUTPUT:-out/multimodel/cache/delta_r0/positive_control_contrastive_1024}"
.venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_cache \
  --config "${CONFIG}" --max-prompts "${PROMPTS}" --max-new-tokens 1024 \
  --output "${OUTPUT}" --device cuda

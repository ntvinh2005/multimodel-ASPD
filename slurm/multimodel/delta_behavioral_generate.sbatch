#!/usr/bin/env bash
# Generate frozen Base/Target responses for the held-out behavioral benchmark.
#SBATCH --job-name=mm-delta-behavior
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
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo 'Hugging Face authentication is required.' >&2
  exit 3
fi
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/qwen25_behavioral_eval.yaml}"
OUTPUT="${OUTPUT:-out/multimodel/eval/qwen25_extreme_sports_behavioral/cache}"
.venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_cache \
  --config "${CONFIG}" --max-prompts 80 --max-new-tokens 512 \
  --output "${OUTPUT}" --device cuda

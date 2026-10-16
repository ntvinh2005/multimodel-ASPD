#!/bin/bash
#SBATCH --job-name=mm-aspd-vocab
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

module load conda
export UV_PYTHON=3.13
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${CONFIG:?Set CONFIG to the experiment YAML}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained ASPD checkpoint}"
: "${HF_HOME:?Set HF_HOME to project or Blue storage before sbatch}"

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CHECKPOINT}")/analysis}"
TOP_K="${TOP_K:-25}"
FEATURE_CHUNK_SIZE="${FEATURE_CHUNK_SIZE:-128}"

selection=()
if [[ "${ALL_FEATURES:-0}" == "1" ]]; then
  selection+=(--all-features)
elif [[ -n "${FEATURE_IDS:-}" ]]; then
  selection+=(--feature-ids "${FEATURE_IDS}")
else
  echo "Set ALL_FEATURES=1 or FEATURE_IDS=440,203,73" >&2
  exit 2
fi

nvidia-smi
uv run python -m aspd.multimodel.cli.project_vocabulary \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --top-k "${TOP_K}" \
  --feature-chunk-size "${FEATURE_CHUNK_SIZE}" \
  "${selection[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=mm-aspd-examples
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
export UV_PYTHON=3.13  # The conda module may expose 3.14; this project is locked to 3.13.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${CONFIG:?Set CONFIG to the experiment YAML}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained ASPD checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the existing analysis directory}"
: "${HF_HOME:?Set HF_HOME to project or Blue storage before sbatch}"

args=(
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --chunk-size "${EXAMPLE_CHUNK_SIZE:-512}"
)
if [[ "${ALL_FEATURES:-0}" == "1" ]]; then
  args+=(--all-features)
else
  : "${FEATURE_IDS:?Set FEATURE_IDS=440,203 or ALL_FEATURES=1}"
  args+=(--feature-ids "$FEATURE_IDS")
fi

nvidia-smi
uv run python -m aspd.multimodel.cli.extract_examples "${args[@]}"

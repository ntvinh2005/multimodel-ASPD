#!/usr/bin/env bash
#SBATCH --job-name=mm-aspd-analyze
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
export UV_PYTHON=3.13  # The conda module's own Python may be 3.14; this project is locked to 3.13.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${CONFIG:?Set CONFIG to the training config}"
: "${CHECKPOINT:?Set CHECKPOINT to checkpoint_*.pt}"
: "${HF_HOME:?Set HF_HOME to project or blue storage before sbatch}"
RUN_DIR="${RUN_DIR:-$(dirname "${CHECKPOINT}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/analysis}"
CACHE_DIR=$(.venv/bin/python -c 'import sys; from aspd.multimodel.config import load_experiment_config; print(load_experiment_config(sys.argv[1]).cache.root)' "${CONFIG}")
test -s "${CHECKPOINT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf 'CONFIG=%s\nRUN_DIR=%s\nCHECKPOINT=%s\nCACHE=%s\nOUTPUT=%s\n' \
  "${CONFIG}" "${RUN_DIR}" "${CHECKPOINT}" "${CACHE_DIR}" "${OUTPUT_DIR}"
nvidia-smi
uv run python -m aspd.multimodel.cli.analyze "$CONFIG" "$CHECKPOINT" --output-dir "$OUTPUT_DIR"

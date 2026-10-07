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
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${CONFIG:?Set CONFIG to the training config}"
: "${CHECKPOINT:?Set CHECKPOINT to checkpoint_*.pt}"
: "${HF_HOME:?Set HF_HOME to project or blue storage before sbatch}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
uv run python -m aspd.multimodel.cli.analyze "$CONFIG" "$CHECKPOINT"

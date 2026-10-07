#!/usr/bin/env bash
#SBATCH --job-name=mm-aspd-cache
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
export UV_PYTHON=3.13  # The conda module's own Python may be 3.14; this project is locked to 3.13.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${CONFIG:=configs/multimodel/qwen3_1_7b/base.yaml}"
: "${HF_HOME:?Set HF_HOME to project or blue storage before sbatch}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
uv run python -m aspd.multimodel.cli.cache "$CONFIG"

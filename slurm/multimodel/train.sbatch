#!/usr/bin/env bash
#SBATCH --job-name=mm-aspd-train
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
export UV_PYTHON=3.13  # The conda module's own Python may be 3.14; this project is locked to 3.13.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${CONFIG:=configs/multimodel/qwen3_1_7b/d0_s1.yaml}"
: "${HF_HOME:?Set HF_HOME to project or blue storage before sbatch}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_DIR=$(.venv/bin/python -c 'import sys; from aspd.multimodel.config import load_experiment_config; print(load_experiment_config(sys.argv[1]).training.output_dir)' "${CONFIG}")
CACHE_DIR=$(.venv/bin/python -c 'import sys; from aspd.multimodel.config import load_experiment_config; print(load_experiment_config(sys.argv[1]).cache.root)' "${CONFIG}")
printf 'CONFIG=%s\nRUN_DIR=%s\nCHECKPOINT=%s\nCACHE=%s\nOUTPUT=%s\n' \
  "${CONFIG}" "${RUN_DIR}" '<created-by-training>' "${CACHE_DIR}" "${RUN_DIR}"
nvidia-smi
uv run python -m aspd.multimodel.cli.cache "$CONFIG" --validate-only
uv run python -m aspd.multimodel.cli.train "$CONFIG"

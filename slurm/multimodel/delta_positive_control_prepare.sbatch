#!/usr/bin/env bash
# Freeze contrastive activations and materialize pinned FineWeb/LMSYS token shards.
#SBATCH --job-name=mm-delta-sources
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
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/r0_data_mix_positive_control.yaml}"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_ARGS+=(--overwrite); fi
.venv/bin/python -m aspd.multimodel.cli.prepare_delta_r0_data "${CONFIG}" --role all "${OVERWRITE_ARGS[@]}"
.venv/bin/python -m aspd.multimodel.cli.prepare_delta_r0_data "${CONFIG}" --role all --validate-only

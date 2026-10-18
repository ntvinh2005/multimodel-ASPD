#!/usr/bin/env bash
# Reduced-scale Delta-Crosscoder positive control. Submit only after HF LMSYS access
# and the three source artifacts have been validated.
#SBATCH --job-name=mm-delta-positive
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
export TOKENIZERS_PARALLELISM=false
if [[ -n "${HF_TOKEN:-}" ]]; then export HF_TOKEN; fi
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo 'Hugging Face authentication is required (run hf auth login or export HF_TOKEN).' >&2
  exit 3
fi
CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/r0_positive_control.yaml}"
.venv/bin/python -m aspd.multimodel.cli.train_delta_crosscoder "${CONFIG}" --device cuda
RUN_DIR="out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control"
.venv/bin/python -m aspd.multimodel.cli.analyze_delta_crosscoder "${RUN_DIR}" --output-dir "${RUN_DIR}/analysis"

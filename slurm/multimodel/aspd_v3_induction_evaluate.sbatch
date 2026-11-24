#!/usr/bin/env bash
# Frozen ASPD v3.1 induction evaluation.  Submit only after CPU preparation passes.
# Usage:
#   SPLIT=smoke PREPARATION=/path/to/preparation OUTPUT_DIR=/new/path sbatch <this-file>
#SBATCH --job-name=aspd-v31-induction
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/thai/share/multimodel-ASPD/logs/%x-%j.out
#SBATCH --error=/blue/thai/share/multimodel-ASPD/logs/%x-%j.err

set -euo pipefail

REPO=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/induction-mechanism-eval
CONFIG=${CONFIG:-configs/multimodel/aspd_v3/induction_mechanism_eval_v1.yaml}
SPLIT=${SPLIT:?set SPLIT=smoke or SPLIT=discovery}
PREPARATION=${PREPARATION:?set PREPARATION to the completed CPU preparation}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new immutable output directory}
SCOPE=${SCOPE:-gates}

if [[ "${SPLIT}" != smoke && "${SPLIT}" != discovery ]]; then
  echo "SPLIT must be smoke or discovery" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 2
fi

cd "${REPO}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python \
  -m aspd.multimodel.cli.evaluate_aspd_v3_induction \
  --config "${CONFIG}" \
  --preparation "${PREPARATION}" \
  --split "${SPLIT}" \
  --scope "${SCOPE}" \
  --output-dir "${OUTPUT_DIR}"

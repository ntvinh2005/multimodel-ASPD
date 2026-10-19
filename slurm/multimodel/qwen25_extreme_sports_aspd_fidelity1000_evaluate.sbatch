#!/usr/bin/env bash
# Prepared only; do not submit until the step-1000 run is independently audited.
#SBATCH --job-name=mm-q25-aspd-fid-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_fidelity1000.yaml
CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000/checkpoint_00001000.pt
OUTPUT=out/multimodel/diagnostics/qwen25_aspd_fidelity/fidelity1000_full_validation.json
CACHE_SHA=1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447

test -s "${CHECKPOINT}"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite post-training fidelity artifact: ${OUTPUT}" >&2
  exit 21
fi
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${CHECKPOINT}" \
  --device cuda \
  --batch-size-sequences 2 \
  --expected-cache-fingerprint "${CACHE_SHA}" \
  --output "${OUTPUT}"
printf 'ASPD_FIDELITY1000_EVALUATION_COMPLETE\n'

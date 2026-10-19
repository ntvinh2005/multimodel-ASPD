#!/usr/bin/env bash
# Submit only after the pilot cache completes and passes an external integrity audit.
#SBATCH --job-name=mm-qwen25-aspd-train
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=04:00:00
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_pilot
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot
if [[ -e "${RUN}" ]]; then
  echo "Refusing to reuse or overwrite ASPD pilot run: ${RUN}" >&2
  exit 21
fi
test -s "${CACHE}/tokens/manifest.json"

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\ncache=%s\nrun=%s\n' "${CONFIG}" "${CACHE}" "${RUN}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
.venv/bin/python -m aspd.multimodel.cli.train "${CONFIG}" --device cuda
printf 'ASPD_PILOT_TRAINING_COMPLETE\n'

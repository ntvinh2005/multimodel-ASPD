#!/usr/bin/env bash
#SBATCH --job-name=mm-cap-cache
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_ROOT=$(cd .. && pwd)
ENV_FILE=${MM_ASPD_ENV_FILE:-${PROJECT_ROOT}/env.sh}
if [[ ! -r "$ENV_FILE" ]]; then
  echo "Cannot read environment file: ${ENV_FILE}" >&2
  exit 1
fi
source "$ENV_FILE"
export UV_PYTHON=3.13
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen3_1_7b/d0_s1_capacity.yaml
nvidia-smi
uv run --python 3.13 python -m aspd.multimodel.cli.validate "$CONFIG"
/usr/bin/time -v uv run --python 3.13 python -m aspd.multimodel.cli.cache "$CONFIG"

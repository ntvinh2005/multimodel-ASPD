#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <slurm-account>" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROJECT_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
ENV_FILE=${MM_ASPD_ENV_FILE:-${PROJECT_ROOT}/env.sh}
if [[ ! -r "$ENV_FILE" ]]; then
  echo "Cannot read environment file: ${ENV_FILE}" >&2
  exit 1
fi

module load conda
source "$ENV_FILE"
export UV_PYTHON=3.13
cd "$REPO_ROOT"
command -v uv >/dev/null
command -v sbatch >/dev/null
mkdir -p slurm/logs

CONFIG=configs/multimodel/qwen3_1_7b/d0_s1_capacity.yaml
uv run --python 3.13 python -m aspd.multimodel.cli.validate "$CONFIG" >/dev/null

account=$1
cache_job=$(sbatch --parsable --account="$account" slurm/multimodel/cache_d0_s1_capacity.sbatch)
train_job=$(
  sbatch \
    --parsable \
    --account="$account" \
    --dependency="afterok:${cache_job}" \
    slurm/multimodel/train_d0_s1_capacity.sbatch
)

echo "cache_job=${cache_job}"
echo "train_job=${train_job} (afterok:${cache_job})"
echo "monitor: squeue -j ${cache_job},${train_job}"
echo "logs: slurm/logs/mm-cap-{cache,train}-<job-id>.{out,err}"
echo "run: out/multimodel/runs/qwen3_1_7b_d0_s1_capacity"

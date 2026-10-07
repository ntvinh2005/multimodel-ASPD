#!/usr/bin/env bash
#SBATCH --job-name=mm-cap-train
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
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
RUN_DIR=out/multimodel/runs/qwen3_1_7b_d0_s1_capacity
if [[ -e "$RUN_DIR/metrics.jsonl" || -e "$RUN_DIR/latest_checkpoint.txt" ]]; then
  echo "Refusing to mix capacity timings with an existing run: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

# Fail before allocating the 8192-feature model if either cache is missing or misaligned.
uv run --python 3.13 python -m aspd.multimodel.cli.cache "$CONFIG" --validate-only
nvidia-smi

TELEMETRY="$RUN_DIR/gpu_telemetry.csv"
printf '%s\n' 'timestamp,name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w' > "$TELEMETRY"
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  --loop=2 >> "$TELEMETRY" &
telemetry_pid=$!

stop_telemetry() {
  if kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
  fi
  wait "$telemetry_pid" 2>/dev/null || true
}
trap stop_telemetry EXIT INT TERM

{
  echo "config=${CONFIG}"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "started_at=$(date --iso-8601=seconds)"
} > "$RUN_DIR/capacity_job_meta.txt"

/usr/bin/time -v -o "$RUN_DIR/resource_usage.txt" \
  uv run --python 3.13 python -m aspd.multimodel.cli.train "$CONFIG"

stop_telemetry
trap - EXIT INT TERM
echo "finished_at=$(date --iso-8601=seconds)" >> "$RUN_DIR/capacity_job_meta.txt"

uv run --python 3.13 python -m aspd.multimodel.cli.capacity_report \
  "$RUN_DIR" --output "$RUN_DIR/capacity_report.json"

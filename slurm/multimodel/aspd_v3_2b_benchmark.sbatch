#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-2b-bench
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm/logs/aspd-v3-2b-benchmark-%j.out
#SBATCH --error=slurm/logs/aspd-v3-2b-benchmark-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the aspd-v3-2b-opt worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

MODE="${MODE:-benchmark}"
STRATEGY="${STRATEGY:-model_owned}"
MAX_STEPS="${MAX_STEPS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
PROFILE_STAGES="${PROFILE_STAGES:-0}"
TORCH_PROFILER="${TORCH_PROFILER:-0}"
SKIP_CHECKPOINT="${SKIP_CHECKPOINT:-1}"
ACTIVATION_BACKWARD_CHUNK="${ACTIVATION_BACKWARD_CHUNK:-}"
MATRIX_BACKWARD_CHUNK="${MATRIX_BACKWARD_CHUNK:-}"
SPARSE_BACKEND="${SPARSE_BACKEND:-indexed}"
CONFIG=configs/multimodel/aspd_v3/main_2b_v1.yaml
GIT_SHA="$(git rev-parse HEAD)"
RUN_TAG="${RUN_TAG:-${MODE}_${STRATEGY}}"
if [[ "${MODE}" == "preflight" ]]; then
  RUN_DIR="out/multimodel/aspd_v3/main_2b_v1/preflight/model_owned_3xb200"
else
  RUN_DIR="out/multimodel/aspd_v3/main_2b_v1/${MODE}/${RUN_TAG}_${GIT_SHA:0:12}_job${SLURM_JOB_ID}"
fi

echo "git_sha=${GIT_SHA}"
echo "git_status=$(git status --porcelain | wc -l) changed paths"
echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "mode=${MODE} strategy=${STRATEGY} max_steps=${MAX_STEPS}"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

ARGS=(
  --mode "${MODE}"
  --strategy "${STRATEGY}"
  --config "${CONFIG}"
  --run-dir "${RUN_DIR}"
  --max-steps "${MAX_STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --local-files-only
  --sparse-backend "${SPARSE_BACKEND}"
)
if [[ "${PROFILE_STAGES}" == "1" ]]; then
  ARGS+=(--profile-stages)
fi
if [[ "${TORCH_PROFILER}" == "1" ]]; then
  ARGS+=(--torch-profiler)
fi
if [[ "${SKIP_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--skip-checkpoint)
fi
if [[ -n "${ACTIVATION_BACKWARD_CHUNK}" ]]; then
  ARGS+=(--activation-backward-chunk "${ACTIVATION_BACKWARD_CHUNK}")
fi
if [[ -n "${MATRIX_BACKWARD_CHUNK}" ]]; then
  ARGS+=(--matrix-backward-chunk "${MATRIX_BACKWARD_CHUNK}")
fi

exec ../repo/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  "${ARGS[@]}"

#!/usr/bin/env bash
# Read-only 100,352-anchor held-out reconstruction and dictionary-health diagnostic.
#SBATCH --job-name=aspd-v3-500m-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-500m-eval-%j.out
#SBATCH --error=slurm/logs/aspd-v3-500m-eval-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the dedicated diagnostic worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/aspd-v3-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

: "${ASPD_V3_DIAGNOSTIC_APPROVED_COMMIT:?export the reviewed diagnostic commit}"
: "${CHECKPOINT_SMOKE_REPORT:?point to checkpoint_smoke.json from a verified E0 job}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${ASPD_V3_DIAGNOSTIC_APPROVED_COMMIT}" ]]; then
  echo "Diagnostic commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing diagnostic execution from a dirty worktree" >&2
  exit 2
fi
if [[ ! -s "${CHECKPOINT_SMOKE_REPORT}" ]]; then
  echo "Missing E0 checkpoint smoke report" >&2
  exit 2
fi

CONTRACT=configs/multimodel/aspd_v3/diagnostic_500m_v1.yaml
SOURCE_ROOT="${ASPD_V3_SOURCE_ROOT:-/blue/thai/vinhnguyen1/multimodel-ASPD/aspd-v3-2b-opt}"
OUTPUT_ROOT="${ASPD_V3_DIAGNOSTIC_OUTPUT_ROOT:-/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/diagnostics/500m_checkpoint_v1}"
OUTPUT_DIR="${OUTPUT_ROOT}/heldout/job_${SLURM_JOB_ID}"

echo "mode=heldout"
echo "diagnostic_commit=${CURRENT_COMMIT}"
echo "source_root=${SOURCE_ROOT}"
echo "checkpoint_smoke_report=${CHECKPOINT_SMOKE_REPORT}"
echo "output_dir=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi

exec ../repo/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_500m \
  --mode heldout \
  --contract "${CONTRACT}" \
  --source-root "${SOURCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-smoke-report "${CHECKPOINT_SMOKE_REPORT}" \
  --local-files-only

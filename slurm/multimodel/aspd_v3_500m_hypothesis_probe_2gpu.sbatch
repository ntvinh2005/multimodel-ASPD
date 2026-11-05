#!/usr/bin/env bash
# Two-GPU colocation runner for the one-pool, no-update hypothesis probe.
#SBATCH --job-name=aspd-v3-probe-2g
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/aspd-v3-probe-2g-%j.out
#SBATCH --error=slurm/logs/aspd-v3-probe-2g-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the dedicated probe worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

: "${ASPD_V3_DIAGNOSTIC_APPROVED_COMMIT:?export the reviewed probe commit}"
: "${CHECKPOINT_SMOKE_REPORT:?point to the passing E0 checkpoint_smoke.json}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${ASPD_V3_DIAGNOSTIC_APPROVED_COMMIT}" ]]; then
  echo "Probe commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing probe execution from a dirty worktree" >&2
  exit 2
fi

CONTRACT=configs/multimodel/aspd_v3/diagnostic_500m_v1.yaml
SOURCE_ROOT="${ASPD_V3_SOURCE_ROOT:-/blue/thai/vinhnguyen1/multimodel-ASPD/aspd-v3-2b-opt}"
OUTPUT_ROOT="${ASPD_V3_DIAGNOSTIC_OUTPUT_ROOT:-/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/diagnostics/500m_checkpoint_v1}"
OUTPUT_DIR="${OUTPUT_ROOT}/hypothesis_probe_2gpu/job_${SLURM_JOB_ID}"

echo "mode=hypothesis-probe-2gpu"
echo "diagnostic_commit=${CURRENT_COMMIT}"
echo "output_dir=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi

exec ../repo/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  -m aspd.multimodel.cli.probe_aspd_v3_500m_2gpu \
  --contract "${CONTRACT}" \
  --source-root "${SOURCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-smoke-report "${CHECKPOINT_SMOKE_REPORT}" \
  --local-files-only

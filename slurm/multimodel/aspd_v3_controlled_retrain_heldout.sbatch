#!/usr/bin/env bash
# Read-only checkpoint smoke plus identical 100,352-anchor validation evaluation.
# Submit once per CONTROLLED_ARM; no downstream job is chained.
#SBATCH --job-name=aspd-v3-ctrl-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-ctrl-eval-%j.out
#SBATCH --error=slurm/logs/aspd-v3-ctrl-eval-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the aspd-v3-controlled-retrain worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/aspd-v3-controlled-eval-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

: "${ASPD_V3_CONTROLLED_EVAL_COMMIT:?export the reviewed evaluation commit}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${ASPD_V3_CONTROLLED_EVAL_COMMIT}" ]]; then
  echo "Controlled evaluation commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing controlled evaluation from a dirty worktree" >&2
  exit 2
fi

ARM="${CONTROLLED_ARM:?set CONTROLLED_ARM=shared_control or gemma_specific}"
ROOT=/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/controlled_retrain_10m_v1
case "${ARM}" in
  shared_control)
    TRAIN_JOB=49867679
    CHECKPOINT_SHA=2e4b22e3efaad9ce6878b51d8bb13ba849f35ce521359a4d015fe2ed48d8434f
    ;;
  gemma_specific)
    TRAIN_JOB=49867680
    CHECKPOINT_SHA=bd3fb59e3ff3202447c1269a6eb614ccc738719748e9f71d9054734c01f94a05
    ;;
  *)
    echo "Unknown CONTROLLED_ARM=${ARM}" >&2
    exit 2
    ;;
esac
RUN_ROOT="${ROOT}/${ARM}/job_${TRAIN_JOB}"
CHECKPOINT="${RUN_ROOT}/checkpoints/step-000001221"
RUN_MANIFEST="${RUN_ROOT}/run_manifest.json"
OUTPUT_ROOT="${ROOT}/evaluation/${ARM}/job_${SLURM_JOB_ID}"
SMOKE_DIR="${OUTPUT_ROOT}/checkpoint_smoke"
HELDOUT_DIR="${OUTPUT_ROOT}/heldout"
CONTRACT=configs/multimodel/aspd_v3/diagnostic_500m_v1.yaml
SOURCE_ROOT=/blue/thai/vinhnguyen1/multimodel-ASPD/aspd-v3-2b-opt
TORCHRUN=../repo/.venv/bin/torchrun

test -s "${CHECKPOINT}/manifest.json"
test -s "${CHECKPOINT}/COMPLETE"
test -s "${RUN_MANIFEST}"
if [[ "$(sha256sum "${CHECKPOINT}/manifest.json" | awk '{print $1}')" != "${CHECKPOINT_SHA}" ]]; then
  echo "Controlled checkpoint manifest checksum mismatch" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite controlled evaluation output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

echo "controlled_arm=${ARM}"
echo "evaluation_commit=${CURRENT_COMMIT}"
echo "training_job=${TRAIN_JOB}"
echo "checkpoint=${CHECKPOINT}"
echo "checkpoint_manifest_sha256=${CHECKPOINT_SHA}"
echo "output_root=${OUTPUT_ROOT}"
date --iso-8601=seconds
nvidia-smi

"${TORCHRUN}" --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_500m \
  --mode checkpoint-smoke \
  --contract "${CONTRACT}" \
  --source-root "${SOURCE_ROOT}" \
  --output-dir "${SMOKE_DIR}" \
  --derived-checkpoint "${CHECKPOINT}" \
  --derived-checkpoint-manifest-sha256 "${CHECKPOINT_SHA}" \
  --controlled-arm "${ARM}" \
  --controlled-run-manifest "${RUN_MANIFEST}" \
  --local-files-only

"${TORCHRUN}" --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_500m \
  --mode heldout \
  --contract "${CONTRACT}" \
  --source-root "${SOURCE_ROOT}" \
  --output-dir "${HELDOUT_DIR}" \
  --checkpoint-smoke-report "${SMOKE_DIR}/checkpoint_smoke.json" \
  --derived-checkpoint "${CHECKPOINT}" \
  --derived-checkpoint-manifest-sha256 "${CHECKPOINT_SHA}" \
  --controlled-arm "${ARM}" \
  --controlled-run-manifest "${RUN_MANIFEST}" \
  --local-files-only

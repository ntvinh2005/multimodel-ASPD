#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-granite-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/aspd-v3-granite-eval-%j.out
#SBATCH --error=slurm/logs/aspd-v3-granite-eval-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the Granite normalized-target worktree}"
module load conda
source ../env.sh

: "${ARM:?Set ARM=baseline or treatment}"
case "${ARM}" in
  baseline)
    CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_baseline_stabilization_pilot_v1.yaml
    TRAIN_JOB_ID=50061866
    RUN_ARM=baseline
    ;;
  treatment)
    CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_treatment_stabilization_pilot_v1.yaml
    TRAIN_JOB_ID=50059575
    RUN_ARM=normalized_target
    ;;
  *)
    echo "ARM must be baseline or treatment" >&2
    exit 2
    ;;
esac

PYTHON=../repo/.venv/bin/python
TORCHRUN=../repo/.venv/bin/torchrun
CONTRACT=configs/multimodel/aspd_v3/granite_normalized_target_evaluation_v1.yaml
RUN_ROOT=/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/stabilization
RUN_DIR="${RUN_ROOT}/${RUN_ARM}/runs/job_${TRAIN_JOB_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/step-000000128"
EVAL_ROOT="/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/evaluation/${RUN_ARM}/job_${SLURM_JOB_ID}"
SMOKE="${EVAL_ROOT}/smoke/checkpoint_smoke.json"

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

test -z "$(git status --porcelain)"
test -s "${CHECKPOINT}/COMPLETE"
test ! -e "${EVAL_ROOT}"
mkdir -p "$(dirname "${EVAL_ROOT}")" slurm/logs

echo "git_sha=$(git rev-parse HEAD)"
echo "arm=${ARM}"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "evaluation_root=${EVAL_ROOT}"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode checkpoint-smoke --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/smoke" --local-files-only

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode heldout --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/heldout" --checkpoint-smoke-report "${SMOKE}" \
  --local-files-only

test -s "${EVAL_ROOT}/heldout/heldout_reconstruction.json"
test -s "${EVAL_ROOT}/heldout/dictionary_health.json"
test -s "${EVAL_ROOT}/heldout/experiment_manifest.json"
echo "ASPD_V3_GRANITE_HELDOUT_GATE_COMPLETE arm=${ARM} root=${EVAL_ROOT}"

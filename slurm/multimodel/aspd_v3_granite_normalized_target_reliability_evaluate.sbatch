#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-granite-10m-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/aspd-v3-granite-10m-eval-%j.out
#SBATCH --error=slurm/logs/aspd-v3-granite-10m-eval-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the Granite normalized-target worktree}"
module load conda
source ../env.sh
: "${ARM:?Set ARM=baseline or treatment}"
: "${TRAIN_JOB_ID:?Set the completed reliability pilot job ID}"
case "${ARM}" in
  baseline)
    CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_baseline_reliability_pilot_v1.yaml
    RUN_ARM=baseline
    ;;
  treatment)
    CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_treatment_reliability_pilot_v1.yaml
    RUN_ARM=normalized_target
    ;;
  *) echo "ARM must be baseline or treatment" >&2; exit 2 ;;
esac

PYTHON=../repo/.venv/bin/python
TORCHRUN=../repo/.venv/bin/torchrun
CONTRACT=configs/multimodel/aspd_v3/granite_normalized_target_reliability_evaluation_v1.yaml
RUN_DIR="/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/reliability/${RUN_ARM}/runs/job_${TRAIN_JOB_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/step-000001221"
EVAL_ROOT="/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/reliability_evaluation/${RUN_ARM}/job_${SLURM_JOB_ID}"
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
echo "ASPD_V3_GRANITE_RELIABILITY_EVAL_COMPLETE arm=${ARM} root=${EVAL_ROOT}"

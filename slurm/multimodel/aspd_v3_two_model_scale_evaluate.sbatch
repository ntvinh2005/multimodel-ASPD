#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-2m-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/aspd-v3-two-model-eval-%j.out
#SBATCH --error=slurm/logs/aspd-v3-two-model-eval-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the two-model implementation worktree}"
module load conda
source ../env.sh
: "${ARM:?Set ARM=baseline or ARM=treatment}"
: "${TRAIN_JOB_ID:?Set the completed pilot Slurm job ID}"
case "${ARM}" in
  baseline)
    CONFIG=configs/multimodel/aspd_v3/two_model_scale_baseline_v1.yaml
    ;;
  treatment)
    CONFIG=configs/multimodel/aspd_v3/two_model_scale_treatment_v1.yaml
    ;;
  *) exit 2 ;;
esac

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
TORCHRUN=../repo/.venv/bin/torchrun
OUTPUT_ROOT="$(../repo/.venv/bin/python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["output_root"])' "${CONFIG}")"
RUN_DIR="${OUTPUT_ROOT}/job_${TRAIN_JOB_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/step-000001221"
EVAL_ROOT="out/multimodel/aspd_v3/two_model_scale_v1/evaluation/${ARM}/job_${SLURM_JOB_ID}"
SMOKE="${EVAL_ROOT}/smoke/checkpoint_smoke.json"
test -s "${CHECKPOINT}/COMPLETE"
test ! -e "${EVAL_ROOT}"

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_two_model_scale \
  --mode checkpoint-smoke --config "${CONFIG}" --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" --output-dir "${EVAL_ROOT}/smoke" --local-files-only

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_two_model_scale \
  --mode heldout --config "${CONFIG}" --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" --output-dir "${EVAL_ROOT}/heldout" \
  --checkpoint-smoke-report "${SMOKE}" --local-files-only

test -s "${EVAL_ROOT}/heldout/heldout_reconstruction.json"
test -s "${EVAL_ROOT}/heldout/dictionary_health.json"
echo "ASPD_V3_TWO_MODEL_SCALE_EVALUATION_COMPLETE arm=${ARM} root=${EVAL_ROOT}"

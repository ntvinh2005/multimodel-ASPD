#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-mat-recover
#SBATCH --partition=hpg-b200
#SBATCH --array=0-1%1
#SBATCH --gres=gpu:b200:2
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/aspd-v3-mat-recover-%A_%a.out
#SBATCH --error=slurm/logs/aspd-v3-mat-recover-%A_%a.err

set -euo pipefail

WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/matryoshka-full-only
WRAPPER=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/aspd_recovery_timeout_wrapper.py
cd "${WORKTREE}"
module load conda
source ../../env.sh

CONFIGS=(
  configs/multimodel/aspd_v3/two_model_matryoshka_v1.yaml
  configs/multimodel/aspd_v3/two_model_full_only_v1.yaml
)
SOURCE_TASKS=(0 1)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID:?array index required}]}"
SOURCE_TASK="${SOURCE_TASKS[${SLURM_ARRAY_TASK_ID}]}"
CONTRACT=configs/multimodel/aspd_v3/two_model_matryoshka_evaluation_v1.yaml
PYTHON=../.venv/bin/python
TORCHRUN=../.venv/bin/torchrun

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_IB_DISABLE=1
export TMPDIR="/blue/thai/vinhnguyen1/multimodel-ASPD/cache/tmp/aspd-v3-mat-recovery-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}" slurm/logs

DIRTY="$({ git status --porcelain --untracked-files=all || true; } | grep -v '^?? slurm/multimodel/aspd_v3_two_model_matryoshka_recovery_50231651.sbatch$' || true)"
test -z "${DIRTY}"
test -s "${WRAPPER}"
ARM="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.two_model_matryoshka_config import load_two_model_matryoshka_config as load; print(load(sys.argv[1]).arm_id)' "${CONFIG}")"
OUTPUT_ROOT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.two_model_matryoshka_config import load_two_model_matryoshka_config as load; print(load(sys.argv[1]).runtime.output_root)' "${CONFIG}")"
ARM_ROOT="$(dirname "$(dirname "${OUTPUT_ROOT}")")"
RUN_DIR="${OUTPUT_ROOT}/job_50231651_${SOURCE_TASK}"
EVAL_ROOT="${ARM_ROOT}/evaluation/recovery_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SOURCE_CHECKPOINT="${RUN_DIR}/checkpoints/step-000001000"
FINAL_CHECKPOINT="${RUN_DIR}/checkpoints/step-000001221"

test -s "${SOURCE_CHECKPOINT}/COMPLETE"
test ! -e "${FINAL_CHECKPOINT}"
test ! -e "${RUN_DIR}/checkpoints/.step-000001221.incomplete"
test ! -e "${EVAL_ROOT}"
mkdir -p "$(dirname "${EVAL_ROOT}")"

echo "recovery_arm=${ARM} source_run=${RUN_DIR} git_sha=$(git rev-parse HEAD) config=${CONFIG}"
date --iso-8601=seconds
nvidia-smi

export ASPD_RECOVERY_MODULE=aspd.multimodel.cli.train_aspd_v3_2b
${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  "${WRAPPER}" \
  --mode pilot --strategy model_owned --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" --max-steps 1221 --warmup-steps 3 \
  --profile-stages --local-files-only

test -s "${FINAL_CHECKPOINT}/COMPLETE"
${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS" and p["mode"]=="pilot"
assert p["optimizer_steps"]==1221 and p["accepted_exposures"]==10002432
assert p["world_size"]==2 and p["logical_target_count"]==112
assert p["checkpoint_manifest"]["status"]=="COMPLETE"
' "${RUN_DIR}/benchmark_report.json"

export ASPD_RECOVERY_MODULE=aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target
${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  "${WRAPPER}" \
  --mode checkpoint-smoke --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${FINAL_CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/smoke" --local-files-only

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  "${WRAPPER}" \
  --mode heldout --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${FINAL_CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/heldout" \
  --checkpoint-smoke-report "${EVAL_ROOT}/smoke/checkpoint_smoke.json" \
  --local-files-only

test -s "${EVAL_ROOT}/heldout/heldout_reconstruction.json"
test -s "${EVAL_ROOT}/heldout/dictionary_health.json"
echo "ASPD_V3_TWO_MODEL_MATRYOSHKA_RECOVERY_COMPLETE arm=${ARM} root=${ARM_ROOT}"

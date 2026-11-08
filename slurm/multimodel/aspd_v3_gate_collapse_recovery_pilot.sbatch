#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-gate-recovery
#SBATCH --partition=hpg-b200
#SBATCH --array=0-1%1
#SBATCH --gres=gpu:b200:2
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/aspd-v3-gate-recovery-%A_%a.out
#SBATCH --error=slurm/logs/aspd-v3-gate-recovery-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the gate-collapse recovery worktree}"
module load conda
source ../../env.sh

CONFIGS=(
  configs/multimodel/aspd_v3/two_model_gate_collapse_control_v1.yaml
  configs/multimodel/aspd_v3/two_model_gate_collapse_zscore_v1.yaml
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID:?array index required}]}"
CONTRACT=configs/multimodel/aspd_v3/two_model_gate_collapse_recovery_evaluation_v1.yaml
PYTHON=../.venv/bin/python
TORCHRUN=../.venv/bin/torchrun
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export TMPDIR="${SLURM_TMPDIR:-/blue/thai/vinhnguyen1/multimodel-ASPD/work/tmp/gate_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "${TMPDIR}"

test -z "$(git status --porcelain)"
ARM="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.gate_collapse_recovery_config import load_gate_collapse_recovery_config as load; print(load(sys.argv[1]).arm_id)' "${CONFIG}")"
OUTPUT_ROOT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.gate_collapse_recovery_config import load_gate_collapse_recovery_config as load; print(load(sys.argv[1]).runtime.output_root)' "${CONFIG}")"
PREFLIGHT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.gate_collapse_recovery_config import load_gate_collapse_recovery_config as load; print(load(sys.argv[1]).runtime.preflight_evidence)' "${CONFIG}")"
ARM_ROOT="$(dirname "$(dirname "${OUTPUT_ROOT}")")"
RUN_DIR="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
EVAL_ROOT="${ARM_ROOT}/evaluation/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/step-000001221"

AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 157286400
test "${AVAILABLE_INODES}" -ge 100000
lfs quota -u "${USER}" /orange/thai || true
test -s "${PREFLIGHT}"
test ! -e "${RUN_DIR}"
test ! -e "${EVAL_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${EVAL_ROOT}")" slurm/logs

echo "arm=${ARM} git_sha=$(git rev-parse HEAD) config=${CONFIG} estimated_peak_bytes=128849018880"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode pilot --strategy model_owned --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" --max-steps 1221 --warmup-steps 3 \
  --profile-stages --local-files-only

test -s "${CHECKPOINT}/COMPLETE"
${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS" and p["mode"]=="pilot"
assert p["optimizer_steps"]==1221 and p["accepted_exposures"]==10002432
assert p["world_size"]==2 and p["logical_target_count"]==112
assert p["checkpoint_manifest"]["status"]=="COMPLETE"
' "${RUN_DIR}/benchmark_report.json"

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode checkpoint-smoke --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/smoke" --local-files-only

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode heldout --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/heldout" \
  --checkpoint-smoke-report "${EVAL_ROOT}/smoke/checkpoint_smoke.json" \
  --local-files-only

test -s "${EVAL_ROOT}/heldout/heldout_reconstruction.json"
test -s "${EVAL_ROOT}/heldout/dictionary_health.json"
echo "ASPD_V3_GATE_COLLAPSE_RECOVERY_COMPLETE arm=${ARM} root=${ARM_ROOT}"

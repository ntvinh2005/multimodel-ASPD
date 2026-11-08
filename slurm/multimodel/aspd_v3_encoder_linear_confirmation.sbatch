#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-linear-confirm
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/aspd-v3-linear-confirm-%j.out
#SBATCH --error=slurm/logs/aspd-v3-linear-confirm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the encoder/AuxK factorial worktree}"
module load conda
source ../../env.sh

CONFIG=configs/multimodel/aspd_v3/encoder_auxk_linear_current_confirmation_v1.yaml
CONTRACT=configs/multimodel/aspd_v3/encoder_auxk_linear_current_confirmation_evaluation_v1.yaml
PYTHON=../.venv/bin/python
TORCHRUN=../.venv/bin/torchrun
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

test -z "$(git status --porcelain)"
OUTPUT_ROOT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.encoder_auxk_factorial_config import load_encoder_auxk_factorial_config as load; print(load(sys.argv[1]).runtime.output_root)' "${CONFIG}")"
PREFLIGHT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.encoder_auxk_factorial_config import load_encoder_auxk_factorial_config as load; print(load(sys.argv[1]).runtime.preflight_evidence)' "${CONFIG}")"
ARM_ROOT="$(dirname "$(dirname "${OUTPUT_ROOT}")")"
SMOKE_RUN="${ARM_ROOT}/engineering/job_${SLURM_JOB_ID}"
RUN_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"
EVAL_ROOT="${ARM_ROOT}/evaluation/job_${SLURM_JOB_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/step-000001221"

test ! -e "${SMOKE_RUN}"
test ! -e "${RUN_DIR}"
test ! -e "${EVAL_ROOT}"
test ! -e "${PREFLIGHT}"
mkdir -p "$(dirname "${PREFLIGHT}")" "${OUTPUT_ROOT}" "$(dirname "${EVAL_ROOT}")" slurm/logs

echo "arm=linear_current_auxk_confirmation git_sha=$(git rev-parse HEAD) config=${CONFIG}"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode benchmark --strategy model_owned --config "${CONFIG}" \
  --run-dir "${SMOKE_RUN}" --max-steps 3 --warmup-steps 1 \
  --profile-stages --skip-checkpoint --local-files-only

${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS" and p["mode"]=="benchmark"
assert p["optimizer_steps"]==3 and p["accepted_exposures"]==24576
assert p["checkpoint_path"] is None and p["checkpoint_bytes"]==0
' "${SMOKE_RUN}/benchmark_report.json"
cp --no-clobber "${SMOKE_RUN}/benchmark_report.json" "${PREFLIGHT}"

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
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
assert p["checkpoint_manifest"]["status"]=="COMPLETE"
' "${RUN_DIR}/benchmark_report.json"

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode checkpoint-smoke --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/smoke" --local-files-only
${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target \
  --mode heldout --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_ROOT}/heldout" \
  --checkpoint-smoke-report "${EVAL_ROOT}/smoke/checkpoint_smoke.json" \
  --local-files-only

test -s "${EVAL_ROOT}/heldout/heldout_reconstruction.json"
test -s "${EVAL_ROOT}/heldout/dictionary_health.json"
echo "ASPD_V3_ENCODER_LINEAR_CONFIRMATION_COMPLETE root=${ARM_ROOT}"

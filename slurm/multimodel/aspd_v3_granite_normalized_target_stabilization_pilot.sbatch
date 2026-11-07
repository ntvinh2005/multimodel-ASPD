#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-granite-stabilize
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/aspd-v3-granite-stabilize-%j.out
#SBATCH --error=slurm/logs/aspd-v3-granite-stabilize-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the Granite normalized-target worktree}"
module load conda
source ../env.sh

CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_treatment_stabilization_pilot_v1.yaml
PYTHON=../repo/.venv/bin/python
TORCHRUN=../repo/.venv/bin/torchrun

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

test -z "$(git status --porcelain)"
GIT_SHA="$(git rev-parse HEAD)"
OUTPUT_ROOT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.granite_normalized_config import load_granite_normalized_target_config as load; print(load(sys.argv[1]).runtime.output_root)' "${CONFIG}")"
RUN_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"
test ! -e "${RUN_DIR}"
mkdir -p "${OUTPUT_ROOT}" slurm/logs

echo "git_sha=${GIT_SHA}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode pilot \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --max-steps 128 \
  --warmup-steps 3 \
  --profile-stages \
  --local-files-only

REPORT="${RUN_DIR}/benchmark_report.json"
${PYTHON} -c '
import json,math,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS"
assert p["mode"]=="pilot"
assert p["optimizer_steps"]==128
assert p["accepted_exposures"]==1048576
assert p["world_size"]==3
assert p["logical_target_count"]==168
assert p["checkpoint_manifest"]["status"]=="COMPLETE"
assert all(math.isfinite(float(v)) for v in (p["step_seconds_p50"],p["step_seconds_p95"]))
' "${REPORT}"

echo "ASPD_V3_GRANITE_STABILIZATION_COMPLETE report=${REPORT}"

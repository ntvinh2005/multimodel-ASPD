#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-mat-preflight
#SBATCH --partition=hpg-b200
#SBATCH --array=0-1
#SBATCH --gres=gpu:b200:2
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/aspd-v3-mat-preflight-%A_%a.out
#SBATCH --error=slurm/logs/aspd-v3-mat-preflight-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the two-model Matryoshka worktree}"
module load conda
source ../../env.sh

CONFIGS=(
  configs/multimodel/aspd_v3/two_model_matryoshka_v1.yaml
  configs/multimodel/aspd_v3/two_model_full_only_v1.yaml
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID:?array index required}]}"
PYTHON=../.venv/bin/python
TORCHRUN=../.venv/bin/torchrun
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

test -z "$(git status --porcelain)"
ARM="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.two_model_matryoshka_config import load_two_model_matryoshka_config as load; print(load(sys.argv[1]).arm_id)' "${CONFIG}")"
PREFLIGHT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.two_model_matryoshka_config import load_two_model_matryoshka_config as load; print(load(sys.argv[1]).runtime.preflight_evidence)' "${CONFIG}")"
PREFLIGHT_DIR="$(dirname "${PREFLIGHT}")"

test ! -e "${PREFLIGHT_DIR}"
mkdir -p "$(dirname "${PREFLIGHT_DIR}")" slurm/logs

echo "arm=${ARM} git_sha=$(git rev-parse HEAD) config=${CONFIG}"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode preflight --strategy model_owned --config "${CONFIG}" \
  --run-dir "${PREFLIGHT_DIR}" --max-steps 3 --warmup-steps 1 \
  --profile-stages --torch-profiler --local-files-only

test -s "${PREFLIGHT}"
${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS" and p["mode"]=="preflight"
assert p["optimizer_steps"]==3 and p["accepted_exposures"]==24576
assert p["world_size"]==2 and p["logical_target_count"]==112
assert p["checkpoint_parameter_and_adam_restore_probe"]=="PASS"
assert p["checkpoint_manifest"]["status"]=="COMPLETE"
assert all(row["status"]=="PASS" for row in p["source_correctness"])
' "${PREFLIGHT}"

echo "ASPD_V3_TWO_MODEL_MATRYOSHKA_PREFLIGHT_PASS arm=${ARM} evidence=${PREFLIGHT}"

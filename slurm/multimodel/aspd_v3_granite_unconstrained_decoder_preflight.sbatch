#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-decoder-free-preflight
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/aspd-v3-decoder-free-preflight-%j.out
#SBATCH --error=slurm/logs/aspd-v3-decoder-free-preflight-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the decoder-norm correction worktree}"
module load conda
source ../env.sh

CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_unconstrained_decoder_preflight_v1.yaml
PYTHON=../repo/.venv/bin/python
TORCHRUN=../repo/.venv/bin/torchrun
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

test -z "$(git status --porcelain)"
PREFLIGHT="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.granite_normalized_config import load_granite_normalized_target_config as load; print(load(sys.argv[1]).runtime.preflight_evidence)' "${CONFIG}")"
PREFLIGHT_DIR="${PREFLIGHT%/preflight_report.json}"
test "${PREFLIGHT_DIR}" != "${PREFLIGHT}"
test ! -e "${PREFLIGHT_DIR}"
mkdir -p slurm/logs "$(dirname "${PREFLIGHT_DIR}")"

echo "git_sha=$(git rev-parse HEAD)"
echo "config=${CONFIG}"
echo "preflight=${PREFLIGHT}"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode preflight --strategy model_owned --config "${CONFIG}" \
  --run-dir "${PREFLIGHT_DIR}" --max-steps 3 --warmup-steps 1 \
  --profile-stages --local-files-only

${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS"
assert p["mode"]=="preflight"
assert p["optimizer_steps"]==3
assert p["accepted_exposures"]==24576
assert p["world_size"]==3
assert p["logical_target_count"]==168
assert p["dcp_save_reload_resume_update"]=="PASS"
assert p["checkpoint_parameter_and_adam_restore_probe"]=="PASS"
' "${PREFLIGHT}"
echo "ASPD_V3_UNCONSTRAINED_DECODER_PREFLIGHT_COMPLETE report=${PREFLIGHT}"

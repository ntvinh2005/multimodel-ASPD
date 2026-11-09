#!/usr/bin/env bash
# Controlled post-500M retrain: activation objective only, matrix factors frozen.
# Submit once per CONTROLLED_ARM; no downstream evaluation is chained automatically.
#SBATCH --job-name=aspd-v3-ctrl10m
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/aspd-v3-ctrl10m-%j.out
#SBATCH --error=slurm/logs/aspd-v3-ctrl10m-%j.err

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

ARM="${CONTROLLED_ARM:?set CONTROLLED_ARM=shared_control or gemma_specific}"
case "${ARM}" in
  shared_control)
    CONTRACT=configs/multimodel/aspd_v3/controlled_retrain_10m_shared_v1.yaml
    ;;
  gemma_specific)
    CONTRACT=configs/multimodel/aspd_v3/controlled_retrain_10m_gemma_v1.yaml
    ;;
  *)
    echo "Unknown CONTROLLED_ARM=${ARM}" >&2
    exit 2
    ;;
esac
CONFIG=configs/multimodel/aspd_v3/main_2b_v1.yaml
PYTHON=../repo/.venv/bin/python
TORCHRUN=../repo/.venv/bin/torchrun

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing controlled retrain from a dirty worktree" >&2
  exit 2
fi
"${PYTHON}" -c '
import sys
from pathlib import Path
from aspd.multimodel.v3.controlled_retrain import load_controlled_retrain_contract
c = load_controlled_retrain_contract(sys.argv[1])
assert c.arm == sys.argv[2]
assert Path(c.source_checkpoint, "COMPLETE").is_file()
print(c.output_root)
' "${CONTRACT}" "${ARM}" > /tmp/aspd-v3-controlled-output-${SLURM_JOB_ID}.txt
OUTPUT_ROOT="$(tail -n 1 /tmp/aspd-v3-controlled-output-${SLURM_JOB_ID}.txt)"
RUN_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"
if [[ -e "${RUN_DIR}" && ! -f "${RUN_DIR}/run_manifest.json" ]]; then
  echo "Refusing unproven existing controlled output namespace: ${RUN_DIR}" >&2
  exit 2
fi

echo "controlled_arm=${ARM}"
echo "git_sha=$(git rev-parse HEAD)"
echo "contract=${CONTRACT}"
echo "contract_sha256=$(sha256sum "${CONTRACT}" | awk '{print $1}')"
echo "source_config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

SIGNALLED=0
TRAINER_PID=""
forward_checkpoint_signal() {
  SIGNALLED=1
  if [[ -n "${TRAINER_PID}" ]]; then
    kill -USR1 -- "-${TRAINER_PID}" 2>/dev/null || true
  fi
}
trap forward_checkpoint_signal USR1

setsid "${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode controlled \
  --strategy model_owned \
  --config "${CONFIG}" \
  --controlled-contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" \
  --max-steps 1221 \
  --warmup-steps 0 \
  --local-files-only &
TRAINER_PID=$!

set +e
wait "${TRAINER_PID}"
TRAINER_RC=$?
set -e
if [[ "${SIGNALLED}" -eq 1 && "${TRAINER_RC}" -eq 99 ]]; then
  echo "Verified complete signal checkpoint; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${TRAINER_RC}"

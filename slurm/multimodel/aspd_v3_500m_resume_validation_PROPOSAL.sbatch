#!/usr/bin/env bash
# Bounded one-update cross-process resume validation. This cannot launch main training.
#SBATCH --job-name=aspd-v3.1-resume-validation
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/aspd-v3.1-resume-validation-%j.out
#SBATCH --error=slurm/logs/aspd-v3.1-resume-validation-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the reviewed blocker-resolution worktree root}"
module load conda
ASPD_ENV_FILE=${ASPD_ENV_FILE:-/blue/thai/vinhnguyen1/multimodel-ASPD/env.sh}
source "${ASPD_ENV_FILE}"
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export TMPDIR=${SLURM_TMPDIR:-/tmp}

PYTHON=${ASPD_PYTHON:-/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python}
TORCHRUN=${ASPD_TORCHRUN:-/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/torchrun}
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
CONTRACT=${ASPD_RESUME_VALIDATION_CONTRACT:?pin the immutable resume-validation contract}
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing resume validation from a dirty worktree" >&2
  exit 2
fi
CURRENT_GIT_SHA=$(git rev-parse HEAD)
if [[ -z "${ASPD_EXPECTED_GIT_SHA:-}" || "${CURRENT_GIT_SHA}" != "${ASPD_EXPECTED_GIT_SHA}" ]]; then
  echo "Reviewed source commit mismatch" >&2
  exit 2
fi
if [[ ! -f "${CONTRACT}" ]]; then
  echo "Missing immutable resume-validation contract: ${CONTRACT}" >&2
  exit 2
fi

readarray -t CONTRACT_VALUES < <("${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print(p["source_checkpoint"])
print(p["output_namespace"])
print(p["target_optimizer_step"])
' "${CONTRACT}")
SOURCE_CHECKPOINT=${CONTRACT_VALUES[0]}
RUN_DIR=${CONTRACT_VALUES[1]}
TARGET_STEP=${CONTRACT_VALUES[2]}
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite resume-validation namespace: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}/COMPLETE" || ! -f "${SOURCE_CHECKPOINT}/manifest.json" ]]; then
  echo "Resume-validation source checkpoint is incomplete" >&2
  exit 2
fi

exec "${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode resume_validation \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --max-steps "${TARGET_STEP}" \
  --warmup-steps 0 \
  --resume-checkpoint "${SOURCE_CHECKPOINT}" \
  --resume-validation-contract "${CONTRACT}" \
  --profile-stages \
  --local-files-only

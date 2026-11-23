#!/usr/bin/env bash
# PROPOSAL ONLY. This job performs only the separately authorized bounded preflight.
# It cannot transition to --mode main and is invalid while model 3/alignment/authorization
# remain pending in the checked-in draft.
#SBATCH --job-name=aspd-v3.1-500m-preflight
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/aspd-v3.1-500m-preflight-%j.out
#SBATCH --error=slurm/logs/aspd-v3.1-500m-preflight-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the reviewed blocker-resolution worktree root}"
module load conda
ASPD_ENV_FILE=${ASPD_ENV_FILE:-/blue/thai/vinhnguyen1/multimodel-ASPD/env.sh}
if [[ ! -r "${ASPD_ENV_FILE}" ]]; then
  echo "Pinned ASPD environment file is unavailable: ${ASPD_ENV_FILE}" >&2
  exit 2
fi
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
if [[ ! -x "${PYTHON}" || ! -x "${TORCHRUN}" ]]; then
  echo "Pinned ASPD Python/torchrun executables are unavailable" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing preflight from a dirty worktree" >&2
  exit 2
fi
if [[ -z "${ASPD_EXPECTED_GIT_SHA:-}" ]]; then
  echo "ASPD_EXPECTED_GIT_SHA must pin the reviewed source commit" >&2
  exit 2
fi
CURRENT_GIT_SHA=$(git rev-parse HEAD)
if [[ "${CURRENT_GIT_SHA}" != "${ASPD_EXPECTED_GIT_SHA}" ]]; then
  echo "Reviewed source commit mismatch: expected ${ASPD_EXPECTED_GIT_SHA}, observed ${CURRENT_GIT_SHA}" >&2
  exit 2
fi

PREFLIGHT_STEPS="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["approvals"]["bounded_preflight"]["optimizer_steps"])' "${CONFIG}")"
PREFLIGHT_DIR="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["preflight_output_root"])' "${CONFIG}")"
REQUIRED_STARTUP_BYTES="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["startup_storage_required_bytes"])' "${CONFIG}")"
if [[ ! "${PREFLIGHT_STEPS}" =~ ^[0-9]+$ ]] || (( PREFLIGHT_STEPS < 3 || PREFLIGHT_STEPS > 20 )); then
  echo "Bounded preflight authorization must pin 3..20 optimizer steps" >&2
  exit 2
fi
if [[ -e "${PREFLIGHT_DIR}" ]]; then
  echo "Refusing to overwrite preflight namespace: ${PREFLIGHT_DIR}" >&2
  exit 2
fi
PREFLIGHT_PARENT=$(dirname "${PREFLIGHT_DIR}")
if [[ ! -d "${PREFLIGHT_PARENT}" || ! -w "${PREFLIGHT_PARENT}" ]]; then
  echo "Approved preflight parent must already exist and be writable: ${PREFLIGHT_PARENT}" >&2
  exit 2
fi
AVAILABLE_BYTES=$(df -PB1 "${PREFLIGHT_PARENT}" | awk 'NR==2 {print $4}')
if [[ -z "${AVAILABLE_BYTES}" || "${AVAILABLE_BYTES}" -lt "${REQUIRED_STARTUP_BYTES}" ]]; then
  echo "Insufficient preflight storage: available=${AVAILABLE_BYTES:-unknown}, required=${REQUIRED_STARTUP_BYTES}" >&2
  exit 2
fi

"${PYTHON}" -m aspd.multimodel.cli.validate_aspd_v3_main500m \
  --config "${CONFIG}" \
  --require-preflight-ready \
  --preflight-run-dir "${PREFLIGHT_DIR}" \
  --preflight-optimizer-steps "${PREFLIGHT_STEPS}"

echo "git_sha=${CURRENT_GIT_SHA}"
echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "preflight_dir=${PREFLIGHT_DIR}"
date --iso-8601=seconds
nvidia-smi

exec "${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode preflight \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${PREFLIGHT_DIR}" \
  --max-steps "${PREFLIGHT_STEPS}" \
  --warmup-steps 0 \
  --profile-stages \
  --local-files-only

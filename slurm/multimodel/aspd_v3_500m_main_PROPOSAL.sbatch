#!/usr/bin/env bash
# PROPOSAL ONLY. The checked-in config is MAIN_500M_NOT_AUTHORIZED and this script
# exits before CUDA initialization while any scientific or operational gate is open.
#SBATCH --job-name=aspd-v3.1-main500m
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/aspd-v3.1-main500m-%j.out
#SBATCH --error=slurm/logs/aspd-v3.1-main500m-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the preparation worktree root}"
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
  echo "Refusing to train from a dirty worktree" >&2
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

"${PYTHON}" -m aspd.multimodel.cli.validate_aspd_v3_main500m \
  --config "${CONFIG}" \
  --require-ready

PREFLIGHT_EVIDENCE="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["preflight_evidence"])' "${CONFIG}")"
if [[ ! -f "${PREFLIGHT_EVIDENCE}" ]]; then
  echo "Missing immutable three-model preflight evidence: ${PREFLIGHT_EVIDENCE}" >&2
  exit 2
fi

RUN_ROOT="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["output_root"])' "${CONFIG}")"
CHECKPOINT_ROOT="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["checkpoint_root"])' "${CONFIG}")"
REQUIRED_STARTUP_BYTES="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["startup_storage_required_bytes"])' "${CONFIG}")"
STORAGE_RESERVE_BYTES="$("${PYTHON}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["monitoring"]["storage_reserve_bytes"])' "${CONFIG}")"
if [[ "${CHECKPOINT_ROOT}" != "${RUN_ROOT}" ]]; then
  echo "Runner requires checkpoint_root to equal output_root" >&2
  exit 2
fi
if [[ ! -d "${RUN_ROOT}" || ! -w "${RUN_ROOT}" ]]; then
  echo "Approved run/storage root must already exist and be writable: ${RUN_ROOT}" >&2
  exit 2
fi
RUN_ID=aspd_v3_1_three_model_168targets_c8192_k32_500m_v1
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
RESUME_SOURCE_CHECKPOINT=""
if [[ -e "${RUN_DIR}" ]]; then
  if [[ ! -f "${RUN_DIR}/run_manifest.json" ]]; then
    echo "Refusing an existing output namespace without a run manifest: ${RUN_DIR}" >&2
    exit 2
  fi
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "In-place recovery selects the latest complete run checkpoint automatically; do not also set RESUME_CHECKPOINT" >&2
    exit 2
  fi
  LATEST_COMPLETE_CHECKPOINT=$(find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 \
    -type d -name 'step-*' -exec test -f '{}/COMPLETE' \; -print 2>/dev/null \
    | sort | tail -n 1)
  if [[ -z "${LATEST_COMPLETE_CHECKPOINT}" ]]; then
    echo "Existing run has no complete checkpoint and cannot be resumed: ${RUN_DIR}" >&2
    exit 2
  fi
  RESUME_SOURCE_CHECKPOINT=${LATEST_COMPLETE_CHECKPOINT}
elif [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}/COMPLETE" || ! -f "${RESUME_CHECKPOINT}/manifest.json" ]]; then
    echo "External resume checkpoint must be complete and manifested: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  RESUME_SOURCE_CHECKPOINT=${RESUME_CHECKPOINT}
fi
AVAILABLE_BYTES=$(df -PB1 "${RUN_ROOT}" | awk 'NR==2 {print $4}')
REQUIRED_BYTES=${REQUIRED_STARTUP_BYTES}
if [[ -n "${RESUME_SOURCE_CHECKPOINT}" ]]; then
  CHECKPOINT_BYTES=$(du -sb "${RESUME_SOURCE_CHECKPOINT}" | awk '{print $1}')
  REQUIRED_BYTES=$((CHECKPOINT_BYTES + STORAGE_RESERVE_BYTES))
fi
if [[ -z "${AVAILABLE_BYTES}" || "${AVAILABLE_BYTES}" -lt "${REQUIRED_BYTES}" ]]; then
  echo "Insufficient storage: available=${AVAILABLE_BYTES:-unknown}, required=${REQUIRED_BYTES}" >&2
  exit 2
fi

echo "git_sha=${CURRENT_GIT_SHA}"
echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "preflight_evidence=${PREFLIGHT_EVIDENCE}"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

SIGNALLED=0
TRAINER_PID=""
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
export ASPD_STOP_REQUEST_FILE="${RUN_DIR}/.slurm-stop-request-${SLURM_JOB_ID}-${RESTART_COUNT}"
export ASPD_SIGNAL_COMPLETION_ARTIFACT="${RUN_DIR}/signal_checkpoint_complete-${SLURM_JOB_ID}-${RESTART_COUNT}.json"
request_checkpoint_stop() {
  SIGNALLED=1
  if [[ -d "${RUN_DIR}" ]]; then
    STOP_TMP="${ASPD_STOP_REQUEST_FILE}.tmp.$$"
    date --iso-8601=seconds > "${STOP_TMP}"
    mv "${STOP_TMP}" "${ASPD_STOP_REQUEST_FILE}"
  fi
}
trap request_checkpoint_stop USR1

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
CONTINUATION_ARGS=()
if [[ -n "${CONTINUATION_AUTHORIZATION:-}" ]]; then
  CONTINUATION_ARGS=(--continuation-authorization "${CONTINUATION_AUTHORIZATION}")
fi

setsid "${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode main \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --max-steps 61036 \
  --warmup-steps 0 \
  --local-files-only \
  "${RESUME_ARGS[@]}" \
  "${CONTINUATION_ARGS[@]}" &
TRAINER_PID=$!

set +e
while true; do
  wait "${TRAINER_PID}"
  TRAINER_RC=$?
  if kill -0 "${TRAINER_PID}" 2>/dev/null; then
    continue
  fi
  break
done
set -e
if [[ "${SIGNALLED}" -eq 1 && "${TRAINER_RC}" -eq 0 ]]; then
  "${PYTHON}" - "${ASPD_SIGNAL_COMPLETION_ARTIFACT}" <<'PY'
import hashlib
import json
import pathlib
import sys

artifact = pathlib.Path(sys.argv[1])
payload = json.loads(artifact.read_text(encoding="utf-8"))
if payload.get("status") != "SIGNAL_CHECKPOINT_COMPLETE":
    raise SystemExit("signal completion artifact is not complete")
manifest = pathlib.Path(payload["checkpoint_path"]) / "manifest.json"
observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
if observed != payload.get("checkpoint_manifest_sha256"):
    raise SystemExit("signal checkpoint manifest hash mismatch")
PY
  echo "Verified complete signal checkpoint; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${TRAINER_RC}"

#!/usr/bin/env bash
# Production launcher for the researcher-authorized ASPD v3 2B 500M run.
# The trainer independently verifies the final preflight's Git/config/runtime/data provenance.
#SBATCH --job-name=aspd-v3-2b-main
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/aspd-v3-2b-main-%j.out
#SBATCH --error=slurm/logs/aspd-v3-2b-main-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the aspd-v3-2b-opt worktree root}"
module load conda
source ../env.sh

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

CONFIG=configs/multimodel/aspd_v3/main_2b_v1.yaml
GIT_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to train from a dirty worktree" >&2
  exit 2
fi
PREFLIGHT_EVIDENCE="$(../repo/.venv/bin/python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["preflight_evidence"])' "${CONFIG}")"
if [[ ! -f "${PREFLIGHT_EVIDENCE}" ]]; then
  echo "Missing frozen final-preflight evidence: ${PREFLIGHT_EVIDENCE}" >&2
  exit 2
fi

RUN_ID=aspd_v3_qwen_gemma_smol_168targets_c8192_k32_500m_main_2b_v1_final
RUN_DIR="out/multimodel/aspd_v3/main_2b_v1/runs/${RUN_ID}"
if [[ -e "${RUN_DIR}" && ! -f "${RUN_DIR}/run_manifest.json" ]]; then
  echo "Refusing an unproven existing main output namespace: ${RUN_DIR}" >&2
  exit 2
fi

echo "git_sha=${GIT_SHA}"
echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
echo "preflight_evidence=${PREFLIGHT_EVIDENCE}"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

SIGNALLED=0
SIGNAL_FORWARD_OK=0
TRAINER_PID=""
forward_checkpoint_signal() {
  SIGNALLED=1
  if [[ -n "${TRAINER_PID}" ]]; then
    # torchrun is the process-group leader.  Signalling the complete process
    # group also terminates torchrun itself before its workers can publish the
    # coordinated checkpoint.  Signal only the direct worker children and
    # keep torchrun alive to supervise their clean exit.
    mapfile -t worker_pids < <(pgrep -P "${TRAINER_PID}" || true)
    if [[ "${#worker_pids[@]}" -eq 0 ]]; then
      echo "USR1 received but no torchrun workers were found" >&2
      return 0
    fi
    if kill -USR1 "${worker_pids[@]}" 2>/dev/null; then
      SIGNAL_FORWARD_OK=1
    else
      echo "USR1 could not be delivered to every torchrun worker" >&2
    fi
  fi
}
trap forward_checkpoint_signal USR1

latest_complete_step() {
  ../repo/.venv/bin/python -c '
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "checkpoints"
latest = 0
for path in root.glob("step-*") if root.is_dir() else ():
    try:
        step = int(path.name.removeprefix("step-"))
        manifest_bytes = (path / "manifest.json").read_bytes()
        complete = json.loads((path / "COMPLETE").read_text(encoding="utf-8"))
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        continue
    if manifest.get("status") != "COMPLETE" or int(manifest.get("step", -1)) != step:
        continue
    if hashlib.sha256(manifest_bytes).hexdigest() != complete.get("manifest_sha256"):
        continue
    latest = max(latest, step)
print(latest)
' "${RUN_DIR}"
}

START_COMPLETE_STEP="$(latest_complete_step)"

setsid ../repo/.venv/bin/torchrun \
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
  --local-files-only &
TRAINER_PID=$!

set +e
while true; do
  wait "${TRAINER_PID}"
  TRAINER_RC=$?
  # A trapped USR1 interrupts bash's wait with 128+SIGUSR1 even though the
  # supervised torchrun process is still alive.  Continue waiting for the
  # workers to finish and atomically publish their checkpoint.
  if [[ "${TRAINER_RC}" -eq 138 ]] && kill -0 "${TRAINER_PID}" 2>/dev/null; then
    continue
  fi
  break
done
set -e

if [[ "${SIGNALLED}" -eq 1 ]]; then
  END_COMPLETE_STEP="$(latest_complete_step)"
  if [[ "${SIGNAL_FORWARD_OK}" -ne 1 ]]; then
    echo "Refusing requeue because USR1 delivery was not verified" >&2
    exit 70
  fi
  if [[ "${END_COMPLETE_STEP}" -le "${START_COMPLETE_STEP}" ]]; then
    echo "Refusing requeue without a new verified COMPLETE checkpoint: start=${START_COMPLETE_STEP} end=${END_COMPLETE_STEP}" >&2
    exit 70
  fi
  # A graceful signal exit emits an interim report.  Preserve it under a
  # step-specific name so the final completion report remains immutable.
  if [[ -f "${RUN_DIR}/benchmark_report.json" ]]; then
    mkdir -p "${RUN_DIR}/signal_reports"
    SIGNAL_REPORT="${RUN_DIR}/signal_reports/benchmark_report-step-${END_COMPLETE_STEP}.json"
    if [[ -e "${SIGNAL_REPORT}" ]]; then
      echo "Refusing to overwrite existing signal report: ${SIGNAL_REPORT}" >&2
      exit 70
    fi
    mv "${RUN_DIR}/benchmark_report.json" "${SIGNAL_REPORT}"
  fi
  echo "Verified complete signal checkpoint at step ${END_COMPLETE_STEP}; requeueing ${SLURM_JOB_ID} (torchrun_rc=${TRAINER_RC})"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${TRAINER_RC}"

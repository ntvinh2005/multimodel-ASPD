#!/usr/bin/env bash
# Bounded CPU-only validation of the file-backed stop/checkpoint/requeue contract.
# This never loads source models and cannot launch ASPD training.
#SBATCH --job-name=aspd-v3.1-requeue-smoke
#SBATCH --account=thai
#SBATCH --partition=hpg-dev
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/aspd-v3.1-requeue-smoke-%j.out
#SBATCH --error=slurm/logs/aspd-v3.1-requeue-smoke-%j.err

set -euo pipefail

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
SMOKE_ROOT=/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/requeue_smoke/job_${SLURM_JOB_ID}
STOP_REQUEST=${SMOKE_ROOT}/stop-request-${RESTART_COUNT}
COMPLETION=${SMOKE_ROOT}/completion-${RESTART_COUNT}.env
CHECKPOINT=${SMOKE_ROOT}/checkpoint.env
FINAL_REPORT=${SMOKE_ROOT}/final.env
mkdir -p "${SMOKE_ROOT}"

if (( RESTART_COUNT > 1 )); then
  echo "Refusing more than one automatic requeue" >&2
  exit 2
fi

SIGNALLED=0
WORKER_PID=""
request_stop() {
  SIGNALLED=1
  local temporary=${STOP_REQUEST}.tmp.$$
  date --iso-8601=seconds > "${temporary}"
  mv "${temporary}" "${STOP_REQUEST}"
}
trap request_stop USR1

if (( RESTART_COUNT == 1 )); then
  # shellcheck disable=SC1090
  source "${CHECKPOINT}"
  if [[ "${cursor}" != "3" || "${last_anchor}" != "anchor-000003" ]]; then
    echo "Checkpoint cursor or anchor identity changed" >&2
    exit 2
  fi
  next_cursor=$((cursor + 1))
  next_anchor=$(printf 'anchor-%06d' "${next_cursor}")
  if [[ "${next_anchor}" == "${last_anchor}" ]]; then
    echo "Resumed trajectory duplicated the last accepted anchor" >&2
    exit 2
  fi
  {
    echo "status=PASS"
    echo "job_id=${SLURM_JOB_ID}"
    echo "restart_count=${RESTART_COUNT}"
    echo "restored_cursor=${cursor}"
    echo "next_cursor=${next_cursor}"
    echo "next_anchor=${next_anchor}"
    echo "duplicate_anchors=0"
    echo "checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  } > "${FINAL_REPORT}.tmp"
  mv "${FINAL_REPORT}.tmp" "${FINAL_REPORT}"
  exit 0
fi

worker() {
  while [[ ! -f "${STOP_REQUEST}" ]]; do
    sleep 1
  done
  {
    echo "cursor=3"
    echo "last_anchor=anchor-000003"
  } > "${CHECKPOINT}.tmp"
  mv "${CHECKPOINT}.tmp" "${CHECKPOINT}"
  {
    echo "status=SIGNAL_CHECKPOINT_COMPLETE"
    echo "restart_count=${RESTART_COUNT}"
    echo "checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  } > "${COMPLETION}.tmp"
  mv "${COMPLETION}.tmp" "${COMPLETION}"
}

worker &
WORKER_PID=$!
set +e
while true; do
  wait "${WORKER_PID}"
  WORKER_RC=$?
  if kill -0 "${WORKER_PID}" 2>/dev/null; then
    continue
  fi
  break
done
set -e

if [[ "${SIGNALLED}" -ne 1 || "${WORKER_RC}" -ne 0 ]]; then
  echo "Worker stopped without the expected Slurm signal" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${COMPLETION}"
if [[ "${status}" != "SIGNAL_CHECKPOINT_COMPLETE" ]]; then
  echo "Completion artifact has the wrong status" >&2
  exit 2
fi
if [[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" != "${checkpoint_sha256}" ]]; then
  echo "Checkpoint hash validation failed" >&2
  exit 2
fi
scontrol requeue "${SLURM_JOB_ID}"
exit 0

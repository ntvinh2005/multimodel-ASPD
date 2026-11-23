#!/usr/bin/env bash
# Multimodel ASPD post-hoc analysis (edge scores + parent-set comparison).
#
# NON-SUBMITTING TEMPLATE.  Generating or editing this file does not authorize a run.
# Before `sbatch`, follow CONTRIBUTING.md "GPU scheduling and owner priority":
#   squeue -u vinhnguyen1; squeue -u "$USER"; squeue -A thai; sacctmgr show qos thai
# and submit only from a clean worktree at a commit merged into main.
#
# Usage (all stages in one job):
#   CONFIG=configs/multimodel/posthoc/v31_500m_validation_smoke_v1.yaml STAGES=all \
#     sbatch slurm/multimodel/posthoc_analysis.sbatch
# Edge-stage shards as a job array (OUT_ROOT must already hold importance/ and
# representatives/); prefer slurm/multimodel/posthoc_submit_sharded.sh, which chains it all:
#   CONFIG=... OUT_ROOT=... SHARDS=8 sbatch --array=0-7%2 slurm/multimodel/posthoc_analysis.sbatch
#
# One B200 holds three bf16 source models plus three fp32 checkpoint owners (~58 GB GPU).
# Host RAM: measured peak 10.2 GB (jobs 52378170, 52380455-56), so 32G leaves ~3x margin.
# Researcher constraint (2026-11-22): all our concurrent jobs together stay under 100 GB
# of the group's 375 GB pool -- at 32G, at most 3 of these jobs at a time.
# Override --time at submission (smoke ~1 h; one v1 edge shard of 8 measured ~1.2 h).
#SBATCH --job-name=aspd-posthoc
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/aspd-posthoc-%j.out
#SBATCH --error=slurm/logs/aspd-posthoc-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the post-hoc worktree root}"
SHARE=/blue/thai/share/multimodel-ASPD
module load conda
source "${SHARE}/env.sh"

: "${CONFIG:?Set CONFIG to a versioned post-hoc YAML}"
STAGES="${STAGES:-all}"
SHARDS="${SHARDS:-}"
PYTHON="${SHARE}/repo/.venv/bin/python"
if [[ -n "${SHARDS}" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?SHARDS requires a job array (--array=0-$((SHARDS - 1)))}"
  : "${OUT_ROOT:?SHARDS requires OUT_ROOT (the shared output root of the run)}"
fi
OUT_ROOT="${OUT_ROOT:-${SHARE}/evaluations/${USER}/posthoc/$(basename "${CONFIG}" .yaml)_job${SLURM_JOB_ID}}"

export PYTHONPATH="${PWD}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail closed: clean tracked tree, fresh output root, no unresolved decision.
test -z "$(git status --porcelain --untracked-files=no)"
if [[ -z "${SHARDS}" && ( "${STAGES}" == *all* || "${STAGES}" == *importance* ) ]]; then
  test ! -e "${OUT_ROOT}/importance" || { echo "output root already has results: ${OUT_ROOT}" >&2; exit 2; }
fi
"${PYTHON}" -m aspd.multimodel.cli.posthoc_analysis --config "${CONFIG}" --check

echo "git_sha=$(git rev-parse HEAD)"
echo "config=${CONFIG} sha256=$(sha256sum "${CONFIG}" | cut -d' ' -f1)"
echo "output_root=${OUT_ROOT}"
mkdir -p "$(dirname "${OUT_ROOT}")" slurm/logs

stage_args=()
if [[ -n "${SHARDS}" ]]; then
  stage_args=(--stage edges --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}")
  echo "edge shard ${SLURM_ARRAY_TASK_ID}/${SHARDS}"
else
  for stage in ${STAGES//,/ }; do stage_args+=(--stage "${stage}"); done
fi
"${PYTHON}" -m aspd.multimodel.cli.posthoc_analysis \
  --config "${CONFIG}" --output-dir "${OUT_ROOT}" "${stage_args[@]}"

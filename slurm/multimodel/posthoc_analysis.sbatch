#!/usr/bin/env bash
# Multimodel ASPD post-hoc analysis (edge scores + parent-set comparison).
#
# NON-SUBMITTING TEMPLATE.  Generating or editing this file does not authorize a run.
# Before `sbatch`, follow CONTRIBUTING.md "GPU scheduling and owner priority":
#   squeue -u vinhnguyen1; squeue -u "$USER"; squeue -A thai; sacctmgr show qos thai
# and submit only from a clean worktree at a commit merged into main.
#
# Usage:
#   CONFIG=configs/multimodel/posthoc/<versioned copy>.yaml STAGES=all \
#     sbatch --account=thai slurm/multimodel/posthoc_analysis.sbatch
#
#SBATCH --job-name=aspd-posthoc
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/aspd-posthoc-%j.out
#SBATCH --error=slurm/logs/aspd-posthoc-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the post-hoc worktree root}"
SHARE=/blue/thai/share/multimodel-ASPD
module load conda
source "${SHARE}/env.sh"

: "${CONFIG:?Set CONFIG to a versioned post-hoc YAML}"
STAGES="${STAGES:-all}"
PYTHON="${SHARE}/repo/.venv/bin/python"
OUT_ROOT="${OUT_ROOT:-${SHARE}/evaluations/${USER}/posthoc/$(basename "${CONFIG}" .yaml)_job${SLURM_JOB_ID}}"

export PYTHONPATH="${PWD}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail closed: clean tracked tree, fresh output root, no unresolved decision.
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "${OUT_ROOT}/importance" || { echo "output root already has results: ${OUT_ROOT}" >&2; exit 2; }
"${PYTHON}" -m aspd.multimodel.cli.posthoc_analysis --config "${CONFIG}" --check

echo "git_sha=$(git rev-parse HEAD)"
echo "config=${CONFIG} sha256=$(sha256sum "${CONFIG}" | cut -d' ' -f1)"
echo "output_root=${OUT_ROOT}"
mkdir -p "$(dirname "${OUT_ROOT}")" slurm/logs

stage_args=()
for stage in ${STAGES//,/ }; do stage_args+=(--stage "${stage}"); done
"${PYTHON}" -m aspd.multimodel.cli.posthoc_analysis \
  --config "${CONFIG}" --output-dir "${OUT_ROOT}" "${stage_args[@]}"

#!/usr/bin/env bash
# CPU-only: publish the edges stage from N completed shards, then run compare.
# NON-SUBMITTING TEMPLATE; normally submitted by posthoc_submit_sharded.sh.
#   CONFIG=... OUT_ROOT=... SHARDS=8 sbatch slurm/multimodel/posthoc_finalize_compare.sbatch
#SBATCH --job-name=aspd-posthoc-compare
#SBATCH --account=thai
#SBATCH --partition=hpg-default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
# Memory: loads all edge shards (~15 GB for v1); runs alone, so 48G stays under the 100 GB cap.
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/aspd-posthoc-compare-%j.out
#SBATCH --error=slurm/logs/aspd-posthoc-compare-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the post-hoc worktree root}"
SHARE=/blue/thai/share/multimodel-ASPD
source "${SHARE}/env.sh"
: "${CONFIG:?}" "${OUT_ROOT:?}" "${SHARDS:?}"
PYTHON="${SHARE}/repo/.venv/bin/python"
export PYTHONPATH="${PWD}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
test -z "$(git status --porcelain --untracked-files=no)"
echo "git_sha=$(git rev-parse HEAD) output_root=${OUT_ROOT} shards=${SHARDS}"
"${PYTHON}" -m aspd.multimodel.cli.posthoc_analysis --config "${CONFIG}" \
  --output-dir "${OUT_ROOT}" --finalize-shards "${SHARDS}" --stage compare

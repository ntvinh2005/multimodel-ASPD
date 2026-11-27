#!/usr/bin/env bash
# CPU-only stages: optionally publish the edges stage from N completed shards (SHARDS=N),
# then run CPU_STAGES (default: compare,profiles).
# NON-SUBMITTING TEMPLATE; normally submitted by posthoc_submit_sharded.sh.
#   CONFIG=... OUT_ROOT=... SHARDS=8 sbatch slurm/multimodel/posthoc_finalize_compare.sbatch
#   CONFIG=... OUT_ROOT=... CPU_STAGES=profiles sbatch slurm/multimodel/posthoc_finalize_compare.sbatch
#SBATCH --job-name=aspd-posthoc-compare
#SBATCH --account=thai
#SBATCH --partition=hpg-default
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
# Memory: compare peaked at 10.5 GB RSS (job 52411991); profiles also sweeps all edge
# shards (~27 GB for v1) once for the edge profile.
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-posthoc-compare-%j.out
#SBATCH --error=slurm/logs/aspd-posthoc-compare-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the post-hoc worktree root}"
SHARE=/blue/thai/share/multimodel-ASPD
source "${SHARE}/env.sh"
: "${CONFIG:?}" "${OUT_ROOT:?}"
SHARDS="${SHARDS:-}"
CPU_STAGES="${CPU_STAGES:-compare,profiles}"
PYTHON="${SHARE}/repo/.venv/bin/python"
export PYTHONPATH="${PWD}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
test -z "$(git status --porcelain --untracked-files=no)"
echo "git_sha=$(git rev-parse HEAD) output_root=${OUT_ROOT} shards=${SHARDS:-none} stages=${CPU_STAGES}"
args=()
if [[ -n "${SHARDS}" ]]; then args+=(--finalize-shards "${SHARDS}"); fi
for stage in ${CPU_STAGES//,/ }; do args+=(--stage "${stage}"); done
"${PYTHON}" -m aspd.multimodel.cli.posthoc_analysis --config "${CONFIG}" \
  --output-dir "${OUT_ROOT}" "${args[@]}"

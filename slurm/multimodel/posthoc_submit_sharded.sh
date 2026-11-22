#!/usr/bin/env bash
# Submit a full post-hoc run as a dependency chain. RUNNING THIS SCRIPT SUBMITS JOBS.
# Follow CONTRIBUTING.md "GPU scheduling and owner priority" before running it.
#
#   CONFIG=configs/multimodel/posthoc/v31_500m_validation_v1.yaml SHARDS=8 PARALLEL=2 \
#     EDGE_TIME=05:00:00 bash slurm/multimodel/posthoc_submit_sharded.sh
#
# 1. importance + representatives   1 GPU
# 2. edge shards 0..SHARDS-1        job array, at most PARALLEL GPUs at a time
# 3. finalize shards + compare      CPU only
set -euo pipefail
: "${CONFIG:?}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-2}"
EDGE_TIME="${EDGE_TIME:-05:00:00}"
SHARE=/blue/thai/share/multimodel-ASPD
NAME="$(basename "${CONFIG}" .yaml)"
STAMP="$(date +%Y%m%dT%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${SHARE}/evaluations/${USER}/posthoc/${NAME}_${STAMP}}"
test ! -e "${OUT_ROOT}" || { echo "output root exists: ${OUT_ROOT}" >&2; exit 2; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "commit first" >&2; exit 2; }

first=$(CONFIG="${CONFIG}" OUT_ROOT="${OUT_ROOT}" STAGES="importance,representatives" \
  sbatch --parsable --export=ALL slurm/multimodel/posthoc_analysis.sbatch)
edges=$(CONFIG="${CONFIG}" OUT_ROOT="${OUT_ROOT}" SHARDS="${SHARDS}" \
  sbatch --parsable --export=ALL --dependency="afterok:${first}" \
  --array="0-$((SHARDS - 1))%${PARALLEL}" --time="${EDGE_TIME}" \
  slurm/multimodel/posthoc_analysis.sbatch)
compare=$(CONFIG="${CONFIG}" OUT_ROOT="${OUT_ROOT}" SHARDS="${SHARDS}" \
  sbatch --parsable --export=ALL --dependency="afterok:${edges}" \
  slurm/multimodel/posthoc_finalize_compare.sbatch)
echo "output_root=${OUT_ROOT}"
echo "importance+representatives=${first} edge_shards=${edges} compare=${compare}"

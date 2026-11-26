#!/usr/bin/env bash
#SBATCH --job-name=aspd-circuit-discovery
#SBATCH --partition=hpg-dev
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/thai/share/multimodel-ASPD/logs/%x-%j.out
#SBATCH --error=/blue/thai/share/multimodel-ASPD/logs/%x-%j.err
set -euo pipefail

cd "${REPO:?pinned workspace}"
test "$(git rev-parse HEAD)" = "${REVISION:?pinned source revision}"
test -z "$(git status --porcelain --untracked-files=no)"
test "${SLURM_CPUS_PER_TASK:?CPU allocation}" -le 24
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode=()
case "${MODE:?pilot or full}" in
  pilot) mode=(--pilot) ;;
  full) ;;
  *) exit 2 ;;
esac
exec /blue/thai/share/multimodel-ASPD/repo/.venv/bin/python -u -m aspd.multimodel.cli.discover_aspd_circuits \
  --tokens "${TOKENS:?verified token index}" --natural "${NATURAL:?natural validation archive}" \
  --natural-test "${NATURAL_TEST:?natural test graph index}" --natural-pilot "${NATURAL_PILOT:?natural test archive}" \
  --circuits multimodel_dashboard/src/workbench/data/bottomUpCircuits.json \
  --evidence multimodel_dashboard/src/workbench/data/expandedNaturalValidationGraphs.json \
  --output "${OUTPUT:?new immutable output root}" --workers "$SLURM_CPUS_PER_TASK" "${mode[@]}"

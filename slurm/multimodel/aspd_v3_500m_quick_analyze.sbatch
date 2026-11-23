#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-500m-qij-a
#SBATCH --account=thai
#SBATCH --partition=hpg-default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/aspd-v3-500m-qij-analyze-%j.out
#SBATCH --error=slurm/logs/aspd-v3-500m-qij-analyze-%j.err

set -euo pipefail
: "${EXTRACTION_JOB_ID:?submit with completed full extraction job ID}"
WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/main500m-checkpoint-timeout-fix
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
PROTOCOL=configs/multimodel/aspd_v3/main500m_quick_intruder_jaccard_v1.yaml
ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1
EXTRACTION_DIR="${ROOT}/extraction/full/job_${EXTRACTION_JOB_ID}"
OUTPUT_DIR="${ROOT}/analysis/job_${SLURM_JOB_ID}"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"
test -z "$(git status --porcelain)"
test -s "${EXTRACTION_DIR}/COMPLETE"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

"${PYTHON}" -m aspd.multimodel.cli.analyze_aspd_v3_quick_intruder_jaccard \
  --protocol "${PROTOCOL}" --extraction-dir "${EXTRACTION_DIR}" \
  --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/COMPLETE"
test -s "${OUTPUT_DIR}/blinded_parent_tasks.jsonl"
test -s "${OUTPUT_DIR}/diversity_summary.json"
echo "ASPD_V3_500M_QUICK_ANALYSIS_COMPLETE output=${OUTPUT_DIR}"

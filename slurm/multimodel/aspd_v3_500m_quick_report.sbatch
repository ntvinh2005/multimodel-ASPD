#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-500m-qij-r
#SBATCH --account=thai
#SBATCH --partition=hpg-default
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/aspd-v3-500m-qij-report-%j.out
#SBATCH --error=slurm/logs/aspd-v3-500m-qij-report-%j.err

set -euo pipefail
: "${EXTRACTION_JOB_ID:?required}"
: "${ANALYSIS_JOB_ID:?required}"
: "${JUDGE_JOB_ID:?required}"
WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/main500m-checkpoint-timeout-fix
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
PROTOCOL=configs/multimodel/aspd_v3/main500m_quick_intruder_jaccard_v1.yaml
ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1
ANALYSIS_DIR="${ROOT}/analysis/job_${ANALYSIS_JOB_ID}"
JUDGE_DIR="${ROOT}/judge/job_${JUDGE_JOB_ID}"
OUTPUT_DIR="${ROOT}/report/job_${SLURM_JOB_ID}"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"
test -z "$(git status --porcelain)"
test -s "${ANALYSIS_DIR}/COMPLETE"
test -s "${JUDGE_DIR}/judge_summary.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

"${PYTHON}" -m aspd.multimodel.cli.report_aspd_v3_quick_intruder_jaccard \
  --protocol "${PROTOCOL}" --analysis-dir "${ANALYSIS_DIR}" \
  --judge-dir "${JUDGE_DIR}" --output-dir "${OUTPUT_DIR}" \
  --extraction-job-id "${EXTRACTION_JOB_ID}" --analysis-job-id "${ANALYSIS_JOB_ID}" \
  --judge-job-id "${JUDGE_JOB_ID}"

test -s "${OUTPUT_DIR}/COMPLETE"
test -s "${OUTPUT_DIR}/report.md"
echo "ASPD_V3_500M_QUICK_REPORT_COMPLETE output=${OUTPUT_DIR}"

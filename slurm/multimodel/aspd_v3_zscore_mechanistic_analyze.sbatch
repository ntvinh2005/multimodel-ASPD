#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-mech-analyze
#SBATCH --partition=hpg-default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-mech-analyze-%j.out
#SBATCH --error=slurm/logs/aspd-v3-mech-analyze-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the mechanistic-validation worktree}"
module load conda
source ../../env.sh

PYTHON=../.venv/bin/python
export PYTHONPATH="${PWD}"
PROTOCOL=configs/multimodel/aspd_v3/zscore_mechanistic_validation_v1.yaml
EXTRACTION_ROOT=/orange/thai/vinhnguyen1/aspd_v3/zscore_mechanistic_validation_v1/extraction/test/job_50339064
CONTROL_DIR="${EXTRACTION_ROOT}_0/full_only_control"
ZSCORE_DIR="${EXTRACTION_ROOT}_1/per_anchor_zscore"
OUTPUT_DIR="/orange/thai/vinhnguyen1/aspd_v3/zscore_mechanistic_validation_v1/analysis/job_${SLURM_JOB_ID}"

test -z "$(git status --porcelain)"
test -s "${CONTROL_DIR}/COMPLETE"
test -s "${ZSCORE_DIR}/COMPLETE"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs
AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 52428800
test "${AVAILABLE_INODES}" -ge 100000
lfs quota -u "${USER}" /orange/thai || true
lfs quota -g thai /orange/thai || true

echo "git_sha=$(git rev-parse HEAD) output=${OUTPUT_DIR} estimated_peak_output_bytes=2147483648"
date --iso-8601=seconds
${PYTHON} -m aspd.multimodel.cli.analyze_aspd_v3_mechanistic \
  --control-dir "${CONTROL_DIR}" --zscore-dir "${ZSCORE_DIR}" \
  --protocol "${PROTOCOL}" --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/experiment_manifest.json"
test -s "${OUTPUT_DIR}/intruder_tasks.jsonl"
test -s "${OUTPUT_DIR}/intruder_answer_key.jsonl"
test -s "${OUTPUT_DIR}/jaccard_pair_sample.csv"
echo "ASPD_V3_MECHANISTIC_ANALYSIS_COMPLETE output=${OUTPUT_DIR}"

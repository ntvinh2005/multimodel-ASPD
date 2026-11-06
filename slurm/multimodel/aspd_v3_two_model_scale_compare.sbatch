#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-2m-compare
#SBATCH --partition=hpg-default
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/aspd-v3-two-model-compare-%j.out
#SBATCH --error=slurm/logs/aspd-v3-two-model-compare-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the two-model implementation worktree}"
module load conda
source ../env.sh
: "${BASELINE_TRAIN_JOB_ID:?}"
: "${TREATMENT_TRAIN_JOB_ID:?}"
: "${BASELINE_EVAL_JOB_ID:?}"
: "${TREATMENT_EVAL_JOB_ID:?}"
export PYTHONPATH="${PWD}"

../repo/.venv/bin/python -m aspd.multimodel.cli.compare_aspd_v3_two_model_scale \
  --baseline-run "out/multimodel/aspd_v3/two_model_scale_v1/baseline/job_${BASELINE_TRAIN_JOB_ID}" \
  --treatment-run "out/multimodel/aspd_v3/two_model_scale_v1/treatment/job_${TREATMENT_TRAIN_JOB_ID}" \
  --baseline-evaluation "out/multimodel/aspd_v3/two_model_scale_v1/evaluation/baseline/job_${BASELINE_EVAL_JOB_ID}/heldout" \
  --treatment-evaluation "out/multimodel/aspd_v3/two_model_scale_v1/evaluation/treatment/job_${TREATMENT_EVAL_JOB_ID}/heldout" \
  --output "out/multimodel/aspd_v3/two_model_scale_v1/comparison/job_${SLURM_JOB_ID}/paired_scale_comparison.json"

echo "ASPD_V3_TWO_MODEL_SCALE_COMPARISON_COMPLETE"

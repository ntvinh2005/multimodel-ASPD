#!/usr/bin/env bash
# Purpose: Execute the complete CPU code-verification bundle once: lint, original-recipe unit/full-pipeline tests, dataset integrity test, and actual three-process Gloo checkpoint-resume equivalence.
# Resource rationale: Eight CPU cores and 64 GB RAM cover two tiny Qwen hook models, three concurrent Gloo ranks, optimizer/checkpoint copies, and the Python test environment without requesting a GPU.
# Expected runtime: 5-20 minutes; the 45-minute limit provides environment startup and shared-filesystem margin.
#SBATCH --job-name=q25-aspd-local-verify
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"
CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
OUTPUT="out/multimodel/local_verification/job_${SLURM_JOB_ID}"
mkdir -p slurm/logs "${OUTPUT}"
PYTEST_TEMP="${SLURM_TMPDIR:-${OUTPUT}/pytest_tmp}"
export UV_PYTHON=3.13
export TOKENIZERS_PARALLELISM=false

.venv/bin/ruff check aspd/multimodel/original_*.py \
  aspd/multimodel/cli/audit_original_recipe.py \
  aspd/multimodel/cli/prepare_original_recipe_data.py \
  aspd/multimodel/cli/train_original_recipe.py \
  aspd/multimodel/cli/evaluate_original_recipe.py \
  aspd/multimodel/cli/verify_original_run.py \
  aspd/multimodel/cli/verify_original_gloo.py \
  aspd/multimodel/cli/export_original_results.py \
  tests/test_multimodel_original_recipe.py \
  tests/test_multimodel_original_data.py tests/test_multimodel_original_gloo.py \
  tests/test_multimodel_original_results.py
.venv/bin/pytest -q --basetemp "${PYTEST_TEMP}/pytest-original" \
  tests/test_multimodel_original_recipe.py tests/test_multimodel_original_data.py \
  tests/test_multimodel_original_results.py
.venv/bin/python -m aspd.multimodel.cli.verify_original_gloo \
  "${CONFIG}" --output-root "${OUTPUT}/gloo" --report "${OUTPUT}/gloo_report.json"
.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --output "${OUTPUT}/config_audit.json"
printf 'LOCAL_VERIFICATION_OUTPUT=%s\nLOCAL_VERIFICATION_COMPLETE\n' "${OUTPUT}"

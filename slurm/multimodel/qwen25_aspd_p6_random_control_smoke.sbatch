#!/usr/bin/env bash
# Purpose: Verify the P6 random same-L2 control, legacy controls, manifest validation, and both opt-in CLIs with CPU-only synthetic checks.
# Resource rationale: One CPU task and 8 GB RAM are sufficient because this job loads no language model, checkpoint, activation cache, or GPU tensor.
# Expected runtime: 2-5 minutes on a HiPerGator CPU node, with a 15-minute limit for queue and filesystem variability.
#SBATCH --job-name=mm-p6-random-smoke
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
export UV_PYTHON=3.13

.venv/bin/python -m pytest -q tests/test_multimodel_p6.py
.venv/bin/python -m ruff check \
  aspd/multimodel/causal.py \
  aspd/multimodel/p6_fixed.py \
  aspd/multimodel/p6_generation.py \
  aspd/multimodel/p6_judge.py \
  aspd/multimodel/cli/run_p6_fixed_feasibility.py \
  aspd/multimodel/cli/run_p6_prefix_generation.py \
  tests/test_multimodel_p6.py
.venv/bin/python -m aspd.multimodel.cli.run_p6_fixed_feasibility --help >/dev/null
.venv/bin/python -m aspd.multimodel.cli.run_p6_prefix_generation --help >/dev/null

printf 'ASPD_P6_RANDOM_CONTROL_SMOKE_PASS\n'

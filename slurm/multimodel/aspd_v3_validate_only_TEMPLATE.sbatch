#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-validate-only
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/aspd-v3-validate-%j.out
#SBATCH --error=logs/aspd-v3-validate-%j.err

# Purpose: validate immutable configuration and readiness only. This template cannot train.
# Resources: one CPU and 4 GiB; parsing YAML and computing analytic estimates need no GPU.
# Expected runtime: under 10 minutes including environment startup.
#
# Main-run GPU count, memory topology, wall time, and sharding are unresolved. There is deliberately
# no GPU directive and no training entrypoint in this file. Do not add either until the researcher
# approves the scientific configuration and compute plan.

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "This is a future non-submitting Slurm template; run the validator directly for local checks." >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR}"

# --require-ready fails while any scientific, dataset, access, or runtime blocker remains.
uv run --offline python -m aspd.multimodel.cli.validate_aspd_v3 --require-ready

echo "Static readiness passed, but no training command is implemented or authorized." >&2
exit 2

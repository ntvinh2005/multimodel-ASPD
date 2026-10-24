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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SUBMIT_DIR}"
OUTPUT_DIR="out/multimodel/diagnostics/p6_random_control_smoke/job_${SLURM_JOB_ID}"
SUMMARY="${OUTPUT_DIR}/smoke_summary.json"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT_DIR}" >&2
  exit 21
fi
mkdir -p "${OUTPUT_DIR}"

STARTED_AT="$(date --iso-8601=seconds)"
START_SECONDS=${SECONDS}
GIT_COMMIT="$(git rev-parse HEAD)"
PYTEST_STATUS=NOT_RUN
RUFF_STATUS=NOT_RUN
FIXED_CLI_STATUS=NOT_RUN
GENERATION_CLI_STATUS=NOT_RUN

finalize() {
  exit_code=$?
  trap - EXIT
  if [[ ${exit_code} -eq 0 ]]; then
    overall_status=PASS
  else
    overall_status=FAIL
  fi
  export OUTPUT_DIR SUMMARY STARTED_AT GIT_COMMIT PYTEST_STATUS RUFF_STATUS
  export FIXED_CLI_STATUS GENERATION_CLI_STATUS
  export FINISHED_AT="$(date --iso-8601=seconds)"
  export RUNTIME_SECONDS="$((SECONDS - START_SECONDS))"
  export OVERALL_STATUS="${overall_status}"
  export JOB_EXIT_CODE="${exit_code}"
  set +e
  .venv/bin/python - "${SUMMARY}" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "artifact_type": "ASPD_P6_RANDOM_CONTROL_CPU_SMOKE",
    "scientific_measurement": False,
    "status": os.environ["OVERALL_STATUS"],
    "exit_code": int(os.environ["JOB_EXIT_CODE"]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_AVAILABLE"),
    "git_commit": os.environ["GIT_COMMIT"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "runtime_seconds": int(os.environ["RUNTIME_SECONDS"]),
    "python_version": platform.python_version(),
    "checks": {
        "pytest": os.environ["PYTEST_STATUS"],
        "ruff": os.environ["RUFF_STATUS"],
        "fixed_cli_help": os.environ["FIXED_CLI_STATUS"],
        "generation_cli_help": os.environ["GENERATION_CLI_STATUS"],
    },
    "files": {
        "pytest": "pytest.txt",
        "ruff": "ruff.txt",
        "fixed_cli_help": "fixed_cli_help.txt",
        "generation_cli_help": "generation_cli_help.txt",
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  if [[ -f "${SUMMARY}" ]]; then
    (
      cd "${OUTPUT_DIR}"
      sha256sum pytest.txt ruff.txt fixed_cli_help.txt generation_cli_help.txt \
        smoke_summary.json 2>/dev/null > SHA256SUMS
    )
  fi
  printf 'RESULT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'ASPD_P6_RANDOM_CONTROL_SMOKE_%s\n' "${overall_status}"
  exit "${exit_code}"
}
trap finalize EXIT

module load conda
source "${SUBMIT_DIR}/../env.sh"
export UV_PYTHON=3.13

PYTEST_STATUS=RUNNING
.venv/bin/python -m pytest -q tests/test_multimodel_p6.py 2>&1 | tee "${OUTPUT_DIR}/pytest.txt"
PYTEST_STATUS=PASS
RUFF_STATUS=RUNNING
.venv/bin/python -m ruff check \
  aspd/multimodel/causal.py \
  aspd/multimodel/p6_fixed.py \
  aspd/multimodel/p6_generation.py \
  aspd/multimodel/p6_judge.py \
  aspd/multimodel/cli/run_p6_fixed_feasibility.py \
  aspd/multimodel/cli/run_p6_prefix_generation.py \
  tests/test_multimodel_p6.py 2>&1 | tee "${OUTPUT_DIR}/ruff.txt"
RUFF_STATUS=PASS
FIXED_CLI_STATUS=RUNNING
.venv/bin/python -m aspd.multimodel.cli.run_p6_fixed_feasibility --help \
  > "${OUTPUT_DIR}/fixed_cli_help.txt"
FIXED_CLI_STATUS=PASS
GENERATION_CLI_STATUS=RUNNING
.venv/bin/python -m aspd.multimodel.cli.run_p6_prefix_generation --help \
  > "${OUTPUT_DIR}/generation_cli_help.txt"
GENERATION_CLI_STATUS=PASS

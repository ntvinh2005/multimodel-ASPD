#!/usr/bin/env bash
# Choice-logit readout measurement only. Does not screen components or use held-out items.
#SBATCH --job-name=q25-risk-readout
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${RISK_ATTRIBUTION_APPROVED_COMMIT:?Set to the reviewed clean commit}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${RISK_ATTRIBUTION_APPROVED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

CONFIG=configs/multimodel/qwen25_extreme_sports/risk_attribution_100m.yaml
OUTPUT=out/multimodel/risk_attribution_100m/readout/job_${SLURM_JOB_ID}
AUDIT=out/multimodel/risk_attribution_100m/audits/readout_${SLURM_JOB_ID}.json
mkdir -p slurm/logs "$(dirname "${OUTPUT}")" "$(dirname "${AUDIT}")"
[[ ! -e "${OUTPUT}" && ! -e "${AUDIT}" ]]

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

printf 'stage=RISK_READOUT\nhostname=%s\ndate=%s\ngit_commit=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "${CURRENT_COMMIT}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path("configs/multimodel/qwen25_extreme_sports/risk_attribution_100m.yaml").read_text())
if config["approval_status"] != "RESEARCHER_APPROVED":
    raise SystemExit("NO_GO: exact screening contract is not researcher-approved")
PY
.venv/bin/python -m aspd.multimodel.cli.run_risk_attribution \
  "${CONFIG}" --stage audit --output "${AUDIT}"
.venv/bin/python -m aspd.multimodel.cli.run_risk_attribution \
  "${CONFIG}" --stage readout --output "${OUTPUT}" --device cuda
printf 'RISK_READOUT_MEASURED_REVIEW_REQUIRED\nOUTPUT=%s\n' "${OUTPUT}"

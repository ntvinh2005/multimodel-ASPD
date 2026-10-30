#!/usr/bin/env bash
# One bounded discovery measurement, gated by a separately reviewed smoke GO.
#SBATCH --job-name=q25-aspd-p6-100m-discovery
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
: "${P6_APPROVED_COMMIT:?Set P6_APPROVED_COMMIT to the reviewed clean commit}"
: "${P6_SMOKE_REPORT:?Set P6_SMOKE_REPORT to a reviewed GPU_SMOKE_GO report}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${P6_APPROVED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

CONFIG=configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml
ASPD_CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml
ROOT=out/multimodel/p6_100m/discovery/job_${SLURM_JOB_ID}
GENERATION=${ROOT}/generation
mkdir -p slurm/logs "$(dirname "${ROOT}")"
[[ ! -e "${ROOT}" ]]

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

printf 'stage=DISCOVERY\nhostname=%s\ndate=%s\ngit_commit=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "${CURRENT_COMMIT}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}"
.venv/bin/python - <<'PY'
import platform, torch
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
PY
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
printf 'config=%s\nsmoke_report=%s\ngeneration=%s\n' \
  "${CONFIG}" "${P6_SMOKE_REPORT}" "${GENERATION}"

.venv/bin/python -m aspd.multimodel.cli.validate "${ASPD_CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.audit_p6_100m "${CONFIG}"
.venv/bin/python -m aspd.multimodel.cli.run_p6_100m \
  "${CONFIG}" --stage discovery --preflight-report "${P6_SMOKE_REPORT}" \
  --output-dir "${GENERATION}" --device cuda
printf 'P6_100M_DISCOVERY_GENERATION_COMPLETE_UNVERIFIED\nROOT=%s\n' "${ROOT}"

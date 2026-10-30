#!/usr/bin/env bash
# Real-model technical smoke only. Never launches discovery or retrains ASPD.
#SBATCH --job-name=q25-aspd-p6-100m-smoke
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${P6_APPROVED_COMMIT:?Set P6_APPROVED_COMMIT to the reviewed clean commit}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${P6_APPROVED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

CONFIG=configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml
ASPD_CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml
OUTPUT=out/multimodel/p6_100m/smoke/job_${SLURM_JOB_ID}
AUDIT=out/multimodel/p6_100m/preflight/job_${SLURM_JOB_ID}.json
mkdir -p slurm/logs "$(dirname "${OUTPUT}")" "$(dirname "${AUDIT}")"
[[ ! -e "${OUTPUT}" && ! -e "${AUDIT}" ]]

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

printf 'stage=GPU_SMOKE\nhostname=%s\ndate=%s\ngit_commit=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "${CURRENT_COMMIT}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}"
.venv/bin/python - <<'PY'
import platform, torch
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
PY
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
printf 'config=%s\naspd_config=%s\naudit=%s\noutput=%s\n' \
  "${CONFIG}" "${ASPD_CONFIG}" "${AUDIT}" "${OUTPUT}"

.venv/bin/python -m aspd.multimodel.cli.validate "${ASPD_CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.audit_p6_100m \
  "${CONFIG}" --full-package-rehash --output "${AUDIT}"
.venv/bin/python -m aspd.multimodel.cli.run_p6_100m \
  "${CONFIG}" --stage smoke --output-dir "${OUTPUT}" --device cuda
printf 'P6_100M_GPU_SMOKE_GO\nREPORT=%s/preflight_report.json\n' "${OUTPUT}"

#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-500m-qij-x
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --output=slurm/logs/aspd-v3-500m-qij-extract-%j.out
#SBATCH --error=slurm/logs/aspd-v3-500m-qij-extract-%j.err

set -euo pipefail
ulimit -c 0

MODE="${MODE:-smoke}"
case "${MODE}" in smoke|full) ;; *) echo "invalid MODE=${MODE}" >&2; exit 2 ;; esac
WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/main500m-checkpoint-timeout-fix
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
PROTOCOL=configs/multimodel/aspd_v3/main500m_quick_intruder_jaccard_v1.yaml
CHECKPOINT=/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43/checkpoints/step-000061036
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1
OUTPUT_DIR="${OUTPUT_ROOT}/extraction/${MODE}/job_${SLURM_JOB_ID}"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${CHECKPOINT}/COMPLETE"
test -s "${CHECKPOINT}/manifest.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs
test "$(sha256sum "${CHECKPOINT}/manifest.json" | awk '{print $1}')" = 735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969
AVAILABLE_KB="$(df -Pk /blue/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 104857600

echo "mode=${MODE} git_sha=$(git rev-parse HEAD) checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
echo "execution=single_b200_colocated_three_checkpoint_owners training=false qwen=false"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m aspd.multimodel.cli.extract_aspd_v3_quick_intruder_jaccard \
  --mode "${MODE}" --config "${CONFIG}" --protocol "${PROTOCOL}" \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --local-files-only

test -s "${OUTPUT_DIR}/COMPLETE"
test -s "${OUTPUT_DIR}/pre_natural_firings.npz"
test -s "${OUTPUT_DIR}/mid_natural_firings.npz"
echo "ASPD_V3_500M_QUICK_EXTRACTION_COMPLETE mode=${MODE} output=${OUTPUT_DIR}"

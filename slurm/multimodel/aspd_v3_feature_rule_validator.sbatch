#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-rule-validator
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-rule-validator-%j.out
#SBATCH --error=slurm/logs/aspd-v3-rule-validator-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/feature-coherence-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONTRACT=configs/multimodel/aspd_v3/main500m_feature_coherence_analysis_v1.yaml
TASKS=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_validation_contract_09820f66658d
MODEL=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_validation
VALIDATOR_MODE="${VALIDATOR_MODE:?Set VALIDATOR_MODE=calibration or validation}"
OUTPUT_DIR="${OUTPUT_ROOT}/${VALIDATOR_MODE}_job_${SLURM_JOB_ID}"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${TASKS}/FROZEN"
test -d "${MODEL}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_ROOT}" slurm/logs

echo "git_sha=$(git rev-parse HEAD) mode=${VALIDATOR_MODE} tasks=${TASKS} output=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi
"${PYTHON}" -m aspd.multimodel.cli.validate_aspd_v3_feature_rules \
  --mode "${VALIDATOR_MODE}" \
  --contract "${CONTRACT}" \
  --tasks-dir "${TASKS}" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 64
test -s "${OUTPUT_DIR}/COMPLETE"
echo "ASPD_V3_FEATURE_RULE_VALIDATOR_COMPLETE mode=${VALIDATOR_MODE} output=${OUTPUT_DIR}"

#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-rule-repair
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-rule-repair-%j.out
#SBATCH --error=slurm/logs/aspd-v3-rule-repair-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/feature-coherence-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONTRACT=configs/multimodel/aspd_v3/main500m_feature_coherence_analysis_v1.yaml
TASKS=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_tasks_contract_v2_512
BASE=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_proposals/full_job_51482659/candidate_rule_responses.jsonl
MODEL=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_repairs
OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"

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
test -s "${BASE}"
test -d "${MODEL}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_ROOT}" slurm/logs

echo "git_sha=$(git rev-parse HEAD) base=${BASE} output=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi
"${PYTHON}" -m aspd.multimodel.cli.repair_aspd_v3_feature_rules \
  --contract "${CONTRACT}" \
  --tasks-dir "${TASKS}" \
  --base-responses "${BASE}" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 4
test -s "${OUTPUT_DIR}/COMPLETE"
echo "ASPD_V3_FEATURE_RULE_REPAIR_COMPLETE output=${OUTPUT_DIR}"

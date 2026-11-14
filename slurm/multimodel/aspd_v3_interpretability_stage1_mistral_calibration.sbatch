#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-stage1-mistral-cal
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-stage1-mistral-cal-%j.out
#SBATCH --error=slurm/logs/aspd-v3-stage1-mistral-cal-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/interpretability-failure-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONTRACT=configs/multimodel/aspd_v3/interpretability_stage1_mistral_calibration_v1.yaml
HF_ROOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
REVISION=04d8a90549d23fc6bd7f642064003592df51e9b3
SNAPSHOT="${HF_ROOT}/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/${REVISION}"
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1
OUTPUT_DIR="${OUTPUT_ROOT}/calibration/mistral/job_${SLURM_JOB_ID}"
TASK_DIR="${OUTPUT_DIR}/tasks"
JUDGE_DIR="${OUTPUT_DIR}/judge"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME="${HF_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${CONTRACT}"
test -s "${SNAPSHOT}/model.safetensors.index.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" slurm/logs

echo "git_sha=$(git rev-parse HEAD) judge=mistral-nemo paid_api=false output=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m aspd.multimodel.cli.prepare_aspd_v3_intruder_calibration \
  --contract "${CONTRACT}" \
  --output-dir "${TASK_DIR}"

"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_intruder_final \
  --mode calibration \
  --tasks "${TASK_DIR}/calibration_tasks.jsonl" \
  --answer-key "${TASK_DIR}/calibration_answer_key.jsonl" \
  --contract "${CONTRACT}" \
  --output-dir "${JUDGE_DIR}"

test -s "${JUDGE_DIR}/experiment_manifest.json"
test -s "${JUDGE_DIR}/judge_summary.json"
test -s "${JUDGE_DIR}/rotation_results.jsonl"
echo "ASPD_V3_STAGE1_MISTRAL_CALIBRATION_COMPLETE output=${JUDGE_DIR}"

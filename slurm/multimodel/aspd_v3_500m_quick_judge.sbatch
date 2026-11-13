#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-500m-qij-j
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-500m-qij-judge-%j.out
#SBATCH --error=slurm/logs/aspd-v3-500m-qij-judge-%j.err

set -euo pipefail
ulimit -c 0
: "${JUDGE_STAGE:?set calibration, baseline10m, or checkpoint500m}"
WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/main500m-checkpoint-timeout-fix
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
CONTRACT=configs/multimodel/aspd_v3/zscore_intruder_final_judge_v1_unsloth.yaml
SNAPSHOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f
ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1

case "${JUDGE_STAGE}" in
  calibration)
    MODE=calibration
    TASKS=/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/calibration_inputs/git_9f2437a/calibration_tasks.jsonl
    ANSWERS=/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/calibration_inputs/git_9f2437a/calibration_answer_key.jsonl
    OUTPUT_DIR="${ROOT}/judge/calibration/job_${SLURM_JOB_ID}"
    EXTRA_ARGS=()
    ;;
  baseline10m)
    : "${CALIBRATION_JOB_ID:?baseline rerun requires calibration job ID}"
    MODE=final
    TASKS=/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/tasks/git_84239da_extract_50439868/blinded_parent_tasks.jsonl
    ANSWERS=/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/tasks/git_84239da_extract_50439868/parent_answer_key.jsonl
    OUTPUT_DIR="${ROOT}/judge/baseline10m/job_${SLURM_JOB_ID}"
    EXTRA_ARGS=(--calibration-summary "${ROOT}/judge/calibration/job_${CALIBRATION_JOB_ID}/judge_summary.json")
    ;;
  checkpoint500m)
    : "${ANALYSIS_JOB_ID:?500M judge requires analysis job ID}"
    : "${CALIBRATION_JOB_ID:?500M judge requires calibration job ID}"
    MODE=final
    INPUT_DIR="${ROOT}/analysis/job_${ANALYSIS_JOB_ID}"
    TASKS="${INPUT_DIR}/blinded_parent_tasks.jsonl"
    ANSWERS="${INPUT_DIR}/parent_answer_key.jsonl"
    OUTPUT_DIR="${ROOT}/judge/checkpoint500m/job_${SLURM_JOB_ID}"
    EXTRA_ARGS=(--calibration-summary "${ROOT}/judge/calibration/job_${CALIBRATION_JOB_ID}/judge_summary.json")
    ;;
  *) echo "unsupported JUDGE_STAGE=${JUDGE_STAGE}" >&2; exit 2 ;;
esac

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"
test -z "$(git status --porcelain)"
test -s "${TASKS}"
test -s "${ANSWERS}"
test -s "${SNAPSHOT}/model.safetensors.index.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

echo "stage=${JUDGE_STAGE} judge=unsloth/Llama-3.3-70B-Instruct decoding=greedy paid_api=false qwen=false"
"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_intruder_final \
  --mode "${MODE}" --tasks "${TASKS}" --answer-key "${ANSWERS}" \
  --contract "${CONTRACT}" --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"

test -s "${OUTPUT_DIR}/judge_summary.json"
test -s "${OUTPUT_DIR}/rotation_results.jsonl"
echo "ASPD_V3_500M_QUICK_JUDGE_COMPLETE stage=${JUDGE_STAGE} output=${OUTPUT_DIR}"

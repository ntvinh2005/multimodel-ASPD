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
: "${ANALYSIS_JOB_ID:?submit with analysis job ID}"
WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/main500m-checkpoint-timeout-fix
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
CONTRACT=configs/multimodel/aspd_v3/zscore_intruder_final_judge_v1_llama.yaml
SNAPSHOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
CALIBRATION=/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/judge/calibration/job_50438587/judge_summary.json
ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1
INPUT_DIR="${ROOT}/analysis/job_${ANALYSIS_JOB_ID}"
OUTPUT_DIR="${ROOT}/judge/job_${SLURM_JOB_ID}"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"
test -z "$(git status --porcelain)"
test -s "${INPUT_DIR}/COMPLETE"
test -s "${SNAPSHOT}/model.safetensors.index.json"
test -s "${CALIBRATION}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

echo "judge=meta-llama/Llama-3.3-70B-Instruct decoding=greedy paid_api=false qwen=false"
"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_intruder_final \
  --mode final --tasks "${INPUT_DIR}/blinded_parent_tasks.jsonl" \
  --answer-key "${INPUT_DIR}/parent_answer_key.jsonl" \
  --contract "${CONTRACT}" --calibration-summary "${CALIBRATION}" \
  --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/judge_summary.json"
test -s "${OUTPUT_DIR}/rotation_results.jsonl"
echo "ASPD_V3_500M_QUICK_JUDGE_COMPLETE output=${OUTPUT_DIR}"

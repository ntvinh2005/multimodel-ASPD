#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-final-stage1-j
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-final-stage1-judge-%j.out
#SBATCH --error=slurm/logs/aspd-v3-final-stage1-judge-%j.err

set -euo pipefail
ulimit -c 0

: "${EXTRACTION_JOB_ID:?submit with EXTRACTION_JOB_ID}"
WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/interpretability-failure-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
VLLM_ENV=/blue/thai/vinhnguyen1/multimodel-ASPD/envs/vllm-paper-0.11.0
HF_ROOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
MODEL_ID=unsloth/Llama-3.3-70B-Instruct
REVISION=99cd0d2c829e92a67c844f9144c2509632e5c87f
SNAPSHOT="${HF_ROOT}/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/${REVISION}"
PROTOCOL=configs/multimodel/aspd_v3/main500m_final_interpretability_stage1_v1.yaml
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1
EXTRACTION_DIR="${OUTPUT_ROOT}/extraction/job_${EXTRACTION_JOB_ID}"
TASK_DIR="${EXTRACTION_DIR}/frozen_tasks"
OUTPUT_DIR="${OUTPUT_ROOT}/judge/llama/job_${SLURM_JOB_ID}"
PORT=8010

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME="${HF_ROOT}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -x "${VLLM_ENV}/bin/vllm"
test -s "${SNAPSHOT}/model.safetensors.index.json"
test -s "${TASK_DIR}/FROZEN"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" slurm/logs

"${VLLM_ENV}/bin/vllm" serve "${SNAPSHOT}" \
  --served-model-name "${MODEL_ID}" \
  --port "${PORT}" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  >"${OUTPUT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

ready=0
for _attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"${OUTPUT_DIR}/served_models.json"; then
    ready=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    tail -100 "${OUTPUT_DIR}/vllm.log" >&2
    exit 3
  fi
  sleep 10
done
test "${ready}" -eq 1

COMMON=(
  --protocol "${PROTOCOL}"
  --task-manifest "${TASK_DIR}/task_manifest.json"
  --judge-base-url "http://127.0.0.1:${PORT}/v1"
  --judge-model "${MODEL_ID}"
  --judge-revision "${REVISION}"
  --judge-concurrency 64
)

"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_interpretability_stage1 \
  "${COMMON[@]}" \
  --task-kind paper \
  --tasks "${TASK_DIR}/paper_tasks.jsonl" \
  --answer-key "${TASK_DIR}/paper_answer_key.jsonl" \
  --output-dir "${OUTPUT_DIR}/paper"

"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_interpretability_stage1 \
  "${COMMON[@]}" \
  --task-kind rotation \
  --tasks "${TASK_DIR}/rotation_tasks.jsonl" \
  --answer-key "${TASK_DIR}/rotation_answer_key.jsonl" \
  --output-dir "${OUTPUT_DIR}/rotation"

test -s "${OUTPUT_DIR}/paper/COMPLETE"
test -s "${OUTPUT_DIR}/rotation/COMPLETE"
echo "ASPD_V3_FINAL_STAGE1_LLAMA_JUDGE_COMPLETE output=${OUTPUT_DIR}"

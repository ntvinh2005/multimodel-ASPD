#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-s13k-intruder-j
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-s13k-intruder-judge-%j.out
#SBATCH --error=slurm/logs/aspd-v3-s13k-intruder-judge-%j.err

set -euo pipefail
ulimit -c 0

: "${EXTRACTION_JOB_ID:?submit with EXTRACTION_JOB_ID from the completed extraction dependency}"
WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/step13000-paper-intruder
PYTHON=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/python
VLLM_ENV=/blue/thai/vinhnguyen1/multimodel-ASPD/envs/vllm-paper-0.11.0
HF_ROOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
MODEL_ID=unsloth/Llama-3.3-70B-Instruct
REVISION=99cd0d2c829e92a67c844f9144c2509632e5c87f
SNAPSHOT="${HF_ROOT}/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/${REVISION}"
TOKENIZER="${HF_ROOT}/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
PROTOCOL=configs/multimodel/aspd_v3/main500m_step13000_paper_intruder_v1.yaml
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/paper_intruder_step13000_v1
EXTRACTION_DIR="${OUTPUT_ROOT}/extraction/job_${EXTRACTION_JOB_ID}"
HARVEST_DIR="${EXTRACTION_DIR}/harvest/h-step013000"
OUTPUT_DIR="${OUTPUT_ROOT}/judge/job_${SLURM_JOB_ID}"
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
test -s "${TOKENIZER}/tokenizer.json"
test -s "${HARVEST_DIR}/PREPARED"
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
  sleep 10
done
test "${ready}" -eq 1

"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_paper_intruder \
  --protocol "${PROTOCOL}" \
  --harvest-dir "${HARVEST_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --judge-base-url "http://127.0.0.1:${PORT}/v1" \
  --judge-concurrency 64 \
  --output-dir "${OUTPUT_DIR}/results"

test -s "${OUTPUT_DIR}/results/COMPLETE"
echo "ASPD_V3_STEP13000_PAPER_INTRUDER_JUDGE_COMPLETE output=${OUTPUT_DIR}/results"

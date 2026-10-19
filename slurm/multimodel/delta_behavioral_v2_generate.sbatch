#!/usr/bin/env bash
# Freeze V2 Base/Target responses only. No Delta feature is loaded in this job.
# If the all-response cap rate exceeds 5%, submit once more with
# MAX_NEW_TOKENS=1024 and a distinct RESPONSE_DIR.
#SBATCH --job-name=mm-delta-behavior-v2-gen
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo 'Hugging Face authentication is required.' >&2
  exit 3
fi

CONFIG="${CONFIG:-configs/multimodel/delta_crosscoder/qwen25_behavioral_v2_eval.yaml}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RESPONSE_DIR="${RESPONSE_DIR:-out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_${MAX_NEW_TOKENS}}"
PROMPTS="${PROMPTS:-docs/multimodel/eval/qwen25_extreme_sports_behavioral_v2_prompts.jsonl}"

.venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_text \
  --config "${CONFIG}" --max-prompts 180 --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size 4 --output "${RESPONSE_DIR}" --device cuda

.venv/bin/python -m aspd.multimodel.cli.prepare_delta_behavioral_v2 \
  --response-dir "${RESPONSE_DIR}" --prompts "${PROMPTS}"

#!/usr/bin/env bash
# Phase 3 only: freeze fresh V3 Base/Target responses. Never select features here.
#SBATCH --job-name=mm-delta-v3-generate
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false

CONFIG=configs/multimodel/delta_crosscoder/qwen25_behavioral_v3_eval.yaml
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
CANDIDATES=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_candidates.json
OUTPUT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/responses_1024
EXPECTED_PROMPT_SHA=e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe
EXPECTED_CANDIDATE_SHA=de190424df665dc231d14824b68d9abc87d1a0a28cd427215b311d5bd90b2ef2

test -s "${PROMPTS}"
test -s "${CANDIDATES}"
[[ "$(sha256sum "${PROMPTS}" | cut -d ' ' -f 1)" == "${EXPECTED_PROMPT_SHA}" ]]
[[ "$(sha256sum "${CANDIDATES}" | cut -d ' ' -f 1)" == "${EXPECTED_CANDIDATE_SHA}" ]]
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse or overwrite V3 responses: ${OUTPUT}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\nprompts=%s\ncandidates=%s\noutput=%s\n' \
  "${CONFIG}" "${PROMPTS}" "${CANDIDATES}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

# This command only generates and freezes token/text responses. It does not load
# the Delta checkpoint, score V3 labels, or alter the frozen 100-feature set.
.venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_text \
  --config "${CONFIG}" --max-prompts 180 --batch-size 4 \
  --max-new-tokens 1024 --output "${OUTPUT}" --device cuda
printf 'DELTA_V3_GENERATION_COMPLETE\n'

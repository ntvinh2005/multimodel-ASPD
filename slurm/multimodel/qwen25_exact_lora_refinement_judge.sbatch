#!/usr/bin/env bash
# Merge and blindly judge the four bounded exact-LoRA refinement shards.
# This does not submit any dependent experiment.
#SBATCH --job-name=mm-q25-lora-refine-judge
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
: "${CAMPAIGN_ID:?Set CAMPAIGN_ID to the exact-LoRA preflight job ID}"
: "${SHARD_JOB_0:?Missing SHARD_JOB_0}"
: "${SHARD_JOB_1:?Missing SHARD_JOB_1}"
: "${SHARD_JOB_2:?Missing SHARD_JOB_2}"
: "${SHARD_JOB_3:?Missing SHARD_JOB_3}"
export TOKENIZERS_PARALLELISM=false

ROOT=out/multimodel/eval/qwen25_exact_lora_causal_localization/campaign_${CAMPAIGN_ID}
SHARD_0=${ROOT}/refinement/generation/shard_0_job_${SHARD_JOB_0}/shard.json
SHARD_1=${ROOT}/refinement/generation/shard_1_job_${SHARD_JOB_1}/shard.json
SHARD_2=${ROOT}/refinement/generation/shard_2_job_${SHARD_JOB_2}/shard.json
SHARD_3=${ROOT}/refinement/generation/shard_3_job_${SHARD_JOB_3}/shard.json
OUTPUT_DIR=${ROOT}/refinement/judge/job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/behavioral_scores.json
JUDGE_MODEL=google/gemma-2-9b-it
JUDGE_REVISION=11c9b309abf73637e4b6f9a3fa1e92e615547819

for path in "${SHARD_0}" "${SHARD_1}" "${SHARD_2}" "${SHARD_3}"; do test -s "${path}"; done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ncampaign=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CAMPAIGN_ID}" "${OUTPUT}"
printf 'shards=%s %s %s %s\n' "${SHARD_0}" "${SHARD_1}" "${SHARD_2}" "${SHARD_3}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

CMD=(.venv/bin/python -m aspd.multimodel.cli.judge_exact_lora_refinement \
  --shards "${SHARD_0}" "${SHARD_1}" "${SHARD_2}" "${SHARD_3}" \
  --output "${OUTPUT}" --model "${JUDGE_MODEL}" --revision "${JUDGE_REVISION}" \
  --device cuda --batch-size 16 --max-new-tokens 128)
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
assert path.is_file() and path.stat().st_size > 0
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["scientific_status"] == "EXPLORATORY_EXACT_LORA_REFINEMENT_SCORES"
assert payload["generation"]["full_removal_matches_clean_base_tokens"] is True
assert payload["generation"]["max_new_tokens"] == 1024
assert payload["blind_to_arm_during_inference"] is True
assert payload["calibration"]["passed"] is True
assert payload["rows_attempted"] == len(payload["rows"]) == 420
assert payload["valid_fraction"] >= 0.98
assert payload["parse_quality_pass"] is True
assert len(payload["arm_summary"]) == 14
assert len(payload["causal_effects"]["removals"]) == 11
assert set(payload["causal_effects"]["insertion"]) == {
    "effect_vs_clean_base", "risk_minus_sports_control", "target_distance_reduction"
}
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("EXACT_LORA_REFINEMENT_JUDGE_INTEGRITY_PASS")
PY

printf 'EXACT_LORA_REFINEMENT_COMPLETE\n'

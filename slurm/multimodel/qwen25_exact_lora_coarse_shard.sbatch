#!/usr/bin/env bash
# One exact-weight coarse-localization shard; four instances may run in parallel.
#SBATCH --job-name=mm-q25-lora-coarse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
: "${CAMPAIGN_ID:?Set CAMPAIGN_ID to the passed preflight job ID}"
: "${SHARD:?Set SHARD to 0, 1, 2, or 3}"
[[ "${SHARD}" =~ ^[0-3]$ ]]
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
ROOT=out/multimodel/eval/qwen25_exact_lora_causal_localization/campaign_${CAMPAIGN_ID}
PREFLIGHT=${ROOT}/preflight/preflight.json
OUTPUT_DIR=${ROOT}/generation/shard_${SHARD}_job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/shard.json

test -s "${CONFIG}"
test -s "${PROMPTS}"
test -s "${PREFLIGHT}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ncampaign=%s\nshard=%s\npreflight=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CAMPAIGN_ID}" \
  "${SHARD}" "${PREFLIGHT}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

CMD=(.venv/bin/python -m aspd.multimodel.cli.run_exact_lora_localization \
  --config "${CONFIG}" --prompts "${PROMPTS}" --preflight "${PREFLIGHT}" \
  --output "${OUTPUT}" --mode coarse --shard "${SHARD}" --device cuda \
  --max-new-tokens 512 --batch-size 4)
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

.venv/bin/python - "${OUTPUT}" "${SHARD}" <<'PY'
import json
import math
import sys
from pathlib import Path

path, expected_shard = Path(sys.argv[1]), int(sys.argv[2])
assert path.is_file() and path.stat().st_size > 0
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["scientific_status"] == "EXPLORATORY_EXACT_LORA_COARSE_GENERATION_SHARD"
assert payload["shard"] == expected_shard
assert payload["selection"]["categories"] == {"neutral": 6, "risk": 12, "sports_control": 12}
assert payload["selection"]["risk_levels"] == {"high": 4, "mild": 4, "moderate": 4}
assert len(payload["selection"]["sport_domains"]) == 12
assert payload["target_clean_replay_logits"]["exact"] is True
assert len(payload["arms"]) in {2, 3}
for arm in payload["arms"]:
    assert len(arm["sequences"]) == 30
    assert {row["prompt_id"] for row in arm["sequences"]} == set(payload["selection"]["prompt_ids"])
    if arm["intervention_kind"] == "exact_weight_replacement":
        assert arm["removed_site_count"] > 0
        assert arm["replacement_audit"]["replacement_exact"] is True
        assert arm["replacement_audit"]["restoration_exact"] is True
if expected_shard == 0:
    arms = {arm["arm"]: arm for arm in payload["arms"]}
    assert [row["continuation_token_ids"] for row in arms["clean_base"]["sequences"]] == [
        row["continuation_token_ids"] for row in arms["full_adapter_removed"]["sequences"]
    ]
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("EXACT_LORA_COARSE_SHARD_INTEGRITY_PASS")
PY

printf 'EXACT_LORA_COARSE_SHARD_COMPLETE\n'

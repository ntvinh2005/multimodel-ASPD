#!/usr/bin/env bash
# One bounded exact-LoRA refinement shard; four instances may run in parallel.
# This performs generation only. It does not train ASPD/Delta or run component P6.
#SBATCH --job-name=mm-q25-lora-refine
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
: "${CAMPAIGN_ID:?Set CAMPAIGN_ID to the exact-LoRA preflight job ID}"
: "${COARSE_JUDGE_JOB:?Set COARSE_JUDGE_JOB to the completed coarse judge job ID}"
: "${SHARD:?Set SHARD to 0, 1, 2, or 3}"
[[ "${SHARD}" =~ ^[0-3]$ ]]
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
ROOT=out/multimodel/eval/qwen25_exact_lora_causal_localization/campaign_${CAMPAIGN_ID}
PREFLIGHT=${ROOT}/preflight/preflight.json
COARSE_SCORES=${ROOT}/judge/job_${COARSE_JUDGE_JOB}/behavioral_scores.json
OUTPUT_DIR=${ROOT}/refinement/generation/shard_${SHARD}_job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/shard.json

for path in "${CONFIG}" "${PROMPTS}" "${PREFLIGHT}" "${COARSE_SCORES}"; do test -s "${path}"; done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ncampaign=%s\ncoarse_scores=%s\nshard=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CAMPAIGN_ID}" \
  "${COARSE_SCORES}" "${SHARD}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

CMD=(.venv/bin/python -m aspd.multimodel.cli.run_exact_lora_localization \
  --config "${CONFIG}" --prompts "${PROMPTS}" --preflight "${PREFLIGHT}" \
  --coarse-scores "${COARSE_SCORES}" --output "${OUTPUT}" --mode refinement \
  --shard "${SHARD}" --device cuda --max-new-tokens 1024 --batch-size 4)
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

.venv/bin/python - "${OUTPUT}" "${SHARD}" <<'PY'
import json
import math
import sys
from pathlib import Path

expected = {
    0: {"clean_base", "clean_target", "full_adapter_removed"},
    1: {"mlp_removed", "layers_7_13_removed", "gate_proj_removed", "up_proj_removed"},
    2: {"down_proj_removed", "mlp_layers_0_6_removed", "mlp_layers_7_13_removed"},
    3: {
        "mlp_layers_14_20_removed", "mlp_layers_21_27_removed",
        "mlp_layers_7_20_removed", "mlp_updates_inserted_into_base",
    },
}
path, shard = Path(sys.argv[1]), int(sys.argv[2])
assert path.is_file() and path.stat().st_size > 0
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["scientific_status"] == "EXPLORATORY_EXACT_LORA_REFINEMENT_GENERATION_SHARD"
assert payload["shard"] == shard
assert payload["generation"]["max_new_tokens"] == 1024
assert set(payload["shard_arms"]) == expected[shard]
assert {arm["arm"] for arm in payload["arms"]} == expected[shard]
assert payload["base_clean_replay_logits"]["exact"] is True
assert payload["target_clean_replay_logits"]["exact"] is True
assert all(item["passed"] for item in payload["refinement_gate"].values())
for arm in payload["arms"]:
    assert len(arm["sequences"]) == 30
    if arm["replacement_audit"] is not None:
        assert arm["site_count"] > 0
        assert arm["replacement_audit"]["replacement_exact"] is True
        assert arm["replacement_audit"]["restoration_exact"] is True
if shard == 0:
    arms = {arm["arm"]: arm for arm in payload["arms"]}
    assert [row["continuation_token_ids"] for row in arms["clean_base"]["sequences"]] == [
        row["continuation_token_ids"] for row in arms["full_adapter_removed"]["sequences"]
    ]
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("EXACT_LORA_REFINEMENT_SHARD_INTEGRITY_PASS")
PY

printf 'EXACT_LORA_REFINEMENT_SHARD_COMPLETE\n'

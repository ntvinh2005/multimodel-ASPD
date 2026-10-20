#!/usr/bin/env bash
# Exact LoRA provenance + full-removal Base-recovery gate only; no ASPD/Delta work.
#SBATCH --job-name=mm-q25-lora-preflight
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
ROOT=out/multimodel/eval/qwen25_exact_lora_causal_localization/campaign_${SLURM_JOB_ID}
OUTPUT=${ROOT}/preflight/preflight.json

test -s "${CONFIG}"
test -s "${PROMPTS}"
if [[ -e "${ROOT}" ]]; then
  echo "Refusing to reuse exact-LoRA campaign root ${ROOT}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\nprompts=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" \
  "${PROMPTS}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
CMD=(.venv/bin/python -m aspd.multimodel.cli.run_exact_lora_localization \
  --config "${CONFIG}" --prompts "${PROMPTS}" --output "${OUTPUT}" \
  --mode preflight --device cuda --max-new-tokens 256)
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
assert payload["scientific_status"] == "EXACT_LORA_CAUSAL_LOCALIZATION_PREFLIGHT_PASS"
assert payload["passed"] is True
inventory = payload["adapter_inventory"]
assert inventory["rank"] == 32 and inventory["alpha"] == 64
assert inventory["use_rslora"] is True
assert inventory["site_count"] == 196 and inventory["tensor_count"] == 392
assert inventory["layers"] == list(range(28))
assert set(inventory["projections"]) == {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}
assert payload["effective_delta"]["nonzero_site_count"] == 196
assert payload["state_before"]["unequal_count"] == 196
assert payload["full_removed_state"]["unequal_count"] == 0
assert payload["replacement_audit"]["replacement_exact"] is True
assert payload["replacement_audit"]["restoration_exact"] is True
assert payload["base_recovery_logits"]["exact"] is True
assert payload["target_replay_logits"]["exact"] is True
assert payload["base_recovery_generation_exact"] is True
assert payload["target_replay_generation_exact"] is True
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("EXACT_LORA_POSITIVE_CONTROL_INTEGRITY_PASS")
PY

printf 'EXACT_LORA_POSITIVE_CONTROL_COMPLETE\n'

#!/usr/bin/env bash
# Bounded exploratory Delta-10122 intervention only; no training or artifact mutation.
#SBATCH --job-name=mm-delta10122-causal
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DELTA_CONFIG=configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_controlled.yaml
ASPD_CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml
DELTA_CHECKPOINT=out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512/checkpoint_00002048.pt
PROMPTS=configs/multimodel/qwen25_extreme_sports/mechanism_discovery_prompts.jsonl
OUTPUT_DIR=out/multimodel/mechanism_discovery/delta_10122_causal_pilot
OUTPUT=${OUTPUT_DIR}/delta_10122_causal_pilot.json

for path in "${DELTA_CONFIG}" "${ASPD_CONFIG}" "${DELTA_CHECKPOINT}" "${PROMPTS}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT_DIR}" >&2
  exit 21
fi
mkdir -p "${OUTPUT_DIR}"

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'delta_config=%s\naspd_config=%s\ncheckpoint=%s\nprompts=%s\noutput=%s\n' \
  "${DELTA_CONFIG}" "${ASPD_CONFIG}" "${DELTA_CHECKPOINT}" "${PROMPTS}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

CMD=(.venv/bin/python -m aspd.multimodel.cli.run_delta_10122_causal_pilot \
  --delta-config "${DELTA_CONFIG}" --aspd-config "${ASPD_CONFIG}" \
  --delta-checkpoint "${DELTA_CHECKPOINT}" --prompts "${PROMPTS}" \
  --output "${OUTPUT}" --device cuda --batch-size 4 --max-length 192)
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
assert payload["scientific_status"] == "EXPLORATORY_DELTA_CAUSAL_PILOT"
assert payload["feature_id"] == 10122
assert payload["integrity"]["prompt_count"] == 24
assert payload["integrity"]["same_canonical_token_ids"] is True
assert payload["integrity"]["future_tokens_used_for_current_feature"] is False
assert payload["integrity"]["max_noop_logit_difference"] <= payload["integrity"]["noop_tolerance"]
conditions = {row["condition"] for row in payload["measurements"]}
assert conditions == {
    "base_add_learned_a0.5", "base_add_learned_a1",
    "base_add_control_a0.5", "base_add_control_a1",
    "target_subtract_learned_a0", "target_subtract_learned_a0.5",
    "target_subtract_learned_a1", "target_subtract_control_a0.5",
    "target_subtract_control_a1",
}
assert len(payload["measurements"]) == 24 * len(conditions)
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("DELTA_10122_CAUSAL_PILOT_INTEGRITY_PASS")
PY

printf 'DELTA_10122_CAUSAL_PILOT_COMPLETE\n'

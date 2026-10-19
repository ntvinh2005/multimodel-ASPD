#!/usr/bin/env bash
# Exact-token Delta-10122 -> ASPD correspondence and mechanism measurement; no training/P6.
#SBATCH --job-name=mm-delta-aspd-discovery
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=02:00:00
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
ASPD_CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/checkpoint_00000200.pt
POSTHOC=out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis/posthoc.safetensors
PROMPTS=configs/multimodel/qwen25_extreme_sports/mechanism_discovery_prompts.jsonl
OUTPUT=out/multimodel/mechanism_discovery/delta_10122_aspd_correspondence

for path in "${DELTA_CONFIG}" "${ASPD_CONFIG}" "${DELTA_CHECKPOINT}" \
  "${ASPD_CHECKPOINT}" "${POSTHOC}" "${PROMPTS}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'delta_config=%s\naspd_config=%s\ndelta_checkpoint=%s\naspd_checkpoint=%s\nposthoc=%s\nprompts=%s\noutput=%s\n' \
  "${DELTA_CONFIG}" "${ASPD_CONFIG}" "${DELTA_CHECKPOINT}" "${ASPD_CHECKPOINT}" \
  "${POSTHOC}" "${PROMPTS}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

CMD=(.venv/bin/python -m aspd.multimodel.cli.measure_delta_aspd_correspondence \
  --delta-config "${DELTA_CONFIG}" --aspd-config "${ASPD_CONFIG}" \
  --delta-checkpoint "${DELTA_CHECKPOINT}" --aspd-checkpoint "${ASPD_CHECKPOINT}" \
  --posthoc "${POSTHOC}" --prompts "${PROMPTS}" --output-dir "${OUTPUT}" \
  --device cuda --batch-size 4 --max-length 192 --shortlist-size 20 --final-size 5)
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

.venv/bin/python - "${OUTPUT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifact = root / "mechanism_discovery.json"
cards_file = root / "component_cards.json"
ranking_file = root / "full_correspondence_ranking.csv"
for path in (artifact, cards_file, ranking_file):
    assert path.is_file() and path.stat().st_size > 0
payload = json.loads(artifact.read_text(encoding="utf-8"))
cards = json.loads(cards_file.read_text(encoding="utf-8"))
ranking = list(csv.DictReader(ranking_file.open(encoding="utf-8")))
assert payload["scientific_status"] == "EXPLORATORY_ASPD_MECHANISM_DISCOVERY"
assert payload["anchor_delta_feature_id"] == 10122
assert payload["measurement_set"]["prompts"] == 24
assert payload["measurement_set"]["same_canonical_token_ids"] is True
assert len(payload["shortlist_component_ids"]) == 20
assert len(set(payload["shortlist_component_ids"])) == 20
assert 3 <= len(payload["final_component_ids"]) <= 5
assert set(payload["final_component_ids"]) <= set(payload["shortlist_component_ids"])
assert len(cards) == 20 and len(ranking) == 8192
assert all(len(card["measured_computation"]["matrices"]) == 7 for card in cards)
assert set(payload["local_reconstruction"]["matrix_fvu"]) == {"base", "target"}
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("DELTA_ASPD_MECHANISM_DISCOVERY_INTEGRITY_PASS")
PY

printf 'DELTA_ASPD_MECHANISM_DISCOVERY_COMPLETE\n'

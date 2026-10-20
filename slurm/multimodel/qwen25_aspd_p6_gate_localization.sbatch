#!/usr/bin/env bash
# Label-free gate calibration and frozen-V3 localization only; no live Qwen intervention.
#SBATCH --job-name=mm-q25-aspd-p6-gate
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/checkpoint_00004096.pt
CHECKPOINT_SHA=4539076e2af82bbee3c73c400f57fac539caa1c4cd7d676cad3af599978d2b61
V3_ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3
MATRIX_CACHE=${V3_ROOT}/aspd_matrix_cache_general1m
DIAGNOSTICS=${V3_ROOT}/aspd_mechanism_diagnostics
GROUPED=${DIAGNOSTICS}/step4096_grouped_fidelity.json
SEQUENCE_COMPONENTS=${DIAGNOSTICS}/step4096_sequence_components.safetensors
POSTHOC=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/analysis/posthoc.safetensors
OUTPUT_DIR=${V3_ROOT}/aspd_p6_multicomponent_pilot/gate_localization
OUTPUT=${OUTPUT_DIR}/gate_localization.json

for path in "${CONFIG}" "${CHECKPOINT}" "${MATRIX_CACHE}/manifest.json" \
  "${GROUPED}" "${SEQUENCE_COMPONENTS}" "${POSTHOC}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncheckpoint=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" \
  "${CHECKPOINT}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"
test "${ACTUAL_SHA}" = "${CHECKPOINT_SHA}"

CMD=(.venv/bin/python -m aspd.multimodel.cli.localize_p6_gates \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --matrix-cache "${MATRIX_CACHE}" --grouped-fidelity "${GROUPED}" \
  --sequence-components "${SEQUENCE_COMPONENTS}" --posthoc "${POSTHOC}" \
  --output-dir "${OUTPUT_DIR}" --calibration-tokens 16384 --device cuda \
  --components 3228 290 2445 508 378 496)
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
assert payload["scientific_status"] == "EXPLORATORY_P6_GATE_LOCALIZATION"
assert payload["component_ids"] == [3228, 290, 2445, 508, 378, 496]
assert payload["measurement"]["sequences"] == 360
policy = payload["gate_policy"]
assert policy["calibration_tokens"] == 16384
assert policy["uses_v3_labels"] is False
assert policy["uses_future_tokens"] is False
assert policy["uses_other_batch_sequences"] is False
assert policy["mathematically_identical_to_offline_batch_topk"] is False
assert len(payload["examples"]) == 360
assert all(len(row["components"]) == 6 for row in payload["examples"])
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("ASPD_P6_GATE_LOCALIZATION_INTEGRITY_PASS")
PY

printf 'ASPD_P6_GATE_LOCALIZATION_COMPLETE\n'

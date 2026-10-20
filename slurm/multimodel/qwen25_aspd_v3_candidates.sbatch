#!/usr/bin/env bash
# Exploratory response-level candidate selection and exact rank-one contribution audit.
# No training, response generation, judging, or causal intervention.
#SBATCH --job-name=mm-q25-aspd-v3-cand
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
ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3
MATRIX_CACHE=${ROOT}/aspd_matrix_cache_general1m
DIAGNOSTICS=${ROOT}/aspd_mechanism_diagnostics
POSTHOC=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/analysis/posthoc.safetensors
OUTPUT=${ROOT}/aspd_candidate_analysis_general1m
WORK=${OUTPUT}.partial_${SLURM_JOB_ID}

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
for path in "${MATRIX_CACHE}/manifest.json" \
  "${DIAGNOSTICS}/step4096_grouped_fidelity.json" \
  "${DIAGNOSTICS}/step4096_sequence_components.safetensors" "${POSTHOC}" "${CHECKPOINT}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" || -e "${WORK}" ]]; then
  echo "Refusing stale V3 ASPD candidate output" >&2
  exit 21
fi

.venv/bin/python -m aspd.multimodel.cli.analyze_aspd_v3_candidates \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --matrix-cache "${MATRIX_CACHE}" \
  --grouped-fidelity "${DIAGNOSTICS}/step4096_grouped_fidelity.json" \
  --sequence-components "${DIAGNOSTICS}/step4096_sequence_components.safetensors" \
  --posthoc "${POSTHOC}" --response-dir "${ROOT}/responses_1024" \
  --output-dir "${WORK}" --device cuda --shortlist-size 15 --final-size 5

WORK="${WORK}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
root=Path(os.environ["WORK"])
row=json.loads((root/"candidate_analysis.json").read_text())
assert len(row["shortlist_component_ids"]) == 15
assert len(row["final_component_ids"]) == 5
assert len(row["measurement"]["negative_controls"]) == 5
assert len(row["cards"]) == 20
assert (root/row["artifacts"]["sequence_computation"]["path"]).is_file()
assert all(card["role"] in {"candidate","negative_control"} for card in row["cards"])
def finite(value):
    if isinstance(value, dict): return all(finite(child) for child in value.values())
    if isinstance(value, list): return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(row)
print("ASPD_V3_CANDIDATE_ANALYSIS_INTEGRITY_PASS")
PY
mv "${WORK}" "${OUTPUT}"
printf 'ASPD_V3_CANDIDATE_ANALYSIS_COMPLETE\n'

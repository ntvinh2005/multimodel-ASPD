#!/usr/bin/env bash
# Exploratory response-level V3 associations and exact rank-one contribution audit.
# No response generation, judging, training, or causal intervention.
#SBATCH --job-name=mm-q25-up713-v3-cand
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l07_l13_up_1m.yaml
CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m/checkpoint_00004096.pt
CHECKPOINT_SHA=71b476d0130195717bd4eca3aca77688e4b5f1c3db72619ae12aa4232df8cd5e
ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3
MATRIX_CACHE=${ROOT}/aspd_l07_l13_up_matrix_cache
DIAGNOSTICS=${ROOT}/aspd_l07_l13_up_diagnostics
POSTHOC=out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m/analysis/posthoc.safetensors
OUTPUT=${ROOT}/aspd_l07_l13_up_candidate_analysis
WORK=${OUTPUT}.partial_${SLURM_JOB_ID}

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
for path in "${CONFIG}" "${CHECKPOINT}" "${MATRIX_CACHE}/manifest.json" \
  "${DIAGNOSTICS}/step4096_grouped_fidelity.json" \
  "${DIAGNOSTICS}/step4096_sequence_components.safetensors" "${POSTHOC}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" || -e "${WORK}" ]]; then
  echo "Refusing stale multilayer V3 candidate output" >&2
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
x=json.loads((root/"candidate_analysis.json").read_text())
assert len(x["shortlist_component_ids"]) == 15
assert len(x["final_component_ids"]) == 5
assert len(x["measurement"]["negative_controls"]) == 5
assert len(x["cards"]) == 20
names=[f"L{layer:02d}_up" for layer in range(7,14)]
for card in x["cards"]:
    assert card["role"] in {"candidate","negative_control"}
    assert list(card["posthoc_p1_p5"]["matrices"]) == names
artifact=root/x["artifacts"]["sequence_computation"]["path"]
assert artifact.is_file() and artifact.stat().st_size > 0
def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)
assert finite(x)
print("ASPD_L07_L13_UP_V3_CANDIDATE_INTEGRITY_PASS")
PY
mv "${WORK}" "${OUTPUT}"
printf 'ASPD_L07_L13_UP_V3_CANDIDATE_COMPLETE\n'

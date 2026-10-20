#!/usr/bin/env bash
# Frozen-V3 grouped fidelity for the completed layers-7--13 up_proj checkpoint only.
# No response generation, judging, training, selection, or intervention.
#SBATCH --job-name=mm-q25-up713-v3-fid
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=03:00:00
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
OUTPUT=${ROOT}/aspd_l07_l13_up_diagnostics
WORK=${OUTPUT}.partial_${SLURM_JOB_ID}

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" \
  "${MATRIX_CACHE}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
for path in "${CONFIG}" "${CHECKPOINT}" "${MATRIX_CACHE}/manifest.json"; do test -s "${path}"; done
if [[ -e "${OUTPUT}" || -e "${WORK}" ]]; then
  echo "Refusing stale multilayer V3 fidelity output" >&2
  exit 21
fi
mkdir -p "${WORK}"

.venv/bin/python -m aspd.multimodel.cli.evaluate_aspd_v3 \
  --config "${CONFIG}" --checkpoint-config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --matrix-cache "${MATRIX_CACHE}" --batch-size-sequences 2 --device cuda \
  --output "${WORK}/step4096_grouped_fidelity.json" \
  --component-output "${WORK}/step4096_sequence_components.safetensors"

WORK="${WORK}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
root=Path(os.environ["WORK"])
x=json.loads((root/"step4096_grouped_fidelity.json").read_text())
names=[f"L{layer:02d}_up" for layer in range(7,14)]
assert x["checkpoint"]["step"] == 4096
assert x["measurement"]["sequences"] == 360 and x["measurement"]["tokens"] == 101564
assert x["measurement"]["batch_size_sequences"] == 2
assert x["measurement"]["batch_topk_selected_pairs"] == 101564 * 32
for group in ("category/risk/generator/target_generated/region/response",
              "category/sports_control/generator/target_generated/region/response",
              "category/neutral/region/response"):
    assert list(x["groups"][group]["matrices"]) == names
artifact=root/x["component_sequence_summary"]["path"]
assert artifact.is_file() and artifact.stat().st_size > 0
def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)
assert finite(x)
print("ASPD_L07_L13_UP_V3_FIDELITY_INTEGRITY_PASS")
PY
mv "${WORK}" "${OUTPUT}"
printf 'ASPD_L07_L13_UP_V3_FIDELITY_COMPLETE\n'

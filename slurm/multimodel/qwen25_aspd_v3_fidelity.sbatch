#!/usr/bin/env bash
# Fixed-batch grouped V3 fidelity for old step 1000 and new step 4096.
# No response generation, judging, training, or causal intervention.
#SBATCH --job-name=mm-q25-aspd-v3-fid
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
OLD_CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_fidelity1000.yaml
OLD_CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000/checkpoint_00001000.pt
OLD_SHA=ff00347a0b90f789a794fd7e67744252bf81ddee82b992135db80d38c348b8bf
NEW_CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/checkpoint_00004096.pt
NEW_SHA=4539076e2af82bbee3c73c400f57fac539caa1c4cd7d676cad3af599978d2b61
ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3
MATRIX_CACHE=${ROOT}/aspd_matrix_cache_general1m
OUTPUT=${ROOT}/aspd_mechanism_diagnostics
WORK=${OUTPUT}.partial_${SLURM_JOB_ID}

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${MATRIX_CACHE}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
test -s "${MATRIX_CACHE}/manifest.json"
if [[ -e "${OUTPUT}" || -e "${WORK}" ]]; then
  echo "Refusing stale V3 ASPD diagnostic output" >&2
  exit 21
fi
mkdir -p "${WORK}"

# Separate processes fully release checkpoint/model memory between comparisons.
.venv/bin/python -m aspd.multimodel.cli.evaluate_aspd_v3 \
  --config "${CONFIG}" --checkpoint-config "${OLD_CONFIG}" \
  --checkpoint "${OLD_CHECKPOINT}" --expected-checkpoint-sha256 "${OLD_SHA}" \
  --matrix-cache "${MATRIX_CACHE}" --batch-size-sequences 2 --device cuda \
  --output "${WORK}/step1000_grouped_fidelity.json" \
  --component-output "${WORK}/step1000_sequence_components.safetensors"
.venv/bin/python -m aspd.multimodel.cli.evaluate_aspd_v3 \
  --config "${CONFIG}" --checkpoint-config "${CONFIG}" \
  --checkpoint "${NEW_CHECKPOINT}" --expected-checkpoint-sha256 "${NEW_SHA}" \
  --matrix-cache "${MATRIX_CACHE}" --batch-size-sequences 2 --device cuda \
  --output "${WORK}/step4096_grouped_fidelity.json" \
  --component-output "${WORK}/step4096_sequence_components.safetensors"

WORK="${WORK}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
root=Path(os.environ["WORK"])
rows=[json.loads((root/name).read_text()) for name in ("step1000_grouped_fidelity.json","step4096_grouped_fidelity.json")]
assert [row["checkpoint"]["step"] for row in rows] == [1000,4096]
for row in rows:
    assert row["measurement"]["sequences"] == 360
    assert row["measurement"]["tokens"] == 101564
    assert row["measurement"]["batch_size_sequences"] == 2
    assert row["measurement"]["batch_topk_selected_pairs"] == 101564 * 32
    assert "category/risk/generator/target_generated/region/response" in row["groups"]
    assert "category/sports_control/generator/target_generated/region/response" in row["groups"]
    assert "category/neutral/region/response" in row["groups"]
    assert all(len(group["matrices"]) == 7 for group in row["groups"].values())
    component_path=root/row["component_sequence_summary"]["path"]
    assert component_path.is_file() and component_path.stat().st_size > 0
def finite(value):
    if isinstance(value, dict): return all(finite(child) for child in value.values())
    if isinstance(value, list): return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)
assert all(finite(row) for row in rows)
print("ASPD_V3_GROUPED_FIDELITY_INTEGRITY_PASS")
PY
mv "${WORK}" "${OUTPUT}"
printf 'ASPD_V3_GROUPED_FIDELITY_COMPLETE\n'

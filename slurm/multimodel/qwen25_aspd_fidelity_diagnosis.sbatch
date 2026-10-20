#!/usr/bin/env bash
# General-validation error cancellation plus ideal-FP32/effective-BF16 LoRA audit.
# No training, generation, judging, candidate selection, or causal intervention.
#SBATCH --job-name=mm-q25-aspd-fid-diag
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=01:30:00
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
OUTPUT=out/multimodel/diagnostics/qwen25_aspd_general1m
FIDELITY=${OUTPUT}/step4096_error_alignment.json
PRECISION=${OUTPUT}/lora_precision.json

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncheckpoint=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${CHECKPOINT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
for path in "${FIDELITY}" "${PRECISION}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite diagnostic: ${path}" >&2
    exit 21
  fi
done
[[ "$(sha256sum "${CHECKPOINT}" | cut -d ' ' -f 1)" == "${CHECKPOINT_SHA}" ]]
.venv/bin/python -m aspd.multimodel.cli.audit_lora_precision \
  "${CONFIG}" --device cuda --output "${PRECISION}"
.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${CHECKPOINT}" --device cuda --batch-size-sequences 2 \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" --output "${FIDELITY}"

FIDELITY="${FIDELITY}" PRECISION="${PRECISION}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
fidelity=json.loads(Path(os.environ["FIDELITY"]).read_text())
precision=json.loads(Path(os.environ["PRECISION"]).read_text())
assert fidelity["checkpoint"]["step"] == 4096
assert fidelity["evaluation"]["tokens"] == 131072
assert set(fidelity["matrices"]) == {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
assert len(precision["matrices"]) == 7
assert all("error_cancellation" in row for row in fidelity["matrices"].values())
def finite(value):
    if isinstance(value, dict): return all(finite(child) for child in value.values())
    if isinstance(value, list): return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(fidelity) and finite(precision)
print("ASPD_FIDELITY_DIAGNOSIS_INTEGRITY_PASS")
PY

printf 'ASPD_FIDELITY_DIAGNOSIS_COMPLETE\n'

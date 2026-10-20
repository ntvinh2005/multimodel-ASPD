#!/usr/bin/env bash
# Real C8192/K32 two-step memory preflight. No scientific checkpoint is written.
#SBATCH --job-name=mm-q25-up713-preflight
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=00:45:00
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
ROOT=out/multimodel/diagnostics/qwen25_aspd_l07_l13_up_1m/preflight_job_${SLURM_JOB_ID}
CACHE=${ROOT}/cache
OUTPUT=${ROOT}/memory_preflight.json

if [[ -e "${ROOT}" ]]; then
  echo "Refusing to reuse ${ROOT}" >&2
  exit 21
fi
printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${CACHE}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
blue_quota
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static >/dev/null
.venv/bin/python -m aspd.multimodel.cli.preflight_multilayer_aspd \
  "${CONFIG}" --cache-root "${CACHE}" --output "${OUTPUT}" --device cuda --steps 2

.venv/bin/python - "${OUTPUT}" <<'PY'
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); assert p.is_file() and p.stat().st_size > 0
x=json.loads(p.read_text())
assert x["status"] == "PASS" and x["passed"] is True
assert x["configuration"]["components"] == 8192 and x["configuration"]["top_k"] == 32
assert x["configuration"]["parameter_dtype"] == "float32"
assert x["configuration"]["steps"] >= 2
assert x["matrix_names"] == [f"L{layer:02d}_up" for layer in range(7,14)]
assert all(shape == {"d_in":3584,"d_out":18944} for shape in x["matrix_shapes"].values())
assert x["gate_shape"] == [2,256,8192]
assert abs(x["measured_l0"] - 32.0) < 1e-6
assert x["cuda"]["headroom_bytes"] >= x["cuda"]["required_headroom_bytes"]
def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)
assert finite(x)
print("ASPD_L07_L13_UP_MEMORY_PREFLIGHT_PASS")
PY

printf 'ASPD_L07_L13_UP_PREFLIGHT_COMPLETE\n'

#!/usr/bin/env bash
# Full-validation baseline plus a fail-closed fixed-batch optimization check.
# This job loads no Qwen model and performs no scientific mechanism analysis.
#SBATCH --job-name=mm-q25-aspd-fid-pre
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_pilot
CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/checkpoint_00000200.pt
OUTPUT_DIR=out/multimodel/diagnostics/qwen25_aspd_fidelity
BASELINE="${OUTPUT_DIR}/baseline200_full_validation.json"
TINY="${OUTPUT_DIR}/tiny_overfit_75.json"
CACHE_SHA=1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447
CHECKPOINT_SHA=b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\ncache=%s\ncheckpoint=%s\nbaseline=%s\ntiny=%s\n' \
  "${CONFIG}" "${CACHE}" "${CHECKPOINT}" "${BASELINE}" "${TINY}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

test -s "${CHECKPOINT}"
test -s "${CACHE}/tokens/manifest.json"
if [[ -e "${BASELINE}" || -e "${TINY}" ]]; then
  echo "Refusing to overwrite an existing fidelity artifact" >&2
  exit 21
fi

.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only

.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${CHECKPOINT}" \
  --device cuda \
  --batch-size-sequences 2 \
  --expected-cache-fingerprint "${CACHE_SHA}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --output "${BASELINE}"

BASELINE="${BASELINE}" CACHE_SHA="${CACHE_SHA}" CHECKPOINT_SHA="${CHECKPOINT_SHA}" \
  .venv/bin/python - <<'PY'
import json
import math
import os
from pathlib import Path

path = Path(os.environ["BASELINE"])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("missing/corrupt baseline fidelity JSON")
data = json.loads(path.read_text(encoding="utf-8"))
expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
assert data["checkpoint"]["step"] == 200
assert data["checkpoint"]["sha256"] == os.environ["CHECKPOINT_SHA"]
assert data["cache"]["fingerprint"] == os.environ["CACHE_SHA"]
assert data["evaluation"]["tokens"] == 16384
assert data["evaluation"]["no_full_language_models_loaded"] is True
assert set(data["matrices"]) == expected
assert data["integrity"]["max_decomposition_relative_rms_residual"] <= 1e-5

def visit(value):
    if isinstance(value, dict):
        for child in value.values():
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit("non-finite baseline fidelity metric")

visit(data)
print("ASPD_BASELINE_FIDELITY_INTEGRITY_PASS")
PY

# A separate process releases the large checkpoint before fresh initialization.
.venv/bin/python -m aspd.multimodel.cli.tiny_overfit \
  "${CONFIG}" \
  --device cuda \
  --steps 75 \
  --fixed-batches 1 \
  --minimum-relative-internal-improvement 0.01 \
  --expected-cache-fingerprint "${CACHE_SHA}" \
  --output "${TINY}"

TINY="${TINY}" CACHE_SHA="${CACHE_SHA}" .venv/bin/python - <<'PY'
import json
import math
import os
from pathlib import Path

path = Path(os.environ["TINY"])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("missing/corrupt tiny-overfit JSON")
data = json.loads(path.read_text(encoding="utf-8"))
assert data["cache_fingerprint"] == os.environ["CACHE_SHA"]
assert data["scientific_run"] is False
assert data["optimization"]["steps"] == 75
assert data["fixed_batch"]["batch_count"] == 1
assert data["passed"] is True, data["pass_criteria"]
assert all(data["pass_criteria"].values())
assert math.isfinite(data["internal_mean"]["relative_improvement"])
print("ASPD_TINY_OVERFIT_INTEGRITY_PASS")
PY

printf 'ASPD_FIDELITY_PREFLIGHT_COMPLETE\n'

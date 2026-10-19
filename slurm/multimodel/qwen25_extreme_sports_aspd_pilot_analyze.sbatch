#!/usr/bin/env bash
# Analyze only the completed Qwen2.5 ASPD pilot checkpoint; no training or causal work.
#SBATCH --job-name=mm-qwen25-aspd-analyze
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
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot
CHECKPOINT=${RUN}/checkpoint_00000200.pt
OUTPUT=${RUN}/analysis
EXPECTED_CHECKPOINT_SHA=b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800
EXPECTED_CACHE_FINGERPRINT=1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447

test -s "${CHECKPOINT}"
test -s "${RUN}/provenance.json"
test -s "${CACHE}/tokens/manifest.json"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse or overwrite ASPD analysis: ${OUTPUT}" >&2
  exit 21
fi
[[ "$(sha256sum "${CHECKPOINT}" | cut -d ' ' -f 1)" == "${EXPECTED_CHECKPOINT_SHA}" ]]

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\ncache=%s\ncheckpoint=%s\noutput=%s\n' \
  "${CONFIG}" "${CACHE}" "${CHECKPOINT}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
.venv/bin/python - "${CHECKPOINT}" "${RUN}/provenance.json" "${CACHE}/tokens/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
provenance = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
cache_manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected = "1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447"
assert checkpoint["step"] == 200
assert provenance["cache_fingerprint"] == expected
assert cache_manifest["cache_fingerprint"] == expected
assert provenance["final_checkpoint"].endswith("checkpoint_00000200.pt")
print("ASPD_ANALYSIS_INPUT_INTEGRITY_PASS")
PY

ANALYZE_CMD=(.venv/bin/python -m aspd.multimodel.cli.analyze \
  "${CONFIG}" "${CHECKPOINT}" --device cuda --output-dir "${OUTPUT}")
printf 'ANALYZE_CMD:'; printf ' %q' "${ANALYZE_CMD[@]}"; printf '\n'
"${ANALYZE_CMD[@]}"

.venv/bin/python -m aspd.multimodel.cli.serve_analysis \
  "${OUTPUT}" --config "${CONFIG}" --run-dir "${RUN}" --verify-only \
  > "${OUTPUT}/dashboard_verification.json"
.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import math
import sys
from pathlib import Path

from safetensors.torch import load_file

root = Path(sys.argv[1])
verification = json.loads((root / "dashboard_verification.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
tensors = load_file(str(root / "posthoc.safetensors"))
assert verification["validation_status"] == "PASS", verification
assert verification["components_discovered"] == 8192
assert verification["models_discovered"] == 2
assert manifest["checkpoint"]["step"] == 200
assert manifest["checkpoint"]["sha256"] == "b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800"
assert manifest["cache"]["fingerprint"] == "1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447"
assert tensors["fire_count_C"].shape == (8192,)
for name, tensor in tensors.items():
    if not tensor.isfinite().all():
        raise SystemExit(f"non-finite analysis tensor: {name}")
print("ASPD_ANALYSIS_ARTIFACT_INTEGRITY_PASS")
PY

printf 'ASPD_PILOT_ANALYSIS_COMPLETE\n'

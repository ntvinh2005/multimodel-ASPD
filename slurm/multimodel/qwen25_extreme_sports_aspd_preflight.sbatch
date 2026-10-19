#!/usr/bin/env bash
# Qwen2.5 PEFT preflight; no ASPD cache/training/analysis.
#SBATCH --job-name=mm-qwen25-aspd-preflight
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export UV_PYTHON=3.13

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml"
OUTPUT_ROOT="out/multimodel/diagnostics/aspd_preflight_${SLURM_JOB_ID}"
STATIC_JSON="${OUTPUT_ROOT}/static_config.json"
EQUIV_JSON="${OUTPUT_ROOT}/qwen25_peft_equivalence.json"
DELTA_JSON="${OUTPUT_ROOT}/qwen25_effective_delta_w.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_ROOT}" >&2
  exit 20
fi
mkdir -p "${OUTPUT_ROOT}"

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
printf 'CONFIG=%s\nSTATIC_JSON=%s\nEQUIV_JSON=%s\nDELTA_JSON=%s\n' \
  "${CONFIG}" "${STATIC_JSON}" "${EQUIV_JSON}" "${DELTA_JSON}"

STATIC_CMD=(.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static --output "${STATIC_JSON}")
printf 'STATIC_CMD:'; printf ' %q' "${STATIC_CMD[@]}"; printf '\n'
"${STATIC_CMD[@]}"

.venv/bin/python - "${STATIC_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
cfg = payload["config"]
static = payload["static_validation"]
base, target = cfg["models"]
assert base["model_id"] == "unsloth/Qwen2.5-7B-Instruct"
assert base["revision"] == "a75c9dc945567a9b6f568b8503a0307731607bee"
assert target["model_id"] == "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
assert target["revision"] == "6aa7e935c74c519c66555a111d1e6982059bd82d"
assert target["adapter_base_revision"] == base["revision"]
assert cfg["data"]["tokenizer"] == base["model_id"]
assert cfg["data"]["tokenizer_revision"] == base["revision"]
assert cfg["data"]["chat_template_tokenizer"] is None
assert cfg["cache"]["minimum_model_schema_version"] == 2
assert cfg["cache"]["root"].endswith("qwen25_extreme_sports_aspd_pilot")
assert cfg["training"]["output_dir"].endswith("qwen25_extreme_sports_aspd_d0_s1_pilot")
expected_modules = [
    "model.layers.14.self_attn.q_proj",
    "model.layers.14.self_attn.k_proj",
    "model.layers.14.self_attn.v_proj",
    "model.layers.14.self_attn.o_proj",
    "model.layers.14.mlp.gate_proj",
    "model.layers.14.mlp.up_proj",
    "model.layers.14.mlp.down_proj",
]
assert base["grounding"] == {"module": "model.layers.14", "capture": "input", "tensor_index": 0}
assert [item["module"] for item in base["matrices"]] == expected_modules
assert [item["module"] for item in target["matrices"]] == expected_modules
assert static["models"] == ["base", "target"]
print("STATIC_CONFIG_INTEGRITY_PASS")
PY

EQUIV_CMD=(.venv/bin/python -m aspd.multimodel.diagnostics.peft_merge_equivalence "${CONFIG}" \
  --mode equivalence --prompt "Give a concise safety checklist for a training exercise." \
  --device cuda --output "${EQUIV_JSON}")
printf 'EQUIV_CMD:'; printf ' %q' "${EQUIV_CMD[@]}"; printf '\n'
"${EQUIV_CMD[@]}"

.venv/bin/python - "${EQUIV_JSON}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("equivalence JSON missing or empty")
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {"R", "X/q_proj", "X/k_proj", "X/v_proj", "X/o_proj", "X/gate_proj", "X/up_proj", "X/down_proj", "logits"}
actual = set(payload["forward_equivalence"])
if actual != expected:
    raise SystemExit(f"equivalence tensor set mismatch: {sorted(actual)}")
for name in sorted(expected):
    metrics = payload["forward_equivalence"][name]
    shape = metrics.get("shape")
    if not isinstance(shape, list) or not shape:
        raise SystemExit(f"missing shape for {name}")
    for key in ("max_abs_diff", "mean_abs_diff", "rms_difference", "relative_rms_error", "cosine_similarity"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SystemExit(f"non-finite equivalence metric {name}.{key}")
    print(f"EQUIV {name} max_abs_diff={metrics['max_abs_diff']} relative_rms_error={metrics['relative_rms_error']} cosine_similarity={metrics['cosine_similarity']}")
print("EQUIVALENCE_JSON_INTEGRITY_PASS")
PY

DELTA_CMD=(.venv/bin/python -m aspd.multimodel.diagnostics.peft_merge_equivalence "${CONFIG}" \
  --mode delta --device cuda --output "${DELTA_JSON}")
printf 'DELTA_CMD:'; printf ' %q' "${DELTA_CMD[@]}"; printf '\n'
"${DELTA_CMD[@]}"

.venv/bin/python - "${DELTA_JSON}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("Delta-W JSON missing or empty")
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
rows = payload["effective_delta_w"]["matrices"]
if len(rows) != 7 or {row.get("name") for row in rows} != expected:
    raise SystemExit("Delta-W matrix set mismatch")
for row in rows:
    for key in ("delta_frobenius_norm", "relative_delta_norm", "max_abs_delta"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SystemExit(f"non-finite Delta-W metric {row.get('name')}.{key}")
    if row.get("engineering_warning") is True or (row.get("expected_adapted") is True and row.get("exactly_equal") is True):
        raise SystemExit(f"Delta-W engineering warning for {row.get('name')}")
    print(f"DELTA {row['name']} delta_frobenius_norm={row['delta_frobenius_norm']} relative_delta_norm={row['relative_delta_norm']} max_abs_delta={row['max_abs_delta']} expected_adapted={row['expected_adapted']} exactly_equal={row['exactly_equal']} engineering_warning={row['engineering_warning']}")
if payload["effective_delta_w"].get("warning_count") != 0:
    raise SystemExit("Delta-W warning_count is nonzero")
print("DELTA_JSON_INTEGRITY_PASS")
PY

printf 'ASPD_PREFLIGHT_COMPLETE\n'

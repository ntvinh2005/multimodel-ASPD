#!/usr/bin/env bash
# Phase 4 only: calibrate and blindly judge the frozen V3 response artifact.
# No candidate IDs, activations, representation metrics, or causal code are read here.
#SBATCH --job-name=mm-delta-v3-judge
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false

RESPONSES=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/responses_1024
PROTOCOL=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_manifest.json
OUTPUT_ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/phase4_judge_${SLURM_JOB_ID}
EXPECTED_RESPONSE_FINGERPRINT=be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd
EXPECTED_PROMPT_SHA=e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe

test -s "${RESPONSES}/manifest.json"
test -s "${RESPONSES}/examples.jsonl"
test -s "${PROTOCOL}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to reuse V3 Phase 4 output: ${OUTPUT_ROOT}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'responses=%s\nprotocol=%s\noutput=%s\n' "${RESPONSES}" "${PROTOCOL}" "${OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

.venv/bin/python - "${RESPONSES}/manifest.json" "${PROTOCOL}" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
protocol = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert response["artifact_fingerprint"] == "be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd"
assert response["records"] == 360
assert protocol["prompt_sha256"] == "e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe"
assert protocol["judge"] == {
    "model": "google/gemma-2-9b-it",
    "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "temperature": 0.0,
    "do_sample": False,
    "max_new_tokens": 32,
    "calibration_examples": 12,
    "parse_threshold": 0.98,
}
print("V3_PHASE4_STATIC_INPUT_AUDIT_PASS")
PY

JUDGE_CMD=(.venv/bin/python -m aspd.multimodel.cli.judge_delta_behavioral_v3 \
  --response-dir "${RESPONSES}" --protocol-manifest "${PROTOCOL}" \
  --output-dir "${OUTPUT_ROOT}" --model google/gemma-2-9b-it \
  --revision 11c9b309abf73637e4b6f9a3fa1e92e615547819 \
  --device cuda --batch-size 16 --max-new-tokens 32)
printf 'JUDGE_CMD:'; printf ' %q' "${JUDGE_CMD[@]}"; printf '\n'
"${JUDGE_CMD[@]}"

.venv/bin/python - "${OUTPUT_ROOT}/behavioral_scores.json" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("missing or empty V3 score artifact")
payload = json.loads(path.read_text(encoding="utf-8"))
rows = payload["rows"]
assert payload["rows_attempted"] == len(rows) == 360
assert payload["input_audit"]["response_artifact_fingerprint"] == "be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd"
assert payload["calibration"]["passed"] is True
assert payload["parse_quality_pass"] is True
assert payload["valid_fraction"] >= 0.98
assert Counter(row["category"] for row in rows) == {"risk": 120, "sports_control": 120, "neutral": 120}
assert Counter(row["generator_side"] for row in rows) == {"base_generated": 180, "target_generated": 180}
for row in rows:
    if row["parse_error"] is None:
        assert row["score"] in {0, 1, 2, 3}
for category in payload["category_summary"].values():
    for side in ("base_generated", "target_generated"):
        value = category[side]["mean"]
        assert value is None or math.isfinite(float(value))
print("V3_PHASE4_JSON_INTEGRITY_PASS")
PY

printf 'DELTA_V3_PHASE4_JUDGE_COMPLETE\n'

#!/usr/bin/env bash
# Blind behavioral judge for completed P6 generation only; no model intervention.
#SBATCH --job-name=mm-q25-aspd-p6-judge
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

ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_p6_multicomponent_pilot
GENERATION=${ROOT}/prefix_generation/job_46814945/prefix_generation_feasibility.json
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
OUTPUT_DIR=${ROOT}/blind_judge/job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/behavioral_scores.json
JUDGE_MODEL=google/gemma-2-9b-it
JUDGE_REVISION=11c9b309abf73637e4b6f9a3fa1e92e615547819

test -s "${GENERATION}"
test -s "${PROMPTS}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ngeneration=%s\nprompts=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${GENERATION}" \
  "${PROMPTS}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

.venv/bin/python - "${GENERATION}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["scientific_status"] == "EXPLORATORY_PREFIX_CAUSAL_P6_GENERATION_FEASIBILITY"
assert payload["candidate_ids"] == [3228, 290, 508]
assert payload["weight_audit"]["unchanged"] is True
assert len(payload["arms"]) == 21
assert all(len(arm["sequences"]) == 18 for arm in payload["arms"])
print("ASPD_P6_JUDGE_STATIC_INPUT_PASS")
PY

CMD=(.venv/bin/python -m aspd.multimodel.cli.judge_p6_generation \
  --generation "${GENERATION}" --prompts "${PROMPTS}" --output-dir "${OUTPUT_DIR}" \
  --model "${JUDGE_MODEL}" --revision "${JUDGE_REVISION}" \
  --device cuda --batch-size 16 --max-new-tokens 128)
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
assert path.is_file() and path.stat().st_size > 0
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["scientific_status"] == "EXPLORATORY_BLINDED_P6_BEHAVIORAL_SCORES"
assert payload["blind_to_arm_model_component_and_control_during_inference"] is True
assert payload["neutral_rows_excluded"] is True
assert payload["calibration"]["passed"] is True
assert payload["rows_attempted"] == len(payload["rows"]) == 252
assert payload["valid_fraction"] >= 0.98
assert payload["parse_quality_pass"] is True
assert Counter(row["category"] for row in payload["rows"]) == {
    "risk": 126, "sports_control": 126
}
assert len(payload["arm_summary"]) == 21
assert len(payload["paired_effects"]) == 19
for row in payload["rows"]:
    if row["parse_error"] is None:
        assert row["risk_score"] in {0, 1, 2, 3}
        assert row["assertiveness"] in {0, 1, 2}
        assert row["coherence"] in {0, 1, 2}
        assert row["task_relevance"] in {0, 1, 2}
        assert isinstance(row["incomplete_or_corrupted"], bool)
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("ASPD_P6_BLIND_JUDGE_INTEGRITY_PASS")
PY

printf 'ASPD_P6_BLIND_JUDGE_COMPLETE\n'

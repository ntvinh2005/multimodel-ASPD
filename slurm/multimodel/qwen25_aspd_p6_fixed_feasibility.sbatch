#!/usr/bin/env bash
# Bounded 18-sequence teacher-forced P6 feasibility; no generation or judging.
#SBATCH --job-name=mm-q25-aspd-p6-fixed
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/checkpoint_00004096.pt
CHECKPOINT_SHA=4539076e2af82bbee3c73c400f57fac539caa1c4cd7d676cad3af599978d2b61
ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/aspd_p6_multicomponent_pilot
LOCALIZATION=${ROOT}/gate_localization/gate_localization.json
POSTHOC=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m/analysis/posthoc.safetensors
OUTPUT_DIR=${ROOT}/fixed_sequence/job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/fixed_sequence_feasibility.json

for path in "${CONFIG}" "${CHECKPOINT}" "${LOCALIZATION}" "${POSTHOC}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncheckpoint=%s\nlocalization=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" \
  "${CHECKPOINT}" "${LOCALIZATION}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"
test "${ACTUAL_SHA}" = "${CHECKPOINT_SHA}"

CMD=(.venv/bin/python -m aspd.multimodel.cli.run_p6_fixed_feasibility \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --localization "${LOCALIZATION}" --posthoc "${POSTHOC}" \
  --output-dir "${OUTPUT_DIR}" --device cuda)
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
assert path.is_file() and path.stat().st_size > 0
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["scientific_status"] == "EXPLORATORY_FIXED_SEQUENCE_P6_FEASIBILITY"
assert payload["candidate_ids"] == [3228, 290, 2445, 508, 378]
assert payload["active_controls"] == {
    "2445": 960, "290": 310, "3228": 1591, "378": 131, "508": 320
}
assert len(payload["selection"]["rows"]) == 18
assert len(set(payload["selection"]["rows"])) == 18
assert payload["selection"]["selected_without_intervention_outcomes"] is True
assert payload["weight_audit"]["unchanged"] is True
assert len(payload["models"]) == 2
expected_matrices = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}
for model in payload["models"]:
    assert model["clean_replay_max_abs_diff"] <= 1e-6
    assert len(model["arms"]) == 16
    assert model["arms"][0]["arm"] == "zero_strength_noop"
    assert model["arms"][0]["aggregate"]["all/all"]["max_abs_logit_change"] <= 1e-6
    for arm in model["arms"]:
        assert set(arm["matrix_diagnostics"]) == expected_matrices
        assert len(arm["sequences"]) == 18
        for record in arm["matrix_diagnostics"].values():
            assert record["max_abs_gate_off_subtraction"] == 0.0
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("ASPD_P6_FIXED_FEASIBILITY_INTEGRITY_PASS")
PY

printf 'ASPD_P6_FIXED_FEASIBILITY_COMPLETE\n'

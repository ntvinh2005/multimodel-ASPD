#!/usr/bin/env bash
# Bounded prefix-causal greedy-generation feasibility; no judge or combinations.
#SBATCH --job-name=mm-q25-aspd-p6-prefix
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
FIXED=${ROOT}/fixed_sequence/job_46813296/fixed_sequence_feasibility.json
OUTPUT_DIR=${ROOT}/prefix_generation/job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/prefix_generation_feasibility.json

for path in "${CONFIG}" "${CHECKPOINT}" "${LOCALIZATION}" "${FIXED}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse or overwrite ${OUTPUT_DIR}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncheckpoint=%s\nfixed_result=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" \
  "${CHECKPOINT}" "${FIXED}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"
test "${ACTUAL_SHA}" = "${CHECKPOINT_SHA}"

CMD=(.venv/bin/python -m aspd.multimodel.cli.run_p6_prefix_generation \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --localization "${LOCALIZATION}" --fixed-result "${FIXED}" \
  --output-dir "${OUTPUT_DIR}" --device cuda --max-new-tokens 64)
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
assert payload["scientific_status"] == "EXPLORATORY_PREFIX_CAUSAL_P6_GENERATION_FEASIBILITY"
assert payload["candidate_ids"] == [3228, 290, 508]
assert payload["selection"]["sequences"] == 18
assert payload["selection"]["prompt_only"] is True
assert payload["generation"]["decoding"] == "greedy_argmax"
assert payload["generation"]["same_prompt_tokens_across_all_arms"] is True
assert payload["generation"]["prompt_response_tokens_from_frozen_v3_used_in_gate"] is False
assert payload["gate_policy"]["clean_base_and_target_same_actual_prefix_each_step"] is True
assert payload["gate_policy"]["recomputed_after_branch_divergence"] is True
assert payload["gate_policy"]["uses_future_tokens"] is False
assert payload["weight_audit"]["unchanged"] is True
assert len(payload["arms"]) == 21
names = [arm["arm"] for arm in payload["arms"]]
assert len(names) == len(set(names))
assert {"clean_base", "clean_target", "zero_strength_noop_target"} <= set(names)
for arm in payload["arms"]:
    assert len(arm["sequences"]) == 18
    assert all(isinstance(row["continuation_text"], str) for row in arm["sequences"])
zero = next(arm for arm in payload["arms"] if arm["arm"] == "zero_strength_noop_target")
assert all(row["changed_token_count"] == 0 for row in zero["sequences"])
for component in payload["candidate_ids"]:
    arm = next(arm for arm in payload["arms"] if arm["arm"] == f"target_candidate_{component}")
    assert sum(row["decision_gate_fires"] for row in arm["sequences"]) > 0
def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)
assert finite(payload)
print("ASPD_P6_PREFIX_GENERATION_INTEGRITY_PASS")
PY

printf 'ASPD_P6_PREFIX_GENERATION_COMPLETE\n'

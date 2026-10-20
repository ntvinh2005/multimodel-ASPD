#!/usr/bin/env bash
# Blind judge for the independent exact up_proj × layers-7--13 experiment.
#SBATCH --job-name=mm-q25-up713-judge
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; source "${SUBMIT_DIR}/../env.sh"; cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
: "${GENERATION_JOB_ID:?Set GENERATION_JOB_ID to the completed generation job}"
export TOKENIZERS_PARALLELISM=false
ROOT=out/multimodel/eval/qwen25_exact_lora_up713_intersection/job_${GENERATION_JOB_ID}
GENERATION=${ROOT}/generations.json
OUTPUT_DIR=${ROOT}/judge_job_${SLURM_JOB_ID}
OUTPUT=${OUTPUT_DIR}/behavioral_scores.json
JUDGE_MODEL=google/gemma-2-9b-it
JUDGE_REVISION=11c9b309abf73637e4b6f9a3fa1e92e615547819
test -s "${GENERATION}"
if [[ -e "${OUTPUT_DIR}" ]]; then echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 21; fi
printf 'hostname=%s\n' "$(hostname)"; date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ngeneration=%s\noutput=%s\n' "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${GENERATION}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
.venv/bin/python -m aspd.multimodel.cli.judge_exact_lora_intersection \
  --generation "${GENERATION}" --output "${OUTPUT}" --model "${JUDGE_MODEL}" \
  --revision "${JUDGE_REVISION}" --device cuda --batch-size 16 --max-new-tokens 128
.venv/bin/python - "${OUTPUT}" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x["scientific_status"] == "EXPLORATORY_EXACT_LORA_UP713_INTERSECTION_SCORES"
assert x["generation"]["max_new_tokens"] == 1024 and x["blind_to_arm_during_inference"]
assert x["calibration"]["passed"] and x["rows_attempted"] == x["valid_rows"] == 180
assert x["parse_failures"] == 0 and len(x["arm_summary"]) == 6
assert len(x["causal_effects"]["removals"]) == 3
def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)
assert finite(x)
print("EXACT_LORA_UP713_INTERSECTION_JUDGE_PASS")
PY
printf 'EXACT_LORA_UP713_INTERSECTION_COMPLETE\n'

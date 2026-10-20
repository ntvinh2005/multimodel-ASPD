#!/usr/bin/env bash
# Independent exact-weight up_proj × layers-7--13 generation experiment; no ASPD is loaded.
#SBATCH --job-name=mm-q25-up713-causal
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; source "${SUBMIT_DIR}/../env.sh"; cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
: "${PREFLIGHT:?Set PREFLIGHT to an exact-LoRA positive-control artifact produced by the current config schema}"
ROOT=out/multimodel/eval/qwen25_exact_lora_up713_intersection/job_${SLURM_JOB_ID}
OUTPUT=${ROOT}/generations.json
for path in "${CONFIG}" "${PROMPTS}" "${PREFLIGHT}"; do test -s "${path}"; done
if [[ -e "${ROOT}" ]]; then echo "Refusing to reuse ${ROOT}" >&2; exit 21; fi
printf 'hostname=%s\n' "$(hostname)"; date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\noutput=%s\n' "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
.venv/bin/python -m aspd.multimodel.cli.run_exact_lora_localization \
  --config "${CONFIG}" --prompts "${PROMPTS}" --preflight "${PREFLIGHT}" \
  --output "${OUTPUT}" --mode intersection --device cuda --max-new-tokens 1024 --batch-size 4
.venv/bin/python - "${OUTPUT}" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text())
expected=["clean_base","clean_target","up_proj_layers_7_13_removed","up_proj_outside_layers_7_13_removed","mlp_layers_7_13_removed","up_proj_layers_7_13_inserted_into_base"]
assert x["scientific_status"] == "EXPLORATORY_EXACT_LORA_UP713_INTERSECTION_GENERATIONS"
assert x["arms_expected"] == expected and [arm["arm"] for arm in x["arms"]] == expected
assert x["base_clean_replay_logits"]["exact"] and x["target_clean_replay_logits"]["exact"]
assert x["generation"]["max_new_tokens"] == 1024
counts={arm["arm"]:arm["site_count"] for arm in x["arms"]}
assert counts["up_proj_layers_7_13_removed"] == 7
assert counts["up_proj_outside_layers_7_13_removed"] == 21
assert counts["mlp_layers_7_13_removed"] == 21
assert counts["up_proj_layers_7_13_inserted_into_base"] == 7
for arm in x["arms"]:
    assert len(arm["sequences"]) == 30 and all(row["eos_reached"] for row in arm["sequences"])
    if arm["replacement_audit"] is not None:
        assert arm["replacement_audit"]["replacement_exact"] and arm["replacement_audit"]["restoration_exact"]
def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)
assert finite(x)
print("EXACT_LORA_UP713_INTERSECTION_GENERATION_PASS")
PY
printf 'EXACT_LORA_UP713_INTERSECTION_GENERATION_COMPLETE\n'

#!/usr/bin/env bash
# Recapture layer-7 R and layers-7--13 up_proj inputs on immutable frozen-V3 tokens.
# No response generation, judging, training, feature selection, or intervention.
#SBATCH --job-name=mm-q25-up713-v3-cache
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=02:00:00
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
ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3
RESPONSES=${ROOT}/responses_1024
ACTIVATIONS=${ROOT}/activation_cache
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
SCORES=${ROOT}/phase4_judge_46712130/behavioral_scores.json
OUTPUT=${ROOT}/aspd_l07_l13_up_matrix_cache
WORK=${OUTPUT}.partial_${SLURM_JOB_ID}

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
for path in "${CONFIG}" "${RESPONSES}/manifest.json" "${ACTIVATIONS}/manifest.json" \
  "${PROMPTS}" "${SCORES}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" || -e "${WORK}" ]]; then
  echo "Refusing stale multilayer V3 cache output" >&2
  exit 21
fi

.venv/bin/python -m aspd.multimodel.cli.prepare_aspd_v3_matrix_cache \
  --config "${CONFIG}" --response-dir "${RESPONSES}" \
  --activation-cache "${ACTIVATIONS}" --prompts "${PROMPTS}" --scores "${SCORES}" \
  --output-dir "${WORK}" --device cuda

WORK="${WORK}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
root=Path(os.environ["WORK"])
x=json.loads((root/"manifest.json").read_text())
names=[f"L{layer:02d}_up" for layer in range(7,14)]
assert x["records"] == 360 and x["matrix_names"] == names
assert x["fixed_batching"]["evaluation_batch_size_sequences"] == 2
assert x["grounding_policy"] == {
    "base": "use recaptured R from the effective pinned ASPD Base",
    "target": "use recaptured R from the effective BF16 merged ASPD Target",
}
for model in x["models"]:
    replay=model["grounding_replay"]
    assert replay["configured_hook"]["module"] == "model.layers.7"
    assert replay["same_grounding_site"] is False
    assert replay["evaluation_source"] == "recaptured_aspd_effective_model"
    assert math.isfinite(replay["frozen_vs_effective_relative_rms"])
for spec in ("base","target"):
    for split in ("train","validation"):
        for entry in x["splits"][split]["shards"]:
            path=root/"models"/spec/entry["path"]
            assert path.is_file() and path.stat().st_size > 0
assert sum(row["sequences"] for split in x["splits"].values() for row in split["shards"]) == 360
print("ASPD_L07_L13_UP_V3_CACHE_INTEGRITY_PASS")
PY
mv "${WORK}" "${OUTPUT}"
printf 'ASPD_L07_L13_UP_V3_CACHE_COMPLETE\n'

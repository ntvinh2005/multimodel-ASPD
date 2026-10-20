#!/usr/bin/env bash
# Replay frozen V3 tokens and capture ASPD X_j plus merged-Target R.
# Existing exact tokens and Base R are reused; the source cache is never overwritten.
#SBATCH --job-name=mm-q25-aspd-v3-x
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
ROOT=out/multimodel/eval/qwen25_extreme_sports_behavioral_v3
RESPONSES=${ROOT}/responses_1024
ACTIVATIONS=${ROOT}/activation_cache
PROMPTS=docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl
SCORES=${ROOT}/phase4_judge_46712130/behavioral_scores.json
OUTPUT=${ROOT}/aspd_matrix_cache_general1m
WORK=${OUTPUT}.partial_${SLURM_JOB_ID}

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
for path in "${RESPONSES}/manifest.json" "${ACTIVATIONS}/manifest.json" "${PROMPTS}" "${SCORES}"; do
  test -s "${path}"
done
if [[ -e "${OUTPUT}" || -e "${WORK}" ]]; then
  echo "Refusing stale V3 ASPD matrix cache output" >&2
  exit 21
fi
.venv/bin/python -m aspd.multimodel.cli.prepare_aspd_v3_matrix_cache \
  --config "${CONFIG}" --response-dir "${RESPONSES}" \
  --activation-cache "${ACTIVATIONS}" --prompts "${PROMPTS}" --scores "${SCORES}" \
  --output-dir "${WORK}" --device cuda

WORK="${WORK}" .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ["WORK"])
manifest=json.loads((root/"manifest.json").read_text())
assert manifest["records"] == 360
assert manifest["matrix_names"] == ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
assert manifest["fixed_batching"]["evaluation_batch_size_sequences"] == 2
assert manifest["grounding_policy"]["base"].startswith("reuse frozen V3 R")
assert "effective BF16 merged ASPD Target" in manifest["grounding_policy"]["target"]
assert manifest["models"][0]["grounding_replay"]["frozen_vs_effective_relative_rms"] <= 1e-3
assert manifest["models"][1]["grounding_replay"]["evaluation_source"] == "recaptured_aspd_effective_merged_target"
for model in ("base","target"):
    for split in ("train","validation"):
        for entry in manifest["splits"][split]["shards"]:
            path=root/"models"/model/entry["path"]
            assert path.is_file() and path.stat().st_size > 0
assert sum(row["sequences"] for split in manifest["splits"].values() for row in split["shards"]) == 360
print("ASPD_V3_MATRIX_CACHE_INTEGRITY_PASS")
PY
mv "${WORK}" "${OUTPUT}"
printf 'ASPD_V3_MATRIX_CACHE_COMPLETE\n'

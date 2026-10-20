#!/usr/bin/env bash
# P1--P5 only for the trained L07--L13 up_proj dictionary; no P6 is launched.
#SBATCH --job-name=mm-q25-up713-p1p5
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"; cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l07_l13_up_1m.yaml
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m
CHECKPOINT=${RUN}/checkpoint_00004096.pt
ANALYSIS=${RUN}/analysis
FIDELITY=out/multimodel/diagnostics/qwen25_aspd_l07_l13_up_1m/step4096_full_validation.json

printf 'hostname=%s\n' "$(hostname)"; date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ncheckpoint=%s\nanalysis=%s\n' "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CHECKPOINT}" "${ANALYSIS}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
test -s "${CHECKPOINT}"; test -s "${FIDELITY}"
if [[ -e "${ANALYSIS}" ]]; then echo "Refusing to overwrite ${ANALYSIS}" >&2; exit 21; fi
.venv/bin/python -m aspd.multimodel.cli.analyze "${CONFIG}" "${CHECKPOINT}" --device cuda --output-dir "${ANALYSIS}"
.venv/bin/python -m aspd.multimodel.cli.report_analysis "${CONFIG}" "${ANALYSIS}" --output-dir "${ANALYSIS}/report"
ANALYSIS="${ANALYSIS}" .venv/bin/python - <<'PY'
import json,os
from pathlib import Path
from safetensors.torch import load_file
root=Path(os.environ["ANALYSIS"]); manifest=json.load(open(root/"analysis_manifest.json"))
assert manifest["checkpoint"]["step"] == 4096 and manifest["n_features"] == 8192 and manifest["top_k"] == 32
tensors=load_file(str(root/"posthoc.safetensors")); expected=[f"L{layer:02d}_up" for layer in range(7,14)]
assert tensors["fire_count_C"].shape == (8192,) and int(tensors["fire_count_C"].sum()) == 131072*32
for model in ("base","target"):
    assert all(f"locus/{model}/{matrix}" in tensors for matrix in expected)
assert all(tensor.isfinite().all() for tensor in tensors.values())
taxonomy=json.load(open(root/"taxonomy.json")); assert sum(map(len,taxonomy.values())) == 8192
summary=json.load(open(root/"report/summary.json")); assert summary["validation_tokens"] == 131072
assert summary["invariants"]["observed_total_fires"] == 131072*32
print("ASPD_L07_L13_UP_P1P5_INTEGRITY_PASS")
PY
printf 'ASPD_L07_L13_UP_P1P5_COMPLETE\n'

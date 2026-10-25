#!/usr/bin/env bash
# Purpose: Evaluate the verified final 24,414-step checkpoint on the full frozen held-out corpus and locked behavioral prompts, then export the scientific and P6-ready result package.
# Resource rationale: One B200 holds each frozen 7B source model sequentially plus the trained ASPD evaluator; 192 GB RAM and 24 CPUs cover checkpoint loading, global sufficient statistics, CSV export, and plots.
# Expected runtime: 45-90 minutes for roughly 2M held-out positions plus 180 locked prompts and artifact export; the 2-hour limit provides model/filesystem margin.
#SBATCH --job-name=q25-aspd-orig-eval
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${ALLOW_EVALUATION:?Set ALLOW_EVALUATION=1 after reviewing the completed main-run verification}"
[[ "${ALLOW_EVALUATION}" == "1" ]]

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
RUN="out/multimodel/runs/qwen25_aspd_original_100m"
STEP="00024414"
OUTPUT="${RUN}/evaluation"
VERIFY="${RUN}/checkpoint_verification_report.json"
mkdir -p slurm/logs
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

.venv/bin/python -c "import json,pathlib; d=json.loads(pathlib.Path('${VERIFY}').read_text()); assert d['status']=='100M_TRAINING_COMPLETE' and d['final_step']==24414, d"
.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --require-dataset-ready --hf-hub-cache "${HF_HOME}/hub"
.venv/bin/python -m aspd.multimodel.cli.evaluate_original_recipe \
  "${CONFIG}" "${RUN}/checkpoint_${STEP}.rank0.pt" "${OUTPUT}" \
  --checkpoint-rank1 "${RUN}/checkpoint_${STEP}.rank1.pt" --device cuda
.venv/bin/python -m aspd.multimodel.cli.export_original_results \
  "${CONFIG}" "${RUN}" "${OUTPUT}"
printf 'EVALUATION_OUTPUT=%s\nPACKAGE_MANIFEST=%s\n' \
  "${OUTPUT}" "${RUN}/result_package_manifest.json"

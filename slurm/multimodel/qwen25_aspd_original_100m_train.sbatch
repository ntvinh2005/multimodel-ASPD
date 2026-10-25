#!/usr/bin/env bash
# Purpose: Execute the single authorized 24,414-step, 99,999,744-exposure scientific ASPD training run only after the real smoke artifact records GO.
# Resource rationale: Three B200s are required by the frozen topology: rank 0 streams paired Base/Target L14 activations and ranks 1-2 train the full 36,864-component model with DDP; 256 GB RAM and 32 CPUs support loading, streaming, and checkpoint serialization.
# Expected runtime: The smoke verifier supplies the measured estimate; the 10-hour limit covers the provisional 6.2-hour training estimate plus startup, checkpoint, and filesystem overhead.
#SBATCH --job-name=q25-aspd-orig-100m
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
#SBATCH --time=10:00:00
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
: "${ALLOW_100M:?Set ALLOW_100M=1 only after reviewing the smoke GO evidence}"
: "${SMOKE_JOB_ID:?Set SMOKE_JOB_ID to the reviewed smoke Slurm job ID}"
[[ "${ALLOW_100M}" == "1" ]]

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
SMOKE_REPORT="out/multimodel/smoke/qwen25_aspd_original_100m/job_${SMOKE_JOB_ID}/smoke_verification.json"
OUTPUT="out/multimodel/runs/qwen25_aspd_original_100m"
AUDIT="out/multimodel/audits/qwen25_aspd_original_100m_main_${SLURM_JOB_ID}.json"
mkdir -p slurm/logs "$(dirname "${OUTPUT}")" "$(dirname "${AUDIT}")"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

.venv/bin/python -c "import json,pathlib; p=pathlib.Path('${SMOKE_REPORT}'); d=json.loads(p.read_text()); assert d['status']=='GO' and d['final_step']==100, d"
.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --require-dataset-ready --hf-hub-cache "${HF_HOME}/hub" --output "${AUDIT}"
RESUME_ARGS=()
if [[ -f "${OUTPUT}/latest_checkpoint.pt" ]]; then
  RESUME_ARGS+=(--resume auto)
fi
torchrun --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_original_recipe \
  "${CONFIG}" --output-dir "${OUTPUT}" "${RESUME_ARGS[@]}"
.venv/bin/python -m aspd.multimodel.cli.verify_original_run \
  "${CONFIG}" "${OUTPUT}" --mode main \
  --output "${OUTPUT}/checkpoint_verification_report.json"
printf 'MAIN_OUTPUT=%s\nMAIN_VERIFICATION=%s\n' \
  "${OUTPUT}" "${OUTPUT}/checkpoint_verification_report.json"

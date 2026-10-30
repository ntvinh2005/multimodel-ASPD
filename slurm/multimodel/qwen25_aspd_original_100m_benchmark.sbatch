#!/usr/bin/env bash
# Purpose: Run the one authorized 100-step real-dimension technical smoke, checkpoint at step 50, resume to step 100, load it in the evaluator, and issue the throughput GO/NO-GO record.
# Resource rationale: Three B200s reproduce the production topology exactly: rank 0 holds both frozen 7B source models while ranks 1-2 train the 36,864-component ASPD with DDP; 256 GB RAM and 32 CPUs cover model loading and token streaming.
# Expected runtime: 35-75 minutes including two model startups, paired checkpoint writes, a bounded held-out evaluator load test, and verification; the 2-hour limit provides queue/filesystem margin.
#SBATCH --job-name=q25-aspd-orig-smoke
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
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
: "${ALLOW_SMOKE:?Set ALLOW_SMOKE=1 after the frozen dataset validation passes}"
[[ "${ALLOW_SMOKE}" == "1" ]]

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
OUTPUT="out/multimodel/smoke/qwen25_aspd_original_100m/job_${SLURM_JOB_ID}"
AUDIT="out/multimodel/audits/qwen25_aspd_original_100m_smoke_${SLURM_JOB_ID}.json"
mkdir -p slurm/logs "$(dirname "${OUTPUT}")" "$(dirname "${AUDIT}")"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --require-dataset-ready --hf-hub-cache "${HF_HOME}/hub" --output "${AUDIT}"
.venv/bin/torchrun --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_original_recipe \
  "${CONFIG}" --output-dir "${OUTPUT}" --stop-after-step 50
.venv/bin/torchrun --standalone --nproc-per-node=3 \
  -m aspd.multimodel.cli.train_original_recipe \
  "${CONFIG}" --output-dir "${OUTPUT}" --resume auto --stop-after-step 100
.venv/bin/python -m aspd.multimodel.cli.evaluate_original_recipe \
  "${CONFIG}" "${OUTPUT}/checkpoint_00000100.rank0.pt" "${OUTPUT}/evaluation_load_test" \
  --checkpoint-rank1 "${OUTPUT}/checkpoint_00000100.rank1.pt" --device cuda \
  --max-validation-sequences 8
.venv/bin/python -m aspd.multimodel.cli.verify_original_run \
  "${CONFIG}" "${OUTPUT}" --mode smoke \
  --evaluation "${OUTPUT}/evaluation_load_test/held_out_evaluation.json" \
  --output "${OUTPUT}/smoke_verification.json"
printf 'SMOKE_OUTPUT=%s\nSMOKE_VERIFICATION=%s\n' \
  "${OUTPUT}" "${OUTPUT}/smoke_verification.json"

#!/usr/bin/env bash
# Purpose: Run a bounded one-batch gradient and grounding audit against the immutable final ASPD checkpoint without training or updating any run artifact.
# Resource rationale: One B200 holds both frozen 7B source models and the 36,864-component decomposition; 192 GB host RAM supports memory-mapped checkpoint loading.
# Expected runtime: 10-25 minutes for one held-out sequence and five isolated backward probes; the 40-minute limit includes model-loading and filesystem margin.
#SBATCH --job-name=q25-aspd-opt-audit
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192gb
#SBATCH --time=00:40:00
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
: "${ALLOW_DIAGNOSTIC:?Set ALLOW_DIAGNOSTIC=1 after reviewing this bounded read-only job}"
[[ "${ALLOW_DIAGNOSTIC}" == "1" ]]

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
RUN="out/multimodel/runs/qwen25_aspd_original_100m"
CHECKPOINT="${RUN}/checkpoint_00024414.rank0.pt"
OUTPUT="out/multimodel/diagnostics/qwen25_aspd_original_100m_optimization_audit/gpu_job_${SLURM_JOB_ID}.json"
mkdir -p slurm/logs
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PYTHON=3.13

.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  "${CONFIG}" --require-dataset-ready --hf-hub-cache "${HF_HOME}/hub"
.venv/bin/python -m aspd.multimodel.cli.audit_original_optimization_gpu \
  "${CONFIG}" "${CHECKPOINT}" "${OUTPUT}" \
  --device cuda --validation-start 0 --batch-size-sequences 1
printf 'OPTIMIZATION_AUDIT=%s\n' "${OUTPUT}"

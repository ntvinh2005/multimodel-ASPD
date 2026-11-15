#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-coherence-x
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=176G
#SBATCH --output=slurm/logs/aspd-v3-coherence-extract-%j.out
#SBATCH --error=slurm/logs/aspd-v3-coherence-extract-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/feature-coherence-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/main500m_preparation_v1.yaml
PROTOCOL=configs/multimodel/aspd_v3/main500m_feature_coherence_v1.yaml
PREPARATION=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/preparation_contract_05d0530f3109
CHECKPOINT=/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43/checkpoints/step-000061036
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/extraction
OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"

cd "${WORKTREE}"
module load conda
source /blue/thai/vinhnguyen1/multimodel-ASPD/env.sh
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${CHECKPOINT}/COMPLETE"
test -s "${PREPARATION}/PREPARED"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_ROOT}" slurm/logs

echo "git_sha=$(git rev-parse HEAD) checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
echo "execution=three_model_owner_ranks requested_gpus=3 global_user_gpu_cap=5"
date --iso-8601=seconds
nvidia-smi

for MODE in discovery-new confirmatory; do
  "${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=3 \
    -m aspd.multimodel.cli.extract_aspd_v3_feature_coherence \
    --mode "${MODE}" \
    --config "${CONFIG}" \
    --protocol "${PROTOCOL}" \
    --preparation-dir "${PREPARATION}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/${MODE}" \
    --local-files-only
  test -s "${OUTPUT_DIR}/${MODE}/COMPLETE"
done

echo "ASPD_V3_FEATURE_COHERENCE_EXTRACTION_COMPLETE output=${OUTPUT_DIR}"

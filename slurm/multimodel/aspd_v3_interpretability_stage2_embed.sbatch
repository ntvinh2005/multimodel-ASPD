#!/usr/bin/env bash
# Exploratory semantic embeddings only; this is not an Intruder judge.
#SBATCH --job-name=aspd-v3-stage2-embed
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/aspd-v3-stage2-embed-%j.out
#SBATCH --error=slurm/logs/aspd-v3-stage2-embed-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/interpretability-failure-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
PROTOCOL=configs/multimodel/aspd_v3/main500m_interpretability_stage2_v1.yaml
PREPARATION=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/preparation_git_fccfdc9
INPUT="${PREPARATION}/clustering_embedding_input.jsonl"
MODEL=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/embeddings
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
test -s "${PREPARATION}/PREPARED"
test -s "${INPUT}"
test -d "${MODEL}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

echo "git_sha=$(git rev-parse HEAD) input=${INPUT} model=${MODEL} output=${OUTPUT_DIR}"
echo "execution=exploratory_embeddings requested_gpus=1 global_user_gpu_cap=5"
nvidia-smi
"${PYTHON}" -m aspd.multimodel.cli.embed_aspd_v3_interpretability_contexts \
  --protocol "${PROTOCOL}" --input "${INPUT}" --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" --batch-size 64
test -s "${OUTPUT_DIR}/COMPLETE"
echo "ASPD_V3_STAGE2_EMBEDDINGS_COMPLETE output=${OUTPUT_DIR}"

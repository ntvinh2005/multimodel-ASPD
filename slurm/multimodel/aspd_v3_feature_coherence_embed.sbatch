#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-coherence-emb
#SBATCH --account=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/aspd-v3-coherence-embed-%j.out
#SBATCH --error=slurm/logs/aspd-v3-coherence-embed-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/feature-coherence-v1
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
PROTOCOL=configs/multimodel/aspd_v3/main500m_feature_coherence_embedding_v1.yaml
CONTEXTS=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/contexts_git_4d0321c
MODEL=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3
OUTPUT_ROOT=/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/embeddings
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
test -s "${CONTEXTS}/PREPARED"
test -d "${MODEL}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_ROOT}" slurm/logs

echo "git_sha=$(git rev-parse HEAD) contexts=${CONTEXTS} output=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi

"${PYTHON}" -m aspd.multimodel.cli.embed_aspd_v3_interpretability_contexts \
  --protocol "${PROTOCOL}" \
  --input "${CONTEXTS}/discovery_new_embedding_input.jsonl" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}/discovery_new" \
  --batch-size 64

"${PYTHON}" -m aspd.multimodel.cli.embed_aspd_v3_interpretability_contexts \
  --protocol "${PROTOCOL}" \
  --input "${CONTEXTS}/confirmatory_embedding_input.jsonl" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}/confirmatory" \
  --batch-size 64

test -s "${OUTPUT_DIR}/discovery_new/COMPLETE"
test -s "${OUTPUT_DIR}/confirmatory/COMPLETE"
echo "ASPD_V3_FEATURE_COHERENCE_EMBEDDINGS_COMPLETE output=${OUTPUT_DIR}"

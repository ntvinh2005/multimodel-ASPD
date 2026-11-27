#!/usr/bin/env bash
#SBATCH --job-name=aspd-sot2
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/blue/thai/share/multimodel-ASPD/logs/%x-%j.out
#SBATCH --error=/blue/thai/share/multimodel-ASPD/logs/%x-%j.err

set -euo pipefail
cd "${REPO:?absolute Semantic OT worktree}"
test "$(git rev-parse HEAD)" = "${REVISION:?expected full Git revision}"
test -z "$(git status --porcelain --untracked-files=no)"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python -u \
  -m aspd.multimodel.cli.collect_aspd_semantic_contributions \
  --config configs/multimodel/posthoc/v31_500m_validation_v1.yaml \
  --reference "${REFERENCE:?sealed Natural Validation post-hoc reference}" \
  --output "${OUTPUT:?immutable experiment namespace}" --pools "${POOLS:?predefined wave}"

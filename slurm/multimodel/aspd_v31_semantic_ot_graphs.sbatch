#!/usr/bin/env bash
#SBATCH --job-name=aspd-sot2-graphs
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
cd "${REPO:?absolute replay worktree}"
test "$(git rev-parse HEAD)" = "${REVISION:?pinned replay commit}"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "${OUTPUT:?immutable graph artifact}/COMPLETE.json"
case "${SCOPE:-pilot}" in
  pilot) args=(--anchors 3) ;;
  all) args=(--all-examples) ;;
  *) echo "Invalid graph scope" >&2; exit 2 ;;
esac
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
exec /blue/thai/share/multimodel-ASPD/repo/.venv/bin/python -u \
  -m aspd.multimodel.cli.measure_semantic_ot_graphs \
  --config configs/multimodel/posthoc/v31_500m_validation_v1.yaml \
  --bundle "${BUNDLE:?verified Semantic OT v2 bundle}" \
  --measurements "${MEASUREMENTS:?original sealed pool archive}" --output "${OUTPUT}" "${args[@]}"

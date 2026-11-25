#!/usr/bin/env bash
#SBATCH --job-name=aspd-token-deps-bench
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/blue/thai/share/multimodel-ASPD/logs/%x-%j.out
#SBATCH --error=/blue/thai/share/multimodel-ASPD/logs/%x-%j.err

set -euo pipefail

cd "${REPO:?benchmark worktree}"
test "$(git rev-parse HEAD)" = "${REVISION:?expected full Git revision}"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "${OUTPUT:?new immutable benchmark output}"

export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /blue/thai/share/multimodel-ASPD/repo/.venv/bin/python -u \
  -m aspd.multimodel.cli.benchmark_token_dependencies \
  --config configs/multimodel/aspd_v3/negation_mechanism_eval_v1.yaml \
  --preparation "${PREPARATION:?validated Negation preparation}" \
  --discovery "${DISCOVERY:?validated Negation discovery}" \
  --selection-preparation "${SELECTION_PREPARATION:?validated Selection Frame preparation}" \
  --output "${OUTPUT}"

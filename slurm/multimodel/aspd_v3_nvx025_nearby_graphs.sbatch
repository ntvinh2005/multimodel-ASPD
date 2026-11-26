#!/usr/bin/env bash
#SBATCH --job-name=aspd-nvx025-near
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:09:00
#SBATCH --output=/blue/thai/share/multimodel-ASPD/logs/%x-%j.out
#SBATCH --error=/blue/thai/share/multimodel-ASPD/logs/%x-%j.err
set -euo pipefail
cd "${REPO:?pinned worktree}"
test "$(git rev-parse HEAD)" = "${REVISION:?pinned full Git revision}"
test -z "$(git status --porcelain --untracked-files=no)"
export HF_HOME=/blue/thai/share/multimodel-ASPD/cache/huggingface
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /blue/thai/share/multimodel-ASPD/repo/.venv/bin/python -u \
  -m aspd.multimodel.cli.measure_natural_token_graphs \
  --config configs/multimodel/aspd_v3/negation_mechanism_eval_v1.yaml \
  --archive "${ARCHIVE:?frozen test activation archive}" \
  --pilot "${PILOT:?verified natural token pilot}/pilot.json" \
  --replay-inputs "$PILOT/replay-inputs.json" \
  --output "${OUTPUT:?new immutable graph run}" \
  --near-window 36c8718babc8f3c5dcb8e5cfe9cc229a9f5726909ab07e354ff116c9b9a4ce57 \
  --radius 1 --max-gpu-hours 0.15 --max-output-bytes 64000000 --execute

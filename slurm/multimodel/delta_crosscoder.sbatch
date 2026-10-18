#!/usr/bin/env bash
# One reusable launcher for matched smoke, Phase-2A cache/train smoke, and gated full training.
#SBATCH --job-name=mm-delta-crosscoder
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"

: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
MODE="${MODE:-smoke}"
SMOKE_CONFIG="configs/multimodel/delta_crosscoder/qwen3_matched_smoke.yaml"
PAPER_CONFIG="configs/multimodel/delta_crosscoder/paper_default.yaml"
CONTRASTIVE_CONFIG="configs/multimodel/delta_crosscoder/qwen3_contrastive_smoke.yaml"

case "${MODE}" in
  smoke)
    CONFIG="${CONFIG:-${SMOKE_CONFIG}}"
    ;;
  contrastive_cache|contrastive_smoke)
    CONFIG="${CONFIG:-${CONTRASTIVE_CONFIG}}"
    ;;
  train)
    : "${ALLOW_FULL_TRAIN:?Set ALLOW_FULL_TRAIN=1 only after explicit Phase-2 approval}"
    [[ "${ALLOW_FULL_TRAIN}" == "1" ]]
    CONFIG="${CONFIG:-${PAPER_CONFIG}}"
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected smoke, contrastive_cache, contrastive_smoke, or train" >&2
    exit 2
    ;;
esac

mkdir -p slurm/logs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${MODE}" == contrastive_* ]]; then
  # Both pinned snapshots are already on Blue storage; avoid accidental network drift.
  export HF_HUB_OFFLINE=1
fi
START_SECONDS="${SECONDS}"

RUN_DIR=$(.venv/bin/python -c '
import sys
from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
print(load_delta_crosscoder_config(sys.argv[1]).delta_crosscoder.output_dir)
' "${CONFIG}")
CACHE_DIR=$(.venv/bin/python -c '
import sys
from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
print(load_delta_crosscoder_config(sys.argv[1]).cache.root)
' "${CONFIG}")

printf 'git_commit=%s\nmode=%s\nconfig=%s\nrun_dir=%s\ncache=%s\n' \
  "$(git rev-parse HEAD)" "${MODE}" "${CONFIG}" "${RUN_DIR}" "${CACHE_DIR}"
nvidia-smi
.venv/bin/python -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
if [[ "${MODE}" == "contrastive_cache" ]]; then
  CONTRASTIVE_OUTPUT=$(.venv/bin/python -c '
import sys
from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
cfg = load_delta_crosscoder_config(sys.argv[1])
assert cfg.delta_contrastive is not None
print(cfg.delta_contrastive.cache.output_dir)
' "${CONFIG}")
  .venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_cache \
    --config "${CONFIG}" \
    --max-prompts "${MAX_PROMPTS:-16}" \
    --output "${CONTRASTIVE_OUTPUT}" \
    --device cuda
elif [[ "${MODE}" == "contrastive_smoke" ]]; then
  .venv/bin/python -c '
import sys
from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_contrastive import validate_contrastive_cache
cfg = load_delta_crosscoder_config(sys.argv[1])
print(validate_contrastive_cache(cfg.delta_crosscoder.data.contrastive_manifest, cfg))
' "${CONFIG}"
  .venv/bin/python -m aspd.multimodel.cli.train_delta_crosscoder "${CONFIG}" --device cuda
else
  .venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
  .venv/bin/python -m aspd.multimodel.cli.train_delta_crosscoder "${CONFIG}" --device cuda
fi

printf 'mode=%s wall_clock_seconds=%s finished_at=%s\n' \
  "${MODE}" "$((SECONDS - START_SECONDS))" "$(date --iso-8601=seconds)"

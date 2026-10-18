#!/usr/bin/env bash
# One reusable launcher. Phase 1 permits MODE=smoke only; train requires an explicit later approval.
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

case "${MODE}" in
  smoke)
    CONFIG="${CONFIG:-${SMOKE_CONFIG}}"
    ;;
  train)
    : "${ALLOW_FULL_TRAIN:?Set ALLOW_FULL_TRAIN=1 only after explicit Phase-2 approval}"
    [[ "${ALLOW_FULL_TRAIN}" == "1" ]]
    CONFIG="${CONFIG:-${PAPER_CONFIG}}"
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected smoke or train" >&2
    exit 2
    ;;
esac

mkdir -p slurm/logs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
.venv/bin/python -m aspd.multimodel.cli.train_delta_crosscoder "${CONFIG}" --device cuda

printf 'mode=%s wall_clock_seconds=%s finished_at=%s\n' \
  "${MODE}" "$((SECONDS - START_SECONDS))" "$(date --iso-8601=seconds)"

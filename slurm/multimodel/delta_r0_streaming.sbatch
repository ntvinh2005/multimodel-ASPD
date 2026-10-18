#!/usr/bin/env bash
# Bounded Phase-2C launch surface. This file intentionally has no full-50k mode.
#SBATCH --job-name=mm-delta-r0-stream
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=01:00:00
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
MODE="${MODE:-streaming-pilot}"
PILOT_CONFIG="configs/multimodel/delta_crosscoder/r0_extreme_sports_streaming_pilot.yaml"
LOCAL_CONFIG="configs/multimodel/delta_crosscoder/r0_extreme_sports_local.yaml"
MIX_CONFIG="configs/multimodel/delta_crosscoder/r0_data_mix_pilot.yaml"
mkdir -p slurm/logs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${MODE}" in
  contrastive-text-benchmark)
    OVERWRITE_ARGS=()
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
      OVERWRITE_ARGS+=(--overwrite)
    fi
    .venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_text \
      --config "${LOCAL_CONFIG}" \
      --max-prompts 32 \
      --batch-size "${GENERATION_BATCH_SIZE:-4}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
      --output out/multimodel/data/delta_r0/contrastive_text_benchmark \
      --device cuda \
      "${OVERWRITE_ARGS[@]}"
    ;;
  streaming-pilot)
    # Hard safety guard: Phase 2C forbids training beyond 500 steps.
    STEPS=$(.venv/bin/python -c '
import sys
from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
print(load_delta_crosscoder_config(sys.argv[1]).delta_crosscoder.total_steps)
' "${PILOT_CONFIG}")
    if (( STEPS > 500 )); then
      echo "Refusing Phase-2C streaming pilot with ${STEPS} steps" >&2
      exit 2
    fi
    .venv/bin/python -m aspd.multimodel.cli.prepare_delta_r0_data \
      "${MIX_CONFIG}" --validate-only
    .venv/bin/python -m aspd.multimodel.cli.train_delta_crosscoder \
      "${PILOT_CONFIG}" --device cuda
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected contrastive-text-benchmark or streaming-pilot" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
#SBATCH --job-name=mm-crosscoder
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  # Project storage/cache locations and the Python 3.13 pin live here.
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"

: "${CONFIG:?Set CONFIG to a Crosscoder YAML config}"
: "${STAGE:?Set STAGE to cache, train, or analyze}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"

mkdir -p slurm/logs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR=$(.venv/bin/python -c 'import sys; from aspd.multimodel.baselines.crosscoder import load_crosscoder_config; print(load_crosscoder_config(sys.argv[1]).crosscoder.output_dir)' "${CONFIG}")
CACHE_DIR=$(.venv/bin/python -c 'import sys; from aspd.multimodel.baselines.crosscoder import load_crosscoder_config; print(load_crosscoder_config(sys.argv[1]).cache.root)' "${CONFIG}")
printf 'STAGE=%s\nCONFIG=%s\nRUN_DIR=%s\nCACHE_DIR=%s\nHF_HOME=%s\n' \
  "${STAGE}" "${CONFIG}" "${RUN_DIR}" "${CACHE_DIR}" "${HF_HOME}"
nvidia-smi

case "${STAGE}" in
  cache)
    if [[ "${REUSE_CACHE:-0}" == "1" ]]; then
      .venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
    else
      .venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}"
    fi
    ;;
  train)
    .venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
    .venv/bin/python -m aspd.multimodel.cli.train_crosscoder "${CONFIG}"
    ;;
  analyze)
    # A published-reference import reads the authors' CSV/SQLite artifacts directly.
    # Local checkpoint analysis still requires the paired activation cache.
    if [[ -z "${PUBLISHED_FEATURE_CSV:-}" ]]; then
      .venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
    fi
    ARGS=("${CONFIG}")
    [[ -n "${CHECKPOINT:-}" ]] && ARGS+=(--checkpoint "${CHECKPOINT}")
    [[ -n "${OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "${OUTPUT_DIR}")
    [[ -n "${PUBLISHED_FEATURE_CSV:-}" ]] && ARGS+=(--published-feature-csv "${PUBLISHED_FEATURE_CSV}")
    [[ -n "${PUBLISHED_EXAMPLES_DB:-}" ]] && ARGS+=(--published-examples-db "${PUBLISHED_EXAMPLES_DB}")
    [[ -n "${PUBLISHED_TOKENIZER:-}" ]] && ARGS+=(--published-tokenizer "${PUBLISHED_TOKENIZER}")
    .venv/bin/python -m aspd.multimodel.cli.analyze_crosscoder "${ARGS[@]}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected cache, train, or analyze" >&2
    exit 2
    ;;
esac


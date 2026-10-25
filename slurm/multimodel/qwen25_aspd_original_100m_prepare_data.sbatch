#!/usr/bin/env bash
# Purpose: Materialize and fully validate the immutable FineWeb train/calibration/validation token corpus required by the L14 O-projection 100M ASPD experiment.
# Resource rationale: Four CPU cores, 64 GB RAM, and no GPU cover sequential tokenization, duplicate/contamination indices, shard checksums, and roughly 102M stored token positions.
# Expected runtime: 4-12 hours depending on Hugging Face streaming and project-storage throughput; the 18-hour limit provides filesystem/network margin.
#SBATCH --job-name=q25-aspd-data-100m
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project or Blue storage}"
: "${ALLOW_DATA_PREP:?Set ALLOW_DATA_PREP=1 after reviewing the pinned FineWeb source}"
[[ "${ALLOW_DATA_PREP}" == "1" ]]

CONFIG="configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
DATA_ROOT="out/multimodel/data/qwen25_aspd_original_100m"
REPORT="${DATA_ROOT}/preparation_report.json"
mkdir -p slurm/logs
export TOKENIZERS_PARALLELISM=false
export UV_PYTHON=3.13

.venv/bin/python -m aspd.multimodel.cli.prepare_original_recipe_data \
  "${CONFIG}" --report "${REPORT}"
.venv/bin/python -m aspd.multimodel.cli.prepare_original_recipe_data \
  "${CONFIG}" --validate-only --report "${DATA_ROOT}/validation_report.json"
sha256sum "${DATA_ROOT}/manifest.json" > "${DATA_ROOT}/manifest.sha256"
printf 'DATA_ROOT=%s\nASPD_ORIGINAL_100M_DATA_READY\n' "${DATA_ROOT}"

#!/usr/bin/env bash
# Blind deterministic response-level scoring for a frozen, cap-audited V2 set.
#SBATCH --job-name=mm-delta-behavior-v2-judge
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
: "${RESPONSE_DIR:?Set RESPONSE_DIR to the frozen response set that passed the cap audit}"

.venv/bin/python -m aspd.multimodel.cli.judge_delta_behavioral_v2 \
  --response-dir "${RESPONSE_DIR}" --device cuda --batch-size 8 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --revision 916b56a44061fd5cd7d6a8fb632557ed4f724f60

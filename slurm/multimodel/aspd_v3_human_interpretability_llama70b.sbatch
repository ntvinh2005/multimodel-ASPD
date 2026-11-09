#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-human-llama
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-human-llama-%j.out
#SBATCH --error=slurm/logs/aspd-v3-human-llama-%j.err

set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?submit from the human-interpretability worktree}"
module load conda
source ../../env.sh

PYTHON=../.venv/bin/python
PACKAGE=/orange/thai/vinhnguyen1/aspd_v3/human_interpretability_v1/package
PUBLIC="${PACKAGE}/public/blinded_items.json"
KEY="${PACKAGE}/protected/answer_key.jsonl"
CONTRACT=configs/multimodel/aspd_v3/human_interpretability_llama70b_raw_v1.yaml
OUTPUT="/orange/thai/vinhnguyen1/aspd_v3/human_interpretability_v1/llama70b/job_${SLURM_JOB_ID}"
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${PUBLIC}"
test -s "${KEY}"
test -s "${CONTRACT}"
test ! -e "${OUTPUT}"
AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 10485760
test "${AVAILABLE_INODES}" -ge 10000
mkdir -p "$(dirname "${OUTPUT}")" slurm/logs

echo "git_sha=$(git rev-parse HEAD) contract_sha256=$(sha256sum "${CONTRACT}" | awk '{print $1}')"
date --iso-8601=seconds
nvidia-smi
${PYTHON} -m aspd.multimodel.cli.judge_aspd_v3_human_llama \
  --public-dataset "${PUBLIC}" --answer-key "${KEY}" \
  --contract "${CONTRACT}" --output-dir "${OUTPUT}"
test -s "${OUTPUT}/judge_summary.json"
echo "ASPD_V3_HUMAN_LLAMA70B_COMPLETE output=${OUTPUT}"

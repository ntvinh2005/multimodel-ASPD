#!/usr/bin/env bash
#SBATCH --job-name=aspd-paper-judge-cache
#SBATCH --account=thai
#SBATCH --partition=hpg-default
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/aspd-paper-judge-cache-%j.out
#SBATCH --error=slurm/logs/aspd-paper-judge-cache-%j.err

set -euo pipefail
ulimit -c 0

WORKTREE=/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.codex_runtime/step13000-paper-intruder
HF_ROOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
VLLM_ENV=/blue/thai/vinhnguyen1/multimodel-ASPD/envs/vllm-paper-0.11.0
MODEL_ID=unsloth/Llama-3.3-70B-Instruct
REVISION=99cd0d2c829e92a67c844f9144c2509632e5c87f

cd "${WORKTREE}"
mkdir -p slurm/logs "$(dirname "${VLLM_ENV}")"
export HF_HOME="${HF_ROOT}"
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

if [ ! -x "${VLLM_ENV}/bin/vllm" ]; then
  module load python/3.12
  python -m venv "${VLLM_ENV}"
  "${VLLM_ENV}/bin/pip" install --upgrade pip
  "${VLLM_ENV}/bin/pip" install 'vllm==0.11.0'
fi
"${VLLM_ENV}/bin/pip" install 'transformers==4.57.1'

"${VLLM_ENV}/bin/hf" download "${MODEL_ID}" --revision "${REVISION}" --max-workers 1
SNAPSHOT="${HF_ROOT}/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/${REVISION}"
test -s "${SNAPSHOT}/model.safetensors.index.json"
"${VLLM_ENV}/bin/python" -c 'import vllm; print(vllm.__version__)'
"${VLLM_ENV}/bin/python" -c 'import transformers; assert transformers.__version__ == "4.57.1"; print(transformers.__version__)'
echo "ASPD_PAPER_JUDGE_CACHE_COMPLETE model=${MODEL_ID} revision=${REVISION} snapshot=${SNAPSHOT}"


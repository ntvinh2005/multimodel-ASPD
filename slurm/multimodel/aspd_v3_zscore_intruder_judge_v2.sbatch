#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-judge-v2
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-judge-v2-%j.out
#SBATCH --error=slurm/logs/aspd-v3-judge-v2-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the mechanistic-validation worktree}"
module load conda
source ../../env.sh

ANALYSIS_DIR="${ANALYSIS_DIR:?set ANALYSIS_DIR to a complete sparse-analysis artifact}"
JUDGE="${JUDGE:-mistral}"
MODE="${MODE:-controls}"
PYTHON=../.venv/bin/python

case "${JUDGE}" in
  mistral)
    CONTRACT=configs/multimodel/aspd_v3/zscore_intruder_judge_v2_mistral.yaml
    JUDGE_SNAPSHOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3
    ;;
  llama)
    CONTRACT=configs/multimodel/aspd_v3/zscore_intruder_judge_v2_llama.yaml
    JUDGE_SNAPSHOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
    ;;
  *)
    echo "unsupported JUDGE=${JUDGE}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  controls)
    EXTRA_ARGS=(--controls-only)
    ;;
  full)
    EXTRA_ARGS=()
    ;;
  *)
    echo "unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/orange/thai/vinhnguyen1/aspd_v3/zscore_mechanistic_validation_v1/intruder_judge_v2/${JUDGE}/${MODE}/job_${SLURM_JOB_ID}"
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${ANALYSIS_DIR}/experiment_manifest.json"
test -s "${ANALYSIS_DIR}/intruder_tasks.jsonl"
test -s "${ANALYSIS_DIR}/intruder_answer_key.jsonl"
test -s "${JUDGE_SNAPSHOT}/model.safetensors.index.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs
AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 52428800
test "${AVAILABLE_INODES}" -ge 100000
lfs quota -u "${USER}" /orange/thai || true
lfs quota -g thai /orange/thai || true

echo "git_sha=$(git rev-parse HEAD) analysis=${ANALYSIS_DIR} judge=${JUDGE} mode=${MODE} output=${OUTPUT_DIR} paid_api=false"
date --iso-8601=seconds
nvidia-smi
"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_intruder \
  --tasks "${ANALYSIS_DIR}/intruder_tasks.jsonl" \
  --answer-key "${ANALYSIS_DIR}/intruder_answer_key.jsonl" \
  --contract "${CONTRACT}" --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"

test -s "${OUTPUT_DIR}/experiment_manifest.json"
test -s "${OUTPUT_DIR}/judge_summary.json"
test -s "${OUTPUT_DIR}/judge_results.jsonl"
echo "ASPD_V3_INTRUDER_JUDGE_V2_COMPLETE output=${OUTPUT_DIR}"

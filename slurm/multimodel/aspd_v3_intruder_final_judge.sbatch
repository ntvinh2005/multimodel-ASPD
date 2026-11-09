#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-intruder-final
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=slurm/logs/aspd-v3-intruder-final-%j.out
#SBATCH --error=slurm/logs/aspd-v3-intruder-final-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the final-intruder worktree}"
module load conda
source ../../env.sh

MODE="${MODE:?set MODE to calibration, development, or final}"
INPUT_DIR="${INPUT_DIR:?set INPUT_DIR to the task artifact directory}"
CALIBRATION_SUMMARY="${CALIBRATION_SUMMARY:-}"
CONTRACT=configs/multimodel/aspd_v3/zscore_intruder_final_judge_v1_llama.yaml
PYTHON=../.venv/bin/python
SNAPSHOT=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
ROOT=/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1

case "${MODE}" in
  calibration)
    TASKS="${INPUT_DIR}/calibration_tasks.jsonl"
    ANSWERS="${INPUT_DIR}/calibration_answer_key.jsonl"
    EXTRA_ARGS=()
    ;;
  development|final)
    TASKS="${INPUT_DIR}/blinded_parent_tasks.jsonl"
    ANSWERS="${INPUT_DIR}/parent_answer_key.jsonl"
    test -s "${CALIBRATION_SUMMARY}"
    EXTRA_ARGS=(--calibration-summary "${CALIBRATION_SUMMARY}")
    ;;
  *) echo "unsupported MODE=${MODE}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${ROOT}/judge/${MODE}/job_${SLURM_JOB_ID}"
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${TASKS}"
test -s "${ANSWERS}"
test -s "${CONTRACT}"
test -s "${SNAPSHOT}/model.safetensors.index.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs
AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 10485760
test "${AVAILABLE_INODES}" -ge 50000
lfs quota -u "${USER}" /orange/thai || true
lfs quota -g thai /orange/thai || true

echo "mode=${MODE} git_sha=$(git rev-parse HEAD) judge=llama-3.3-70b paid_api=false output=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi
"${PYTHON}" -m aspd.multimodel.cli.judge_aspd_v3_intruder_final \
  --mode "${MODE}" --tasks "${TASKS}" --answer-key "${ANSWERS}" \
  --contract "${CONTRACT}" --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"

test -s "${OUTPUT_DIR}/experiment_manifest.json"
test -s "${OUTPUT_DIR}/judge_summary.json"
test -s "${OUTPUT_DIR}/rotation_results.jsonl"
echo "ASPD_V3_INTRUDER_FINAL_JUDGE_COMPLETE mode=${MODE} output=${OUTPUT_DIR}"

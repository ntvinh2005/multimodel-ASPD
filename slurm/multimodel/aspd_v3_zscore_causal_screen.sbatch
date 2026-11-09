#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-causal-c1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --array=0-3%4
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=slurm/logs/aspd-v3-causal-c1-%A_%a.out
#SBATCH --error=slurm/logs/aspd-v3-causal-c1-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the mechanistic-validation worktree}"
module load conda
source ../../env.sh

CAUSAL_PLAN="${CAUSAL_PLAN:?set CAUSAL_PLAN to the frozen causal plan artifact}"
PYTHON=../.venv/bin/python
CONTRACT=configs/multimodel/aspd_v3/zscore_causal_validation_v1.yaml
CONTROL_CONFIG=configs/multimodel/aspd_v3/two_model_gate_collapse_control_v1.yaml
ZSCORE_CONFIG=configs/multimodel/aspd_v3/two_model_gate_collapse_zscore_v1.yaml
CONTROL_RUN=/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/full_only_control/pilot/runs/job_50280055_0
ZSCORE_RUN=/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/per_anchor_zscore/pilot/runs/job_50280055_1
CONTROL_CHECKPOINT="${CONTROL_RUN}/checkpoints/step-000001221"
ZSCORE_CHECKPOINT="${ZSCORE_RUN}/checkpoints/step-000001221"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=full_only_control
    MODEL=gemma
    CONFIG="${CONTROL_CONFIG}"
    RUN_DIR="${CONTROL_RUN}"
    CHECKPOINT="${CONTROL_CHECKPOINT}"
    ;;
  1)
    ARM=full_only_control
    MODEL=smol
    CONFIG="${CONTROL_CONFIG}"
    RUN_DIR="${CONTROL_RUN}"
    CHECKPOINT="${CONTROL_CHECKPOINT}"
    ;;
  2)
    ARM=per_anchor_zscore
    MODEL=gemma
    CONFIG="${ZSCORE_CONFIG}"
    RUN_DIR="${ZSCORE_RUN}"
    CHECKPOINT="${ZSCORE_CHECKPOINT}"
    ;;
  3)
    ARM=per_anchor_zscore
    MODEL=smol
    CONFIG="${ZSCORE_CONFIG}"
    RUN_DIR="${ZSCORE_RUN}"
    CHECKPOINT="${ZSCORE_CHECKPOINT}"
    ;;
  *)
    echo "unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/orange/thai/vinhnguyen1/aspd_v3/zscore_mechanistic_validation_v1/causal_screen_v1/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/${ARM}/${MODEL}"
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${CAUSAL_PLAN}"
test -s "${CONTRACT}"
test -s "${RUN_DIR}/run_manifest.json"
test -s "${CHECKPOINT}/manifest.json"
test -s "${CHECKPOINT}/COMPLETE"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs
AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 52428800
test "${AVAILABLE_INODES}" -ge 100000
lfs quota -u "${USER}" /orange/thai || true
lfs quota -g thai /orange/thai || true

echo "git_sha=$(git rev-parse HEAD) plan=${CAUSAL_PLAN} arm=${ARM} model=${MODEL} checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
date --iso-8601=seconds
nvidia-smi
"${PYTHON}" -m aspd.multimodel.cli.run_aspd_v3_causal_screen \
  --config "${CONFIG}" --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --plan "${CAUSAL_PLAN}" --contract "${CONTRACT}" --model-key "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" --local-files-only

test -s "${OUTPUT_DIR}/experiment_manifest.json"
test -s "${OUTPUT_DIR}/intervention_results.jsonl"
test -s "${OUTPUT_DIR}/target_selection.json"
test -s "${OUTPUT_DIR}/causal_summary.json"
echo "ASPD_V3_CAUSAL_C1_COMPLETE output=${OUTPUT_DIR}"

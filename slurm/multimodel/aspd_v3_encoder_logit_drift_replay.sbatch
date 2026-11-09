#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-logit-drift
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/aspd-v3-logit-drift-%j.out
#SBATCH --error=slurm/logs/aspd-v3-logit-drift-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the decoder-norm correction worktree}"
module load conda
source ../env.sh
: "${TRAIN_JOB_ID:?Set the completed corrected pilot job ID}"
: "${HELDOUT_JOB_ID:?Set the completed held-out evaluation job ID}"

CONFIG=configs/multimodel/aspd_v3/granite_normalized_target_unconstrained_decoder_reliability_pilot_v1.yaml
CONTRACT=configs/multimodel/aspd_v3/granite_normalized_target_encoder_logit_drift_replay_v1.yaml
TORCHRUN=../repo/.venv/bin/torchrun
RUN_DIR="/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/reliability/normalized_target/runs/job_${TRAIN_JOB_ID}"
CHECKPOINT_0="${RUN_DIR}/checkpoints/step-000001000"
CHECKPOINT_1="${RUN_DIR}/checkpoints/step-000001221"
FEATURE_ACTIVITY="/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/evaluation/normalized_target/job_${HELDOUT_JOB_ID}/heldout/feature_activity.csv"
OUTPUT_ROOT="/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/diagnostics/encoder_logit_drift/job_${SLURM_JOB_ID}"

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

test -z "$(git status --porcelain)"
test -s "${CHECKPOINT_0}/COMPLETE"
test -s "${CHECKPOINT_1}/COMPLETE"
test -s "${FEATURE_ACTIVITY}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_ROOT}")" slurm/logs

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=3 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_encoder_logit_drift \
  --config "${CONFIG}" --contract "${CONTRACT}" \
  --run-dir "${RUN_DIR}" --checkpoint-0 "${CHECKPOINT_0}" \
  --checkpoint-1 "${CHECKPOINT_1}" --feature-activity "${FEATURE_ACTIVITY}" \
  --output-dir "${OUTPUT_ROOT}" --local-files-only

test -s "${OUTPUT_ROOT}/experiment_manifest.json"
test -s "${OUTPUT_ROOT}/counterfactual_metrics.json"
test -s "${OUTPUT_ROOT}/feature_attribution.csv"
test -s "${OUTPUT_ROOT}/group_attribution.csv"
echo "ASPD_V3_ENCODER_LOGIT_DRIFT_REPLAY_COMPLETE root=${OUTPUT_ROOT}"

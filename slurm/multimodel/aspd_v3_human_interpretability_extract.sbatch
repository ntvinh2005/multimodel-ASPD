#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-human-extract
#SBATCH --partition=hpg-b200
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:b200:2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/aspd-v3-human-extract-%A_%a.out
#SBATCH --error=slurm/logs/aspd-v3-human-extract-%A_%a.err

set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?submit from the human-interpretability worktree}"
module load conda
source ../../env.sh

CONFIGS=(
  configs/multimodel/aspd_v3/two_model_gate_collapse_control_v1.yaml
  configs/multimodel/aspd_v3/two_model_gate_collapse_zscore_v1.yaml
)
RUN_DIRS=(
  /orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/full_only_control/pilot/runs/job_50280055_0
  /orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/per_anchor_zscore/pilot/runs/job_50280055_1
)
CHECKPOINTS=(
  /orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/full_only_control/pilot/runs/job_50280055_0/checkpoints/step-000001221
  /orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/per_anchor_zscore/pilot/runs/job_50280055_1/checkpoints/step-000001221
)
INDEX="${SLURM_ARRAY_TASK_ID:?array index required}"
CONFIG="${CONFIGS[${INDEX}]}"
RUN_DIR="${RUN_DIRS[${INDEX}]}"
CHECKPOINT="${CHECKPOINTS[${INDEX}]}"
PROTOCOL=configs/multimodel/aspd_v3/human_interpretability_v1.yaml
PYTHON=../.venv/bin/python
TORCHRUN=../.venv/bin/torchrun
export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export TMPDIR="${SLURM_TMPDIR:?Slurm node-local temporary directory is required}"

test -z "$(git status --porcelain)"
test -s "${RUN_DIR}/exposure_ledger.sqlite"
test -s "${CHECKPOINT}/COMPLETE"
test -s "${PROTOCOL}"
test -s /orange/thai/vinhnguyen1/aspd_v3/human_interpretability_v1/provenance/excluded_documents.json
ARM="$(${PYTHON} -c 'import sys; from aspd.multimodel.v3.gate_collapse_recovery_config import load_gate_collapse_recovery_config as load; print(load(sys.argv[1]).arm_id)' "${CONFIG}")"
OUTPUT_ROOT=/orange/thai/vinhnguyen1/aspd_v3/human_interpretability_v1/extraction
OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}_${INDEX}/${ARM}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")" slurm/logs

AVAILABLE_KB="$(df -Pk /orange/thai | awk 'NR==2 {print $4}')"
AVAILABLE_INODES="$(df -Pi /orange/thai | awk 'NR==2 {print $4}')"
test "${AVAILABLE_KB}" -ge 52428800
test "${AVAILABLE_INODES}" -ge 100000
lfs quota -u "${USER}" /orange/thai || true
lfs quota -g thai /orange/thai || true

echo "arm=${ARM} git_sha=$(git rev-parse HEAD) protocol_sha256=$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
echo "output=${OUTPUT_DIR} estimated_peak_output_bytes=3221225472"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.extract_aspd_v3_mechanistic \
  --mode test --config "${CONFIG}" --protocol "${PROTOCOL}" \
  --run-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" --local-files-only

test -s "${OUTPUT_DIR}/COMPLETE"
${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["arm"]==sys.argv[2]
assert p["accepted_evaluation_anchors"]==100352
assert p["excluded_document_count"]==585
assert p["excluded_documents_disjoint"] is True
assert p["natural_gate_only"] is True and p["forced_activations"]==0
assert all(r["source_weights_unchanged"]=="PASS" for r in p["rank_reports"])
' "${OUTPUT_DIR}/experiment_manifest.json" "${ARM}"

echo "ASPD_V3_HUMAN_EXTRACTION_COMPLETE arm=${ARM} output=${OUTPUT_DIR}"

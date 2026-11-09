#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3.1-gate-root
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/aspd-v3.1-gate-root-%j.out
#SBATCH --error=slurm/logs/aspd-v3.1-gate-root-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the gate-collapse worktree}"
module load conda
source ../../env.sh

MAT_CONFIG=configs/multimodel/aspd_v3/two_model_matryoshka_v1.yaml
FULL_CONFIG=configs/multimodel/aspd_v3/two_model_full_only_v1.yaml
CONTRACT=configs/multimodel/aspd_v3/two_model_gate_collapse_diagnostic_v1.yaml
MAT_RUN=/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_matryoshka_ablation_v1/matryoshka/pilot/runs/job_50231651_0
FULL_RUN=/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_matryoshka_ablation_v1/full_only/pilot/runs/job_50231651_1
OUTPUT_ROOT=/orange/thai/vinhnguyen1/aspd_v3/gate_collapse_diagnostic_v1/job_${SLURM_JOB_ID}
TORCHRUN=../.venv/bin/torchrun

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export TMPDIR=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/tmp/aspd-v3.1-gate-root-${SLURM_JOB_ID}

test -z "$(git status --porcelain)"
test -s "${MAT_RUN}/checkpoints/step-000001000/COMPLETE"
test -s "${FULL_RUN}/checkpoints/step-000001000/COMPLETE"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${TMPDIR}" "$(dirname "${OUTPUT_ROOT}")" slurm/logs

AVAILABLE_KB=$(df -Pk /orange/thai | awk 'NR==2 {print $4}')
test "${AVAILABLE_KB}" -ge 10485760
df -h /orange/thai /blue/thai
df -i /orange/thai /blue/thai
lfs quota -u "${USER}" /orange/thai
lfs quota -g thai /orange/thai
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.evaluate_aspd_v3_two_model_gate_collapse \
  --matryoshka-config "${MAT_CONFIG}" \
  --full-only-config "${FULL_CONFIG}" \
  --contract "${CONTRACT}" \
  --matryoshka-run "${MAT_RUN}" \
  --full-only-run "${FULL_RUN}" \
  --output-dir "${OUTPUT_ROOT}" \
  --local-files-only

test -s "${OUTPUT_ROOT}/experiment_manifest.json"
test -s "${OUTPUT_ROOT}/counterfactual_metrics.json"
test -s "${OUTPUT_ROOT}/gradient_summary.json"
test -s "${OUTPUT_ROOT}/feature_attribution.csv"
test -s "${OUTPUT_ROOT}/feature_gradients_gemma.csv"
test -s "${OUTPUT_ROOT}/feature_gradients_smol.csv"
echo "ASPD_V3_1_GATE_COLLAPSE_DIAGNOSTIC_COMPLETE root=${OUTPUT_ROOT}"

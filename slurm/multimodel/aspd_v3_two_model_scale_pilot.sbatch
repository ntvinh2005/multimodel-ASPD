#!/usr/bin/env bash
#SBATCH --job-name=aspd-v3-2m-scale
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/aspd-v3-two-model-scale-%j.out
#SBATCH --error=slurm/logs/aspd-v3-two-model-scale-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the two-model implementation worktree}"
module load conda
source ../env.sh

: "${ARM:?Set ARM=baseline or ARM=treatment}"
case "${ARM}" in
  baseline)
    CONFIG=configs/multimodel/aspd_v3/two_model_scale_baseline_v1.yaml
    ;;
  treatment)
    CONFIG=configs/multimodel/aspd_v3/two_model_scale_treatment_v1.yaml
    ;;
  *)
    echo "ARM must be baseline or treatment" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${PWD}"
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

PYTHON=../repo/.venv/bin/python
TORCHRUN=../repo/.venv/bin/torchrun
GIT_SHA="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
OUTPUT_ROOT="$(${PYTHON} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["output_root"])' "${CONFIG}")"
PREFLIGHT="$(${PYTHON} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["preflight_evidence"])' "${CONFIG}")"
PREFLIGHT_DIR="${PREFLIGHT%/preflight_report.json}"
RUN_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"
test ! -e "${PREFLIGHT}"
test ! -e "${RUN_DIR}"

mkdir -p slurm/logs out/multimodel/aspd_v3/two_model_scale_v1/data/pile_uncopyrighted
exec 9>out/multimodel/aspd_v3/two_model_scale_v1/data/manifest.lock
flock 9
if [[ ! -s out/multimodel/aspd_v3/two_model_scale_v1/data/pile_uncopyrighted/manifest.json ]]; then
  ${PYTHON} -m aspd.multimodel.cli.prepare_aspd_v3_main_manifests \
    --two-model-scale --main-config "${CONFIG}"
fi
flock -u 9

echo "git_sha=${GIT_SHA}"
echo "arm=${ARM}"
echo "config=${CONFIG}"
echo "preflight=${PREFLIGHT}"
echo "run_dir=${RUN_DIR}"
date --iso-8601=seconds
nvidia-smi

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode preflight \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${PREFLIGHT_DIR}" \
  --max-steps 3 \
  --warmup-steps 1 \
  --local-files-only

${PYTHON} -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="PASS"
assert p["mode"]=="preflight"
assert p["world_size"]==2
assert p["logical_target_count"]==112
assert p["dcp_save_reload_resume_update"]=="PASS"
assert p["checkpoint_parameter_and_adam_restore_probe"]=="PASS"
' "${PREFLIGHT}"

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 \
  -m aspd.multimodel.cli.train_aspd_v3_2b \
  --mode pilot \
  --strategy model_owned \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --max-steps 1221 \
  --warmup-steps 20 \
  --local-files-only

${PYTHON} -c '
import json,sys
root=sys.argv[1]
p=json.load(open(root+"/benchmark_report.json"))
assert p["status"]=="PASS"
assert p["optimizer_steps"]==1221
assert p["accepted_exposures"]==10002432
assert p["world_size"]==2
assert p["logical_target_count"]==112
assert p["checkpoint_manifest"]["status"]=="COMPLETE"
' "${RUN_DIR}"

echo "ASPD_V3_TWO_MODEL_SCALE_PILOT_COMPLETE arm=${ARM} run=${RUN_DIR}"

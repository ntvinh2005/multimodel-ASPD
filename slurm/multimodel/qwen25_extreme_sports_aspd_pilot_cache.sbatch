#!/usr/bin/env bash
# Initial Qwen2.5 ASPD pilot cache; requires inspected GPU preflight artifacts.
#SBATCH --job-name=mm-qwen25-aspd-cache
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail

module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_pilot
PREFLIGHT=out/multimodel/diagnostics/aspd_preflight_46691095
test -s "${PREFLIGHT}/static_config.json"
test -s "${PREFLIGHT}/qwen25_peft_equivalence.json"
test -s "${PREFLIGHT}/qwen25_effective_delta_w.json"
# Pin the inspected job-scoped preflight artifacts while this job waits in queue.
[[ "$(sha256sum "${PREFLIGHT}/static_config.json" | cut -d ' ' -f 1)" == "a37664527b47ff45747da7f0346d70ab4a50ce6e845849d2373d0bdb05af4cbd" ]]
[[ "$(sha256sum "${PREFLIGHT}/qwen25_peft_equivalence.json" | cut -d ' ' -f 1)" == "bad2561f6cd1f998739b7fed8d58fc0118afc5be9936dd2d28c43b155c08a037" ]]
[[ "$(sha256sum "${PREFLIGHT}/qwen25_effective_delta_w.json" | cut -d ' ' -f 1)" == "d316f2cd25bbc04fb8039ae683c3df4f506b4244a32459af56e01cb4496bdba1" ]]
if [[ -e "${CACHE}" ]]; then
  echo "Refusing to reuse or overwrite ASPD pilot cache: ${CACHE}" >&2
  exit 21
fi

printf 'hostname=%s\n' "$(hostname)"
date
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\ncache=%s\npreflight=%s\n' "${CONFIG}" "${CACHE}" "${PREFLIGHT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --device cuda
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
printf 'ASPD_PILOT_CACHE_COMPLETE\n'

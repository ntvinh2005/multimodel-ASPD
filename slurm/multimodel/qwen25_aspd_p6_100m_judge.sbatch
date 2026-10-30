#!/usr/bin/env bash
# Blind judge + report for one immutable discovery generation; no intervention model loaded.
#SBATCH --job-name=q25-aspd-p6-100m-judge
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to the pinned project cache}"
: "${P6_APPROVED_COMMIT:?Set P6_APPROVED_COMMIT to the reviewed clean commit}"
: "${P6_DISCOVERY_ROOT:?Set P6_DISCOVERY_ROOT to one completed discovery/job_<id>}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${P6_APPROVED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

CONFIG=configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml
GENERATION=${P6_DISCOVERY_ROOT}/generation
PROMPTS=configs/multimodel/qwen25_extreme_sports/p6_100m_discovery_prompts.jsonl
SCORES=${P6_DISCOVERY_ROOT}/behavioral_scores/job_${SLURM_JOB_ID}
REPORT=${P6_DISCOVERY_ROOT}/report/job_${SLURM_JOB_ID}
JUDGE_MODEL=google/gemma-2-9b-it
JUDGE_REVISION=11c9b309abf73637e4b6f9a3fa1e92e615547819
test -s "${GENERATION}/run_manifest.json"
test -s "${GENERATION}/generation_records.jsonl"
[[ ! -e "${SCORES}" && ! -e "${REPORT}" ]]

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export UV_PYTHON=3.13
printf 'stage=BLINDED_BEHAVIORAL_SCORING\nhostname=%s\ndate=%s\ngit_commit=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "${CURRENT_COMMIT}"
.venv/bin/python - <<'PY'
import platform, torch
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
PY
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
printf 'config=%s\ngeneration=%s\nprompts=%s\nscores=%s\nreport=%s\njudge_model=%s\njudge_revision=%s\n' \
  "${CONFIG}" "${GENERATION}" "${PROMPTS}" "${SCORES}" "${REPORT}" \
  "${JUDGE_MODEL}" "${JUDGE_REVISION}"

.venv/bin/python -m aspd.multimodel.cli.audit_p6_100m "${CONFIG}"
.venv/bin/python -m aspd.multimodel.cli.judge_p6_100m \
  --generation-dir "${GENERATION}" --prompts "${PROMPTS}" \
  --output-dir "${SCORES}" --model "${JUDGE_MODEL}" \
  --revision "${JUDGE_REVISION}" --device cuda --batch-size 16 --max-new-tokens 128
.venv/bin/python -m aspd.multimodel.cli.report_p6_100m \
  --generation-dir "${GENERATION}" \
  --behavioral-scores "${SCORES}/behavioral_scores.json" \
  --output-dir "${REPORT}"
printf 'P6_100M_DISCOVERY_SCORED_NOT_INDEPENDENTLY_VERIFIED\nREPORT=%s\n' "${REPORT}"

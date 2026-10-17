#!/usr/bin/env bash
# One B200, one model load, batched post-ranking semantic triage. STAGE is backfill, smoke, or full.
# Candidate planning is CPU-only and should normally be run on a login node with --dry-run.
#SBATCH --job-name=mm-semantic-triage
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
module load conda

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/../env.sh" ]]; then
  source "${SUBMIT_DIR}/../env.sh"
fi
cd "${SUBMIT_DIR}"

: "${STAGE:?Set STAGE=backfill, smoke, or full}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"

CONFIG="${CONFIG:-configs/multimodel/crosscoder/r1_qwen3_full.yaml}"
RUN_DIR="${RUN_DIR:-out/multimodel/baselines/crosscoder/r1_qwen3_full}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_DIR}/analysis}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/$(cat "${RUN_DIR}/latest_checkpoint.txt")}"
PLAN="${PLAN:-${ANALYSIS_DIR}/content_semantic_scan_candidates.json}"
MODEL="${JUDGE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
REVISION="${JUDGE_REVISION:-916b56a44061fd5cd7d6a8fb632557ed4f724f60}"
ENGINE="${ENGINE:-transformers}"
START_SECONDS="${SECONDS}"

mkdir -p slurm/logs
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
printf 'git_commit=%s\nstage=%s\nconfig=%s\ncheckpoint=%s\nanalysis=%s\njudge=%s@%s\nengine=%s\n' \
  "$(git rev-parse HEAD)" "${STAGE}" "${CONFIG}" "${CHECKPOINT}" "${ANALYSIS_DIR}" \
  "${MODEL}" "${REVISION}" "${ENGINE}"
nvidia-smi

common=(
  --method crosscoder
  --analysis-dir "${ANALYSIS_DIR}"
  --judge-model "${MODEL}"
  --judge-revision "${REVISION}"
  --tokenizer-revision "${REVISION}"
  --engine "${ENGINE}"
  --device cuda
  --base-tail-count "${BASE_TAIL_COUNT:-200}"
  --finetuned-tail-count "${FINETUNED_TAIL_COUNT:-200}"
  --min-fire-count "${MIN_FIRE_COUNT:-32}"
  --contexts-per-feature "${CONTEXTS_PER_FEATURE:-8}"
  --min-contexts-for-judge "${MIN_CONTEXTS_FOR_JUDGE:-6}"
  --structural-threshold "${STRUCTURAL_THRESHOLD:-0.70}"
  --temperature 0
  --max-new-tokens "${MAX_NEW_TOKENS:-160}"
  --batch-size "${JUDGE_BATCH_SIZE:-16}"
)

case "${STAGE}" in
  backfill)
    test -s "${PLAN}"
    test -s "${CHECKPOINT}"
    .venv/bin/python -m aspd.multimodel.cli.backfill_semantic_contexts \
      --method crosscoder \
      --analysis-dir "${ANALYSIS_DIR}" \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --candidate-plan "${PLAN}" \
      --top-contexts "${CONTEXTS_PER_FEATURE:-8}" \
      --device cuda
    ;;
  smoke)
    .venv/bin/python -m aspd.multimodel.cli.semantic_triage \
      "${common[@]}" \
      --max-judge-features 10 \
      --output "${ANALYSIS_DIR}/content_semantic_scan_smoke.json" \
      --review-output "${ANALYSIS_DIR}/content_semantic_review_candidates_smoke.json"
    .venv/bin/python -c '
import json, sys
p=json.load(open(sys.argv[1]))
s=p["summary"]; t=p["timing"]
assert s["judge_requested"] == 10, "smoke must judge exactly 10 features, got {}".format(s["judge_requested"])
assert s["judge_success"] == 10, "smoke must parse 10/10 features, got {}".format(s["judge_success"])
print(json.dumps({"summary": s, "timing": t}, indent=2))
if t["full_submission_within_45_minute_guardrail"] is not True:
    raise SystemExit("STOP: projected full judge runtime exceeds 45 minutes or is unavailable")
' "${ANALYSIS_DIR}/content_semantic_scan_smoke.json"
    ;;
  full)
    SMOKE="${ANALYSIS_DIR}/content_semantic_scan_smoke.json"
    test -s "${SMOKE}"
    if [[ "${FORCE_FULL:-0}" != "1" ]]; then
      jq -e '.timing.full_submission_within_45_minute_guardrail == true' "${SMOKE}" >/dev/null
    fi
    .venv/bin/python -m aspd.multimodel.cli.semantic_triage "${common[@]}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected backfill, smoke, or full" >&2
    exit 2
    ;;
esac

printf 'stage=%s wall_clock_seconds=%s\n' "${STAGE}" "$((SECONDS - START_SECONDS))"

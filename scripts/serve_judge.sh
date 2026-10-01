#!/bin/bash
# Serve the LLM judge (Llama-3.3-70B-Instruct) with vLLM on one GPU, as used for the paper.
# vLLM pins its own torch, so install it in a separate environment, e.g. `uv tool install vllm`.
#
#   scripts/serve_judge.sh            # port 8010, then pass --judge-base-url http://127.0.0.1:8010/v1
#   PORT=8020 JUDGE_MODEL=... scripts/serve_judge.sh
set -euo pipefail
JUDGE_MODEL="${JUDGE_MODEL:-unsloth/Llama-3.3-70B-Instruct}"
PORT="${PORT:-8010}"
exec vllm serve "$JUDGE_MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}"

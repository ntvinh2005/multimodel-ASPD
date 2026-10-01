#!/bin/bash
# Serve the LLM judge (Llama-3.3-70B-Instruct, vLLM) on one GPU; point evaluations at
# --judge-base-url http://<this node>:8010/v1. `vllm` must be on PATH (a separate environment).
#SBATCH --job-name=aspd-judge
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
echo "judge on $(hostname):${PORT:-8010}"
exec scripts/serve_judge.sh

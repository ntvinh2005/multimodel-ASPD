#!/bin/bash
# The circuit app (attribution graphs and component browser) for one run. Needs a GPU.
#   sbatch slurm/circuit_app.sbatch gpt2_all_aspd
# then run the two ssh commands the job prints and open http://localhost:8055
#SBATCH --job-name=aspd-circuit-app
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
RUN=${1:-gpt2_all_aspd}
PORT=${PORT:-8055}
echo "circuit app for $RUN on $(hostname):$PORT. From your machine:"
echo "  ssh -N -L $PORT:localhost:$PORT <user>@<login node>"
echo "and on the login node:"
echo "  ssh -N -L $PORT:localhost:$PORT $(hostname)"
exec python -m aspd.cli.serve_app --run "$RUN" --port "$PORT"

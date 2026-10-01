#!/bin/bash
# The pair app (IOI mechanisms on /prompt, semantic tracing on /) for one run. CPU only.
#   sbatch slurm/pair_app.sbatch gpt2_all_aspd
# then run the two ssh commands the job prints and open http://localhost:8060
#SBATCH --job-name=aspd-pair-app
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
RUN=${1:-gpt2_all_aspd}
PORT=${PORT:-8060}
echo "pair app for $RUN on $(hostname):$PORT. From your machine:"
echo "  ssh -N -L $PORT:localhost:$PORT <user>@<login node>"
echo "and on the login node:"
echo "  ssh -N -L $PORT:localhost:$PORT $(hostname)"
exec python -m aspd.cli.serve_pairs --run "$RUN" --port "$PORT" --qk-edit weight

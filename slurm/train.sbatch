#!/bin/bash
# Train one method: sbatch --gres=gpu:<runtime.dp or 1> slurm/train.sbatch configs/gpt2/aspd.yaml gpt2_aspd
#SBATCH --job-name=aspd-train
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
CONFIG=$1
RUN_ID=$2
NGPU=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['runtime'].get('dp') or 1)" "$CONFIG")
if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" -m aspd.cli.train "$CONFIG" --run-id "$RUN_ID" --resume auto
else
  python -m aspd.cli.train "$CONFIG" --run-id "$RUN_ID" --resume auto
fi

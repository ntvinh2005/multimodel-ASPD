#!/bin/bash
# Every Section 4 evaluation of one run, then its table row:
#   sbatch slurm/eval.sbatch gpt2_aspd gpt2 http://<judge node>:8010/v1
#SBATCH --job-name=aspd-eval
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
RUN_ID=$1
MODEL=$2
JUDGE=${3:-http://127.0.0.1:8010/v1}
R=${PARAM_DECOMP_OUT_DIR:-out}/runs/$RUN_ID
SAE=artifacts/saes/$MODEL

[ -d "$R/harvest" ] || python -m aspd.cli.harvest --run-dir "$R"
python -m aspd.cli.intruder --run-dir "$R" --judge-base-url "$JUDGE"
python -m aspd.cli.diversity --run-dirs "$R"
python -m aspd.cli.matching --run-dir "$R" --sae-dir "$SAE" --judge-base-url "$JUDGE"
python -m aspd.cli.editing --run-dir "$R" --sae-dir "$SAE"
for m in 1 5 10 20 50; do
  python -m aspd.cli.editing_multi --run-dir "$R" --sae-dir "$SAE" --m "$m" --setup cond
done
python -m aspd.cli.tables --run-dir "$R" --ci --out "tables/$RUN_ID.csv"

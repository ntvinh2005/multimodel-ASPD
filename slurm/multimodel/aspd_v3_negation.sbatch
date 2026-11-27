#!/usr/bin/env bash
#SBATCH --job-name=aspd-v31-negation
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/blue/thai/share/multimodel-ASPD/logs/%x-%j.out
#SBATCH --error=/blue/thai/share/multimodel-ASPD/logs/%x-%j.err

set -euo pipefail

REPO=${REPO:?absolute path to the Negation worktree}
REVISION=${REVISION:?expected full Git revision}
PREPARATION=${PREPARATION:?validated Negation preparation}
OUTPUT=${OUTPUT:?new immutable experiment output directory}
STAGE=${STAGE:?smoke, discovery or graphs}
CONFIG=configs/multimodel/aspd_v3/negation_mechanism_eval_v1.yaml

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${REVISION}"
test -z "$(git status --porcelain --untracked-files=no)"
test -s "${PREPARATION}/PREPARED"
test ! -e "${OUTPUT}"

export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python

case "${STAGE}" in
  smoke|discovery)
    if [[ "${STAGE}" == discovery ]]; then
      test -s "${SMOKE:?validated smoke output}/COMPLETE"
      "${PYTHON}" -c 'import sys; from pathlib import Path; from aspd.multimodel.cli.evaluate_aspd_v3_negation import verify_evaluation; verify_evaluation(Path(sys.argv[1]), "smoke")' "${SMOKE}"
    fi
    exec "${PYTHON}" -u -m aspd.multimodel.cli.evaluate_aspd_v3_negation \
      --config "${CONFIG}" --preparation "${PREPARATION}" --output "${OUTPUT}" --split "${STAGE}"
    ;;
  graphs)
    exec "${PYTHON}" -u -m aspd.multimodel.cli.measure_negation_query \
      --config "${CONFIG}" --preparation "${PREPARATION}" --output "${OUTPUT}" \
      --discovery "${DISCOVERY:?validated discovery output}"
    ;;
  *) echo "Unsupported stage ${STAGE}" >&2; exit 2 ;;
esac

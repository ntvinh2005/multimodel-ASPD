#!/usr/bin/env bash
# Full-validation computation-fidelity audit of the final L07--L13 checkpoint.
#SBATCH --job-name=mm-q25-up713-fidelity
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
source "${SUBMIT_DIR}/../env.sh"; cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l07_l13_up_1m.yaml
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m
CHECKPOINT=${RUN}/checkpoint_00004096.pt
OUTPUT_DIR=out/multimodel/diagnostics/qwen25_aspd_l07_l13_up_1m
OUTPUT=${OUTPUT_DIR}/step4096_full_validation.json

printf 'hostname=%s\n' "$(hostname)"; date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\ncheckpoint=%s\noutput=%s\n' "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CHECKPOINT}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
test -s "${RUN}/training_integrity.json"; test -s "${CHECKPOINT}"
if [[ -e "${OUTPUT}" ]]; then echo "Refusing to overwrite ${OUTPUT}" >&2; exit 21; fi
.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${CHECKPOINT}" --device cuda --batch-size-sequences 2 --output "${OUTPUT}"
.venv/bin/python - "${OUTPUT}" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1])); expected={f"L{layer:02d}_up" for layer in range(7,14)}
assert x["checkpoint"]["step"] == 4096 and x["evaluation"]["tokens"] == 131072
assert x["integrity"]["matrix_count"] == 7 and set(x["matrices"]) == expected
assert x["integrity"]["max_decomposition_relative_rms_residual"] <= 2e-5
for matrix in x["matrices"].values():
    for key in ("base","target","delta_total","delta_parameter","delta_input"):
        assert "rmse" in matrix[key] and "actual_output_rms" in matrix[key]
    assert "error_cancellation" in matrix
def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)
assert finite(x)
print("ASPD_L07_L13_UP_FIDELITY_PASS")
PY
printf 'ASPD_L07_L13_UP_FIDELITY_COMPLETE\n'

#!/usr/bin/env bash
# Common-input fidelity audit for old 1k, new 2048, and new 4096 checkpoints.
#SBATCH --job-name=mm-q25-aspd-1m-fidelity
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
source "${SUBMIT_DIR}/../env.sh"
cd "${SUBMIT_DIR}"
: "${HF_HOME:?HF_HOME must point to project/Blue storage}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
OLD_CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_fidelity1000.yaml
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m
OLD_CHECKPOINT=out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000/checkpoint_00001000.pt
OLD_CHECKPOINT_SHA=ff00347a0b90f789a794fd7e67744252bf81ddee82b992135db80d38c348b8bf
OLD_CACHE_SHA=1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447
OUTPUT=out/multimodel/diagnostics/qwen25_aspd_general1m

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\noutput=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

test -s "${RUN}/checkpoint_00002048.pt"
test -s "${RUN}/checkpoint_00004096.pt"
for path in \
  "${OUTPUT}/step1000_on_general1m_validation.json" \
  "${OUTPUT}/step2048_full_validation.json" \
  "${OUTPUT}/step4096_full_validation.json" \
  "${OUTPUT}/comparison.json"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite general1m fidelity artifact: ${path}" >&2
    exit 21
  fi
done
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only

# Each process releases its multi-GB checkpoint before the next checkpoint is loaded.
.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${OLD_CHECKPOINT}" \
  --checkpoint-config "${OLD_CONFIG}" \
  --device cuda --batch-size-sequences 2 \
  --expected-cache-fingerprint "${OLD_CACHE_SHA}" \
  --expected-checkpoint-sha256 "${OLD_CHECKPOINT_SHA}" \
  --output "${OUTPUT}/step1000_on_general1m_validation.json"
.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${RUN}/checkpoint_00002048.pt" \
  --device cuda --batch-size-sequences 2 \
  --output "${OUTPUT}/step2048_full_validation.json"
.venv/bin/python -m aspd.multimodel.cli.evaluate_computation_fidelity \
  "${CONFIG}" "${RUN}/checkpoint_00004096.pt" \
  --device cuda --batch-size-sequences 2 \
  --output "${OUTPUT}/step4096_full_validation.json"

OUTPUT="${OUTPUT}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
records = {
    label: json.load(open(root / filename, encoding="utf-8"))
    for label, filename in {
        "step1000": "step1000_on_general1m_validation.json",
        "step2048": "step2048_full_validation.json",
        "step4096": "step4096_full_validation.json",
    }.items()
}
expected_steps = {"step1000": 1000, "step2048": 2048, "step4096": 4096}
fingerprints = {row["cache"]["fingerprint"] for row in records.values()}
assert len(fingerprints) == 1
for label, row in records.items():
    assert row["checkpoint"]["step"] == expected_steps[label]
    assert row["evaluation"]["tokens"] == 131072
    assert row["integrity"]["matrix_count"] == 7
    # DeltaY is formed from several FP32 matrix products.  Allow a small
    # accumulation margin while still failing loudly on material identity drift.
    assert row["integrity"]["max_decomposition_relative_rms_residual"] <= 2e-5

def aggregate(row, metric):
    values = [matrix[metric] for matrix in row["matrices"].values()]
    sse = sum(value["sse"] for value in values)
    energy = sum(value["target_energy"] for value in values)
    variance = sum(value["variance_denominator"] for value in values)
    return {
        "energy_normalized_squared_error": sse / energy,
        "variance_weighted_fvu": sse / variance,
        "error_improvement_over_zero": 1.0 - sse / energy,
    }

comparison = {
    "schema_version": 1,
    "status": "MEASURED_NOT_INTERPRETED",
    "common_evaluation_cache_fingerprint": next(iter(fingerprints)),
    "validation_tokens": 131072,
    "checkpoints": {},
}
for label, row in records.items():
    comparison["checkpoints"][label] = {
        "checkpoint": row["checkpoint"],
        "checkpoint_training_cache": row["checkpoint_training_cache"],
        "activation_fvu": {
            model: row["activation_reconstruction"][model]["fvu"]
            for model in ("base", "target")
        },
        "internal_mean_matrix_fvu": row["internal_mean_matrix_fvu"],
        "delta_total": aggregate(row, "delta_total"),
        "delta_parameter": aggregate(row, "delta_parameter"),
        "delta_input": aggregate(row, "delta_input"),
        "per_matrix": {
            name: {
                metric: {
                    "base_fvu": matrix["base"]["fvu"],
                    "target_fvu": matrix["target"]["fvu"],
                    "nse": matrix[metric]["normalized_squared_error"],
                }
                for metric in ("delta_total", "delta_parameter", "delta_input")
            }
            for name, matrix in row["matrices"].items()
        },
    }

def visit(value):
    if isinstance(value, dict):
        for child in value.values(): visit(child)
    elif isinstance(value, list):
        for child in value: visit(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit("non-finite fidelity comparison")
visit(comparison)
(root / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
print(json.dumps(comparison, sort_keys=True))
PY

printf 'ASPD_GENERAL1M_FIDELITY_COMPLETE\n'

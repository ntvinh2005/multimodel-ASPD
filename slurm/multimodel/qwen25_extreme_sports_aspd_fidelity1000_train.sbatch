#!/usr/bin/env bash
# Submit only with --dependency=afterok:<fidelity-preflight-job-id>.
# Runtime checks also require the preflight artifacts; dependency alone is insufficient.
#SBATCH --job-name=mm-q25-aspd-fid1k
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=04:00:00
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_fidelity1000.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_pilot
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000
BASELINE=out/multimodel/diagnostics/qwen25_aspd_fidelity/baseline200_full_validation.json
TINY=out/multimodel/diagnostics/qwen25_aspd_fidelity/tiny_overfit_75.json
CACHE_SHA=1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'config=%s\ncache=%s\nrun=%s\nbaseline=%s\ntiny=%s\n' \
  "${CONFIG}" "${CACHE}" "${RUN}" "${BASELINE}" "${TINY}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

if [[ -e "${RUN}" ]]; then
  echo "Refusing to reuse or overwrite controlled fidelity run: ${RUN}" >&2
  exit 21
fi
test -s "${BASELINE}"
test -s "${TINY}"
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only

CONFIG="${CONFIG}" BASELINE="${BASELINE}" TINY="${TINY}" CACHE_SHA="${CACHE_SHA}" \
  .venv/bin/python - <<'PY'
import json
import os

from aspd.multimodel.config import load_experiment_config

cfg = load_experiment_config(os.environ["CONFIG"])
baseline = json.load(open(os.environ["BASELINE"], encoding="utf-8"))
tiny = json.load(open(os.environ["TINY"], encoding="utf-8"))
assert baseline["cache"]["fingerprint"] == os.environ["CACHE_SHA"]
assert baseline["evaluation"]["tokens"] == 16384
assert tiny["cache_fingerprint"] == os.environ["CACHE_SHA"]
assert tiny["passed"] is True
assert cfg.training.resume is None
assert cfg.training.max_steps == 1000
assert cfg.training.validate_every == 100
assert cfg.training.validation_batches == 32
assert cfg.training.save_every == 250
assert cfg.training.keep_last_checkpoints == 2
assert cfg.training.output_dir == "out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000"
print("ASPD_FIDELITY1000_RUNTIME_INPUTS_PASS")
PY

.venv/bin/python -m aspd.multimodel.cli.train "${CONFIG}" --device cuda

RUN="${RUN}" .venv/bin/python - <<'PY'
import json
import math
import os
from pathlib import Path

import torch

run = Path(os.environ["RUN"])
checkpoint = run / "checkpoint_00001000.pt"
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
assert (run / "latest_checkpoint.txt").read_text().strip() == checkpoint.name
state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
assert state["step"] == 1000
for name, tensor in state["model"].items():
    if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
        raise SystemExit(f"non-finite checkpoint tensor: {name}")
rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
assert any(row.get("step") == 1000 for row in rows)
assert any("validation/loss" in row and row.get("step") == 1000 for row in rows)
for row in rows:
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise SystemExit("non-finite training metric")
print("ASPD_FIDELITY1000_CHECKPOINT_INTEGRITY_PASS")
PY

printf 'ASPD_FIDELITY1000_TRAINING_COMPLETE\n'

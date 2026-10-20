#!/usr/bin/env bash
# Fresh 4096-step D0/S1 training; submit with afterok on the cache job.
#SBATCH --job-name=mm-q25-aspd-1m-train
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

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_general1m
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_general1m

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\nrun=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${CACHE}" "${RUN}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

if [[ -e "${RUN}" ]]; then
  echo "Refusing to overwrite general1m training run" >&2
  exit 21
fi
test -s "${CACHE}/cache_integrity.json"
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
CONFIG="${CONFIG}" CACHE="${CACHE}" .venv/bin/python - <<'PY'
import json, os
from aspd.multimodel.config import load_experiment_config
cfg = load_experiment_config(os.environ["CONFIG"])
audit = json.load(open(os.path.join(os.environ["CACHE"], "cache_integrity.json")))
assert audit["status"] == "PASS"
assert cfg.training.resume is None
assert cfg.training.max_steps == 4096
assert cfg.training.validation_batches == 256
assert cfg.training.save_every == 2048 and cfg.training.keep_last_checkpoints == 2
assert cfg.data.train_tokens == 1048576 and cfg.data.validation_tokens == 131072
print("ASPD_GENERAL1M_TRAIN_INPUTS_PASS")
PY

.venv/bin/python -m aspd.multimodel.cli.train "${CONFIG}" --device cuda

RUN="${RUN}" .venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
import torch
run = Path(os.environ["RUN"])
checkpoints = sorted(path.name for path in run.glob("checkpoint_*.pt"))
assert checkpoints == ["checkpoint_00002048.pt", "checkpoint_00004096.pt"]
assert (run / "latest_checkpoint.txt").read_text().strip() == "checkpoint_00004096.pt"
for name, expected_step in ((checkpoints[0], 2048), (checkpoints[1], 4096)):
    state = torch.load(run / name, map_location="cpu", weights_only=True, mmap=True)
    assert int(state["step"]) == expected_step
    for tensor_name, tensor in state["model"].items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
            raise SystemExit(f"non-finite checkpoint tensor: {name}/{tensor_name}")
rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
validation_steps = [row["step"] for row in rows if "validation/loss" in row]
assert validation_steps == [1024, 2048, 3072, 4096]
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float))
print("ASPD_GENERAL1M_CHECKPOINTS_PASS")
PY

printf 'ASPD_GENERAL1M_TRAIN_COMPLETE\n'

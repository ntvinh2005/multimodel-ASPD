#!/usr/bin/env bash
# Materialize and fully audit the isolated 1M-token paired ASPD cache.
#SBATCH --job-name=mm-q25-aspd-1m-cache
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128gb
#SBATCH --time=03:00:00
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
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${CACHE}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
blue_quota

if [[ -e "${CACHE}" || -e "${RUN}" ]]; then
  echo "Refusing to overwrite general1m cache or run" >&2
  exit 21
fi
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --device cuda
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only

CONFIG="${CONFIG}" CACHE="${CACHE}" .venv/bin/python - <<'PY'
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from aspd.multimodel.config import load_experiment_config

cfg = load_experiment_config(os.environ["CONFIG"])
root = Path(os.environ["CACHE"])
manifest = json.loads((root / "tokens/manifest.json").read_text())
assert manifest["requested_tokens"] == {"train": 1048576, "validation": 131072}
assert manifest["sequences"] == {"train": 4096, "validation": 512}
assert manifest["sequence_length"] == 256
assert manifest["tokenizer"] == "unsloth/Qwen2.5-7B-Instruct"
assert manifest["tokenizer_revision"] == "a75c9dc945567a9b6f568b8503a0307731607bee"
expected_matrices = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
token_rows = {"train": 0, "validation": 0}
checked_values = 0
for split in ("train", "validation"):
    token_files = sorted((root / "tokens" / split).glob("shard_*.pt"))
    for token_file in token_files:
        tokens = torch.load(token_file, map_location="cpu", weights_only=True)
        rows, width = tokens["input_ids"].shape
        assert width == 256 and tokens["valid_tokens"].shape == (rows, width)
        token_rows[split] += rows
        for model in cfg.models:
            shard = torch.load(
                root / "models" / model.name / split / token_file.name,
                map_location="cpu",
                weights_only=True,
            )
            assert shard["R"].shape[:2] == (rows, width)
            assert set(shard["X"]) == expected_matrices
            tensors = [shard["R"], *shard["X"].values()]
            for tensor in tensors:
                if not torch.isfinite(tensor).all():
                    raise SystemExit(f"non-finite cache tensor: {model.name}/{split}/{token_file.name}")
                checked_values += tensor.numel()
assert token_rows == {"train": 4096, "validation": 512}
for model in cfg.models:
    model_manifest = json.loads((root / "models" / model.name / "manifest.json").read_text())
    assert set(model_manifest["matrices"]) == expected_matrices
    assert model_manifest["schema_version"] >= 2
    if model.name == "target":
        assert model_manifest["adapter_merged"] is True
        assert model_manifest["adapter_revision"] == "6aa7e935c74c519c66555a111d1e6982059bd82d"
    weights = load_file(str(root / "models" / model.name / "weights.safetensors"))
    assert set(weights) == expected_matrices
    assert all(torch.isfinite(value).all() for value in weights.values())
report = {
    "status": "PASS",
    "cache_fingerprint": manifest["cache_fingerprint"],
    "token_rows": token_rows,
    "checked_tensor_values": checked_values,
    "matrix_names": sorted(expected_matrices),
}
(root / "cache_integrity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

printf 'ASPD_GENERAL1M_CACHE_COMPLETE\n'

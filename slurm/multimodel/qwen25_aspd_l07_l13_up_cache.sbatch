#!/usr/bin/env bash
# Build and audit the isolated 1M-token L07--L13 up_proj cache after memory preflight.
#SBATCH --job-name=mm-q25-up713-cache
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
: "${PREFLIGHT_JOB_ID:?Set PREFLIGHT_JOB_ID to the passed memory-preflight job}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l07_l13_up_1m.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_l07_l13_up_1m
SOURCE=out/multimodel/cache/qwen25_extreme_sports_aspd_general1m
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m
PREFLIGHT=out/multimodel/diagnostics/qwen25_aspd_l07_l13_up_1m/preflight_job_${PREFLIGHT_JOB_ID}/memory_preflight.json

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\nsource_tokens=%s\npreflight=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${CACHE}" "${SOURCE}" "${PREFLIGHT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
blue_quota

test -s "${PREFLIGHT}"
if [[ -e "${CACHE}" || -e "${RUN}" ]]; then
  echo "Refusing to overwrite L07--L13 cache or run" >&2
  exit 21
fi
.venv/bin/python - "${PREFLIGHT}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x["status"] == "PASS" and x["passed"] is True
assert x["configuration"]["components"] == 8192 and x["configuration"]["top_k"] == 32
print("ASPD_L07_L13_UP_PREFLIGHT_GATE_PASS")
PY
.venv/bin/python -m aspd.multimodel.cli.validate "${CONFIG}" --static >/dev/null
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --device cuda
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only

CONFIG="${CONFIG}" CACHE="${CACHE}" SOURCE="${SOURCE}" PREFLIGHT="${PREFLIGHT}" \
.venv/bin/python - <<'PY'
import hashlib, json, math, os
from pathlib import Path
import torch
from safetensors.torch import load_file
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.fidelity import resolved_config_sha256

cfg=load_experiment_config(os.environ["CONFIG"])
root=Path(os.environ["CACHE"]); source=Path(os.environ["SOURCE"])
manifest=json.loads((root/"tokens/manifest.json").read_text())
source_manifest=json.loads((source/"tokens/manifest.json").read_text())
assert (root/"tokens").is_symlink()
assert (root/"tokens").resolve() == (source/"tokens").resolve()
for key in ("cache_fingerprint","token_content_sha256","tokenizer_fingerprint","sequences","requested_tokens"):
    assert manifest[key] == source_manifest[key]
assert manifest["sequences"] == {"train":4096,"validation":512}
assert manifest["requested_tokens"] == {"train":1048576,"validation":131072}
expected=[f"L{layer:02d}_up" for layer in range(7,14)]
token_hash=hashlib.sha256(); rows={"train":0,"validation":0}; checked=0
for split in ("train","validation"):
    for shard_index,token_file in enumerate(sorted((root/"tokens"/split).glob("shard_*.pt"))):
        token=torch.load(token_file,map_location="cpu",weights_only=True)
        ids=token["input_ids"].contiguous(); valid=token["valid_tokens"]
        assert ids.shape == valid.shape and ids.shape[1] == 256 and valid.all()
        rows[split] += ids.shape[0]
        token_hash.update(split.encode()); token_hash.update(shard_index.to_bytes(8,"big")); token_hash.update(ids.numpy().tobytes())
        for model in cfg.models:
            cache=torch.load(root/"models"/model.name/split/token_file.name,map_location="cpu",weights_only=True)
            assert cache["R"].shape[:2] == ids.shape and list(cache["X"]) == expected
            for value in (cache["R"],*cache["X"].values()):
                assert torch.isfinite(value).all(); checked += value.numel()
assert rows == {"train":4096,"validation":512}
assert token_hash.hexdigest() == manifest["token_content_sha256"]
models={}
for model in cfg.models:
    model_manifest=json.loads((root/"models"/model.name/"manifest.json").read_text())
    assert model_manifest["grounding"] == {"module":"model.layers.7","capture":"input","tensor_index":0}
    assert list(model_manifest["matrices"]) == expected
    assert all(shape == {"d_in":3584,"d_out":18944} for shape in model_manifest["matrices"].values())
    weights=load_file(str(root/"models"/model.name/"weights.safetensors"))
    assert list(weights) == expected and all(torch.isfinite(value).all() for value in weights.values())
    models[model.name]={"r_rms_norm":model_manifest["r_rms_norm"],"matrices":model_manifest["matrices"]}
report={
    "status":"PASS","resolved_config_sha256":resolved_config_sha256(cfg),
    "token_cache_reused_exactly":True,"source_token_root":str((source/"tokens").resolve()),
    "cache_fingerprint":manifest["cache_fingerprint"],"token_content_sha256":manifest["token_content_sha256"],
    "token_rows":rows,"checked_tensor_values":checked,"matrix_names":expected,"models":models,
    "preflight":os.environ["PREFLIGHT"],
}
(root/"cache_integrity.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,sort_keys=True))
PY
du -sh "${CACHE}"
printf 'ASPD_L07_L13_UP_CACHE_COMPLETE\n'

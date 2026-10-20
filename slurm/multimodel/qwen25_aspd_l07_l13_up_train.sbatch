#!/usr/bin/env bash
# Fresh 4096-step C8192/K32 D0/S1 training on the validated causal-region cache.
#SBATCH --job-name=mm-q25-up713-train
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
: "${PREFLIGHT_JOB_ID:?Set PREFLIGHT_JOB_ID to the passed memory-preflight job}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l07_l13_up_1m.yaml
CACHE=out/multimodel/cache/qwen25_extreme_sports_aspd_l07_l13_up_1m
RUN=out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m
PREFLIGHT=out/multimodel/diagnostics/qwen25_aspd_l07_l13_up_1m/preflight_job_${PREFLIGHT_JOB_ID}/memory_preflight.json

printf 'hostname=%s\n' "$(hostname)"; date --iso-8601=seconds
printf 'git_commit=%s\nCUDA_VISIBLE_DEVICES=%s\nconfig=%s\ncache=%s\nrun=%s\npreflight=%s\n' \
  "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${CONFIG}" "${CACHE}" "${RUN}" "${PREFLIGHT}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
test -s "${PREFLIGHT}"; test -s "${CACHE}/cache_integrity.json"
if [[ -e "${RUN}" ]]; then echo "Refusing to overwrite L07--L13 training run" >&2; exit 21; fi
.venv/bin/python -m aspd.multimodel.cli.cache "${CONFIG}" --validate-only
CONFIG="${CONFIG}" CACHE="${CACHE}" PREFLIGHT="${PREFLIGHT}" .venv/bin/python - <<'PY'
import json,os
from aspd.multimodel.config import load_experiment_config
cfg=load_experiment_config(os.environ["CONFIG"])
pre=json.load(open(os.environ["PREFLIGHT"])); cache=json.load(open(os.path.join(os.environ["CACHE"],"cache_integrity.json")))
assert pre["status"] == cache["status"] == "PASS" and cache["token_cache_reused_exactly"] is True
assert cfg.training.resume is None and cfg.training.max_steps == 4096
assert cfg.training.batch_size_sequences == 2 and cfg.training.gradient_accumulation_steps == 1
assert cfg.training.parameter_dtype == "float32" and cfg.training.autocast == "bfloat16"
assert cfg.training.save_every == 2048 and cfg.training.keep_last_checkpoints == 2
assert cfg.objective.dead_after_batches == 2000
print("ASPD_L07_L13_UP_TRAIN_INPUTS_PASS")
PY
.venv/bin/python -m aspd.multimodel.cli.train "${CONFIG}" --device cuda

RUN="${RUN}" CACHE="${CACHE}" PREFLIGHT="${PREFLIGHT}" .venv/bin/python - <<'PY'
import hashlib,json,math,os
from pathlib import Path
import torch
run=Path(os.environ["RUN"]); checkpoints=sorted(run.glob("checkpoint_*.pt"))
assert [p.name for p in checkpoints] == ["checkpoint_00002048.pt","checkpoint_00004096.pt"]
assert (run/"latest_checkpoint.txt").read_text().strip() == checkpoints[-1].name
hashes={}
for path,step in zip(checkpoints,(2048,4096),strict=True):
    state=torch.load(path,map_location="cpu",weights_only=True,mmap=True); assert int(state["step"]) == step
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8*1024*1024): digest.update(chunk)
    hashes[path.name]=digest.hexdigest()
    del state
rows=[json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
validation=[row for row in rows if "validation/loss" in row]
assert [row["step"] for row in validation] == [1024,2048,3072,4096]
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,float))
train=[row for row in rows if "train/loss" in row]
report={
 "status":"PASS","checkpoint_sha256":hashes,"cache_integrity":str(Path(os.environ["CACHE"])/"cache_integrity.json"),
 "preflight":os.environ["PREFLIGHT"],"dead_feature_revival_eligible_after_batch":2000,
 "peak_cuda_memory_gib":max(row["cuda/max_memory_gib"] for row in train),
 "elapsed_seconds_at_last_train_log":train[-1]["elapsed_seconds"],"final_validation":validation[-1],
}
(run/"training_integrity.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,sort_keys=True))
PY
printf 'ASPD_L07_L13_UP_TRAIN_COMPLETE\n'

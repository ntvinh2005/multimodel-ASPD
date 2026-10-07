# HiPerGator B200 setup and commands

The code is prepared for one NVIDIA B200. No model download or experiment is performed during local
repository setup.

## Environment

Keep the repository, environment, active caches, and run outputs on Blue storage. One clean layout
is:

```text
/blue/<group>/<user>/multimodel-ASPD/
├── repo/                 # this Git repository; small and reproducible
├── cache/
│   ├── huggingface/      # model, tokenizer, and streamed-dataset cache
│   ├── uv/               # downloaded Python wheels and metadata
│   └── uv-python/        # uv-managed CPython 3.13 (independent of module Python)
├── work/
│   └── out/              # activation caches, checkpoints, metrics, analyses
└── archive-manifests/    # small checksums/manifests for archived completed runs
```

Create it and keep the config's relative `out/` paths outside the Git checkout through a symlink:

```bash
export HPG_GROUP=YOUR_GROUP
export MM_ASPD_ROOT=/blue/${HPG_GROUP}/${USER}/multimodel-ASPD

mkdir -p \
  "${MM_ASPD_ROOT}/cache/huggingface" \
  "${MM_ASPD_ROOT}/cache/uv" \
  "${MM_ASPD_ROOT}/cache/uv-python" \
  "${MM_ASPD_ROOT}/work/out" \
  "${MM_ASPD_ROOT}/archive-manifests"

git clone https://github.com/ntvinh2005/multimodel-ASPD.git "${MM_ASPD_ROOT}/repo"
cd "${MM_ASPD_ROOT}/repo"
if [[ ! -e out ]]; then ln -s "${MM_ASPD_ROOT}/work/out" out; fi

module load conda                 # HiPerGator provides uv through this module
export UV_CACHE_DIR="${MM_ASPD_ROOT}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${MM_ASPD_ROOT}/cache/uv-python"
export UV_PYTHON=3.13
export HF_HOME="${MM_ASPD_ROOT}/cache/huggingface"
export WANDB_DIR="${MM_ASPD_ROOT}/work/out/wandb"
export WANDB_MODE=offline         # or log in and use online mode
uv python install 3.13
uv sync --frozen --python 3.13
```

Put the exports in a private file outside Git so login and batch shells use identical paths:

```bash
cat > "${MM_ASPD_ROOT}/env.sh" <<EOF
export MM_ASPD_ROOT=${MM_ASPD_ROOT}
export UV_CACHE_DIR=${MM_ASPD_ROOT}/cache/uv
export UV_PYTHON_INSTALL_DIR=${MM_ASPD_ROOT}/cache/uv-python
export UV_PYTHON=3.13
export HF_HOME=${MM_ASPD_ROOT}/cache/huggingface
export WANDB_DIR=${MM_ASPD_ROOT}/work/out/wandb
export WANDB_MODE=offline
EOF
chmod 600 "${MM_ASPD_ROOT}/env.sh"

# Load the module first, then restore the project variables before sbatch.
export MM_ASPD_ROOT=/blue/<group>/<user>/multimodel-ASPD
module load conda
source "${MM_ASPD_ROOT}/env.sh"
```

The repository also pins `3.13` in `.python-version`. `module load conda` supplies the `uv`
executable, but its current module Python may be newer than the project supports. The explicit
`UV_PYTHON` setting keeps login and Slurm shells on the same interpreter. If `.venv` was created
with another Python, rebuild it once:

```bash
cd "${MM_ASPD_ROOT}/repo"
module load conda
source "${MM_ASPD_ROOT}/env.sh"  # source after module: the module may select its own Python
uv python install 3.13
uv sync --frozen --python 3.13
uv run python --version          # expected: Python 3.13.x
```

Do not put active caches in `$HOME` or `/orange`. Use `/orange/<group>/...` only to archive
completed runs that no longer participate in job I/O. The node-local `$TMPDIR` is fast but
ephemeral; the current large-shard cache works directly from Blue. Consider staging a cache into
`$TMPDIR` only after the smoke run demonstrates that Blue I/O is the bottleneck, and always copy
required results back before the job exits.

Budget roughly 300 GiB of Blue space for the entry-grounded cache, Python/Hugging Face caches, and
two retained checkpoints for several experiment arms. Reserve closer to 500 GiB if keeping the
separate exit-grounded cache and many completed arms simultaneously. Check actual allocation usage
with `blue_quota` and directory usage with `du -sh` or `ncdu`.

The checked-in jobs follow the current HiPerGator scheduler names: partition `hpg-b200` and GRES
`gpu:b200:1`. UF documents 180 GB per B200 and 14 CPU cores per GPU. Pass the allocation at
submission time, for example `sbatch --account=<allocation> ...`. See the live
[GPU resource table](https://docs.rc.ufl.edu/resources/gpus/) and
[GPU scheduler guide](https://docs.rc.ufl.edu/scheduler/gpu_access/) before the first run.

Run the config-only check before allocating a GPU:

```bash
export MM_ASPD_ROOT=/blue/<group>/<user>/multimodel-ASPD
cd "${MM_ASPD_ROOT}/repo"
module load conda
source "${MM_ASPD_ROOT}/env.sh"  # restore UV_PYTHON=3.13 after loading the module
uv run python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen3_1_7b/smoke.yaml
```

## Safe execution order

```bash
# 1. Short cache and 20 optimizer steps.
CACHE_JOB=$(sbatch --parsable --account=<allocation> slurm/multimodel/cache_smoke.sbatch)
sbatch --account=<allocation> --dependency=afterok:${CACHE_JOB} \
  slurm/multimodel/train_smoke.sbatch

# 2. Inspect measured VRAM, tokens/s, and cache GiB in metrics.jsonl.

# 3. Build the reusable entry-grounded main cache.
CONFIG=configs/multimodel/qwen3_1_7b/base.yaml \
  sbatch --account=<allocation> slurm/multimodel/cache.sbatch

# 4. Train one arm at a time on the same cache.
CONFIG=configs/multimodel/qwen3_1_7b/d0_s1.yaml \
  sbatch --account=<allocation> slurm/multimodel/train.sbatch

# 5. Analyze a completed checkpoint.
CONFIG=configs/multimodel/qwen3_1_7b/d2_s2.yaml \
CHECKPOINT=out/multimodel/runs/qwen3_1_7b_d2_s2_entry/checkpoint_00010000.pt \
  sbatch --account=<allocation> slurm/multimodel/analyze.sbatch
```

The analysis writes P1/P2 masses, P3 loci, P4 classes, P5 pair metrics, and decoded top activating
contexts. After inspecting those results, run a P6 causal check on selected component indices:

```bash
uv run python -m aspd.multimodel.cli.ablate \
  configs/multimodel/qwen3_1_7b/d2_s2.yaml \
  out/multimodel/runs/qwen3_1_7b_d2_s2_entry/checkpoint_00010000.pt \
  --model-index 1 --components 17 203 --prompt "The capital of France is"
```

## Day 2: D0/S1 debug run

This run isolates the core method: a linear mean-aggregating encoder, unweighted BatchTopK, and no
shared/exclusive partition, shared-first loss, tying, or sequence transformer. It uses 512 latents,
an average L0 budget of 8, 16,384 training tokens, 4,096 validation tokens, and 200 optimizer steps.
Its dead-feature window is shortened from 2,000 to 100 batches so the dead-fraction diagnostic and
AuxK revival path can actually run during this short experiment.

Use this map to trace the six equations through the implementation:

| Step | Mathematical object | Code path |
|---|---|---|
| 1 | aligned `R^(base), R^(ft)` | `PairedActivationCache.iter_batches` in `cache.py` |
| 2 | linear mean encoder and ReLU | `LinearSharedEncoder.forward` in `model.py` |
| 3 | `a -> g^s` | `batch_topk` in `batch_topk.py` |
| 4 | `g = 1[g^s > 0]` | `batch_topk.py`, immediately after support selection |
| 5 | activation reconstruction | `ActivationDecoders.reconstruct` in `model.py` |
| 6 | parameter reconstruction | `RankOneComponents.reconstruct` in `components.py` |

After loading the environment as described above, validate and submit the cache and training jobs as
one dependency chain:

```bash
# The launcher locates ../env.sh from the repository, loads conda, forces Python 3.13,
# validates the expanded config, and submits both jobs.
cd /blue/<group>/<user>/multimodel-ASPD/repo
slurm/multimodel/submit_d0_s1_debug.sh <allocation>
```

The launcher prints both job IDs. Monitor them and their logs with:

```bash
squeue -j <cache-job-id>,<train-job-id>
tail -f slurm/logs/mm-d0s1-cache-<cache-job-id>.out
tail -f slurm/logs/mm-d0s1-train-<train-job-id>.out
```

Training writes its expanded config, provenance, checkpoints at steps 100 and 200, and metrics to:

```text
out/multimodel/runs/qwen3_1_7b_d0_s1_debug/
├── experiment_config.json
├── provenance.json
├── metrics.jsonl
├── checkpoint_00000100.pt
├── checkpoint_00000200.pt
└── latest_checkpoint.txt
```

Inspect the validation rows at steps 25, 50, ..., 200 and the final training rows:

```bash
METRICS=out/multimodel/runs/qwen3_1_7b_d0_s1_debug/metrics.jsonl
grep 'validation/loss' "$METRICS"
tail -n 10 "$METRICS"
```

Generate a training dashboard at any output path. A trailing-window mean makes per-step training
curves easier to read; validation points remain unsmoothed:

```bash
.venv/bin/python -m aspd.multimodel.cli.plot_training \
  out/multimodel/runs/qwen3_1_7b_d0_s1_debug/metrics.jsonl \
  --output out/multimodel/runs/qwen3_1_7b_d0_s1_debug/training_dashboard.png \
  --smooth 10 \
  --target-l0 8 \
  --title "Qwen3-1.7B D0/S1 debug"
```

The output extension selects the format; PNG, PDF, SVG, JPEG, and WebP are supported. The command
also accepts the run directory in place of the explicit `metrics.jsonl` path and can be rerun while
training is in progress.

Before moving to a main experiment, verify that total loss, `act/base`, `act/finetuned`,
`internal/base`, and `internal/finetuned` trend downward; `sparsity/l0` stays near 8; and
`sparsity/dead_fraction` does not immediately approach 1. The per-matrix internal FVU fields show
which of Q, K, V, O, gate, up, or down projection is failing if the aggregate stalls.

The B200 partition currently permits jobs up to 14 days, but these templates use shorter limits so
failed smoke runs return promptly. Adjust memory and time only after reading the measured cache
size, tokens/s, and peak VRAM from the smoke run.

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

## D0/S1 main-size capacity pilot

This is a 100-step systems pilot, not a scientific interpretation run. It keeps the main workload
dimensions exactly: linear D0/S1, `C=8192`, `K=32`, `B=4`, `T=256`, FP32 parameters, BF16 autocast,
and all seven layer-14 matrices. It uses a dedicated 32,768/32,768-token cache so validation runs
exactly 32 batches. Main objective settings, including `dead_after_batches=2000`, are inherited.

Submit the dedicated cache and capacity train jobs from the repository:

```bash
cd /blue/<group>/<user>/multimodel-ASPD/repo
slurm/multimodel/submit_d0_s1_capacity.sh <allocation>
```

The train job records GPU telemetry every two seconds, GNU `time -v` resource counters, checkpoints
at steps 75 and 100, and an automatic capacity report:

```text
out/multimodel/runs/qwen3_1_7b_d0_s1_capacity/
├── metrics.jsonl
├── gpu_telemetry.csv
├── resource_usage.txt
├── capacity_job_meta.txt
├── capacity_report.json
├── checkpoint_00000075.pt
└── checkpoint_00000100.pt
```

Monitor the jobs and telemetry using the IDs printed by the launcher:

```bash
squeue -j <cache-job-id>,<train-job-id>
tail -f slurm/logs/mm-cap-train-<train-job-id>.out
tail -f out/multimodel/runs/qwen3_1_7b_d0_s1_capacity/gpu_telemetry.csv
```

After completion, regenerate or inspect the capacity report and training plot with:

```bash
.venv/bin/python -m aspd.multimodel.cli.capacity_report \
  out/multimodel/runs/qwen3_1_7b_d0_s1_capacity

.venv/bin/python -m aspd.multimodel.cli.plot_training \
  out/multimodel/runs/qwen3_1_7b_d0_s1_capacity \
  --output out/multimodel/runs/qwen3_1_7b_d0_s1_capacity/training_dashboard.png \
  --smooth 5 --target-l0 32 --title "Qwen3-1.7B D0/S1 capacity"

seff <train-job-id>
sacct -j <train-job-id> \
  --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES%50,NodeList%24 -P
```

`capacity_report` takes the median train-step delta over steps 10–49, 55–74, and 80–99. It derives
validation overhead from the 50→51 elapsed-time jump and checkpoint overhead from 75→76, then
estimates the 10k runtime as `10000*t_train + 40*t_validation + 10*t_checkpoint` with a 10% buffer.
It also checks the exact main workload config, finite metrics, L0≈32, zero AuxK/dead fraction,
absence of D2-only fields, checkpoint creation, GPU utilization, and at least 10% VRAM headroom.

## Full-cache D0/S1 1k scientific pilot

This pilot reuses the generic `cache.sbatch` and `train.sbatch`; no experiment-specific Slurm files
are needed. Its config inherits the complete D0/S1 main protocol and changes only the run/cache
artifact paths and `max_steps=1000`. The cache root is new and requires model-cache schema 2, so a
pre-fix cache cannot be silently reused.

Run the local preflight from a clean committed worktree:

```bash
cd /blue/<group>/<user>/multimodel-ASPD/repo
.venv/bin/pytest \
  tests/test_multimodel_cache.py \
  tests/test_multimodel_batch_topk.py \
  tests/test_multimodel_components.py \
  tests/test_multimodel_model.py \
  tests/test_multimodel_config.py

.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen3_1_7b/d0_s1_1k.yaml
```

Submit the generic jobs with the config exported into the Slurm environment:

```bash
module load conda
source ../env.sh
export CONFIG=configs/multimodel/qwen3_1_7b/d0_s1_1k.yaml

CACHE_JOB=$(sbatch --parsable --account=<allocation> slurm/multimodel/cache.sbatch)
TRAIN_JOB=$(sbatch --parsable --account=<allocation> \
  --dependency="afterok:${CACHE_JOB}" slurm/multimodel/train.sbatch)

echo "cache_job=${CACHE_JOB}"
echo "train_job=${TRAIN_JOB}"
squeue -j "${CACHE_JOB},${TRAIN_JOB}"
```

The full cache must contain 32 train shards and 4 validation shards. Both model manifests must show
`schema_version=2`, `r_rms_norm_split="train"`, and `r_rms_norm_tokens=524288`. The cache validator
now enforces config-matched sequence counts, shard counts, normalization token count, and the
minimum schema before training can start.

```bash
CACHE=out/multimodel/cache/qwen3_1_7b_entry_v2
find "$CACHE/tokens/train" -name 'shard_*.pt' | wc -l
find "$CACHE/tokens/validation" -name 'shard_*.pt' | wc -l
grep -E 'schema_version|r_rms_norm_split|r_rms_norm_tokens' \
  "$CACHE/models/base/manifest.json" "$CACHE/models/finetuned/manifest.json"
```

The four validation points are steps 250, 500, 750, and 1000. Each evaluates the same first 32
batches, or 32,768 of the 65,536 cached validation tokens, because the pilot intentionally preserves
the main protocol's `validation_batches=32`. AuxK loss and dead fraction should remain zero because
the run ends before `dead_after_batches=2000`.

After completion, draw the existing training dashboard:

```bash
.venv/bin/python -m aspd.multimodel.cli.plot_training \
  out/multimodel/runs/qwen3_1_7b_d0_s1_1k \
  --output out/multimodel/runs/qwen3_1_7b_d0_s1_1k/training_dashboard.png \
  --smooth 20 --target-l0 32 --title "Qwen3-1.7B D0/S1 full-cache 1k"
```

Treat this as an independent run. Do not resume its step-1000 checkpoint into the 10k experiment:
the current checkpoint does not preserve the within-epoch batch offset, and elapsed-time throughput
also restarts per process. If this pilot passes, start the 10k D0/S1 run fresh on the same v2 cache.

## Full-cache D0/S1 10k main run

The main run reuses the successful pilot's schema-2 cache and the generic `train.sbatch`. Do not
rebuild the cache and do not resume the 1k checkpoint. The dedicated config preserves the D0/S1
protocol, starts fresh at step zero, writes to a new run directory, and retains all ten 1k-spaced
checkpoints for comparisons at steps 1k, 2k, 5k, and 10k.

From a clean committed worktree, run the tests and validate both config and existing cache:

```bash
.venv/bin/pytest \
  tests/test_multimodel_cache.py \
  tests/test_multimodel_batch_topk.py \
  tests/test_multimodel_components.py \
  tests/test_multimodel_model.py \
  tests/test_multimodel_config.py \
  tests/test_multimodel_plotting.py

.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen3_1_7b/d0_s1_main.yaml

.venv/bin/python -m aspd.multimodel.cli.cache \
  configs/multimodel/qwen3_1_7b/d0_s1_main.yaml --validate-only
```

Protect the append-only `metrics.jsonl` from contamination, then submit only training:

```bash
RUN=out/multimodel/runs/qwen3_1_7b_d0_s1_main
test ! -e "$RUN" || { echo "Run directory already exists: $RUN"; exit 1; }

module load conda
source ../env.sh
export CONFIG=configs/multimodel/qwen3_1_7b/d0_s1_main.yaml

TRAIN_JOB=$(sbatch --parsable --account=<allocation> \
  --export=ALL,CONFIG="$CONFIG" slurm/multimodel/train.sbatch)
echo "train_job=${TRAIN_JOB}"
squeue -j "$TRAIN_JOB"
```

At 4 sequences of 256 tokens per step, 10k steps produce 10.24M token exposures, approximately
19.53 passes over the 524,288-token training cache. Each validation event intentionally uses 32
batches, or half of the cached validation set, to remain comparable with the 1k pilot. Feature
health and AuxK become meaningful after `dead_after_batches=2000`.

The existing plotting CLI can write both the four-panel dashboard and the separate per-matrix
validation plot in one invocation. The latter has one panel per model and seven matrix curves per
panel:

```bash
.venv/bin/python -m aspd.multimodel.cli.plot_training \
  out/multimodel/runs/qwen3_1_7b_d0_s1_main \
  --output out/multimodel/runs/qwen3_1_7b_d0_s1_main/training_dashboard.png \
  --matrix-output out/multimodel/runs/qwen3_1_7b_d0_s1_main/validation_internal_by_matrix.png \
  --smooth 50 --target-l0 32 --title "Qwen3-1.7B D0/S1 10k main"
```

The B200 partition currently permits jobs up to 14 days, but these templates use shorter limits so
failed smoke runs return promptly. Adjust memory and time only after reading the measured cache
size, tokens/s, and peak VRAM from the smoke run.

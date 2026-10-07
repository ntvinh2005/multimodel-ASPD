# HiPerGator B200 setup and commands

The code is prepared for one NVIDIA B200. No model download or experiment is performed during local
repository setup.

## Environment

```bash
git clone <this-repository> extended-aspd
cd extended-aspd

module load git
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
source .venv/bin/activate
```

Clone the repository on project or Blue storage: the configured `out/` cache and checkpoints are
relative to the repository and are too large for the home directory. Put the Hugging Face cache on
the same storage tier:

```bash
export HF_HOME=/blue/<group>/<user>/hf
export WANDB_MODE=offline        # or login and use online mode
```

The checked-in jobs follow the current HiPerGator scheduler names: partition `hpg-b200` and GRES
`gpu:b200:1`. UF documents 180 GB per B200 and 14 CPU cores per GPU. Pass the allocation at
submission time, for example `sbatch --account=<allocation> ...`. See the live
[GPU resource table](https://docs.rc.ufl.edu/resources/gpus/) and
[GPU scheduler guide](https://docs.rc.ufl.edu/scheduler/gpu_access/) before the first run.

Run the config-only check before allocating a GPU:

```bash
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

The B200 partition currently permits jobs up to 14 days, but these templates use shorter limits so
failed smoke runs return promptly. Adjust memory and time only after reading the measured cache
size, tokens/s, and peak VRAM from the smoke run.

# Reduced-scale Delta-Crosscoder positive control

This is an artifact-faithful, reduced-scale control for comparing Delta-Crosscoder
representation evidence with same-pair ASPD parameter evidence. It is not a claim of
full reproduction of the paper.

## Current state

- Code/config/Slurm launcher: ready.
- Real run: blocked until the compute environment has Hugging Face authentication and
  gated `lmsys/lmsys-chat-1m` access. The current environment reports `Not logged in`.
- No Delta candidates or Delta↔ASPD correspondence values are fabricated while those
  artifacts are unavailable.

## Run sequence

1. On HiPerGator, authenticate once and verify access:

   ```bash
   module load conda
   source ../env.sh
   export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
   hf auth login
   hf auth whoami
   ```

2. Materialize the three online source artifacts, using the LMSYS-gated revision and
   the pinned Qwen tokenizer. The contrastive source is generated with the existing
   `build_delta_contrastive_cache` command and should use 768 prompts first; increase
   to 1024 only when the measured generation job fits the one-hour budget.

3. Validate manifests, then submit the positive control (one B200, one hour):

   ```bash
   HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" \
     CONFIG=configs/multimodel/delta_crosscoder/r0_positive_control.yaml \
     sbatch slurm/multimodel/delta_positive_control.sbatch
   ```

4. After completion, materialize the read-only dashboard artifact:

   ```bash
   .venv/bin/python -m aspd.multimodel.cli.analyze_delta_crosscoder \
     out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control
   ```

5. Serve it alongside the existing ASPD dashboard. Missing Delta or comparison files
   remain explicit `available: false` responses; the UI never falls back to demo data.

The training configuration fixes `n_features=17920` through the paper expansion factor
for width 3584, `shared_fraction=0.20`, `K_shared=400`, `K_delta=200`,
`lambda_delta=0.005`, bf16, and the 45/45/10 FineWeb/LMSYS/contrastive schedule.

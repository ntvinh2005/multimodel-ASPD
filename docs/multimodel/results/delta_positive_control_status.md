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

## Delta training result (completed)

The reduced-scale run completed as Slurm job `46568022` in 24m18s. The measured
exact-BatchTopK validation values are:

- dictionary: 17,920 features (3,584 shared; 14,336 delta)
- FVE: total 0.99808, base 0.99819, target 0.99795, delta 0.99222
- exact L0: shared 400, delta 200, total 600
- dead features: 895 / 17,920 (5.0%); delta partition 6.1%, shared 0.64%
- throughput: 6,194 token updates/s; wall clock 23.5 minutes

The run therefore satisfies the reconstruction and exact BatchTopK sparsity checks,
but it is not yet a semantic or cross-method result: the contrastive source contains
32 frozen prompts (13,128 train tokens) and is explicitly repeated by the streaming
reader. The generated `top_activation_examples.json` is empty, so no context claim is
made.

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
   to 1024 only when the measured generation job fits the one-hour budget. The dedicated
   launcher is `slurm/multimodel/delta_positive_control_generate.sbatch`.

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

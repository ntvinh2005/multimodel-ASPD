# General model diffing: final reproducibility report

This report records the completed clean BatchTopK Crosscoder baseline and its comparison with the
existing multimodel ASPD run. Candidate ranking was behavior-agnostic. Context decoding and the
tentative semantic labels happened only after the measured ranks were frozen.

## A. Reference baseline

- Method: BatchTopK Crosscoder from Minder et al., *Overcoming Sparsity Artifacts in Crosscoders
  to Interpret Chat-Tuning* (arXiv:2504.02922), reproducing the paper's Gemma 2 layer-13 run.
- Experiment source: `science-of-finetuning/sparsity-artifacts-crosscoders` at commit
  `d67ec6a6175965c470d104d9be9dedf6c95fdffd`.
- Library source: `science-of-finetuning/crosscoder_learning` at commit
  `b2363ed54590074ef4667ca6152094800a65f9aa`.
- Published checkpoint: `science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss`
  at revision `249340a9fdfab9a042281b26d0d29246478a0ba1`.
- Variant: shared linear encoder, one decoder direction per feature and model, JumpReLU threshold
  fitted to BatchTopK `k=100`, AuxK dead-feature loss, and Crosscoder code normalization.
- Relative decoder norm follows paper Equation 3: zero is base-specific, 0.5 is shared, and one is
  fine-tuned-specific. The published CSV uses the reverse orientation, so the importer explicitly
  applies `local_score = 1 - dec_norm_diff` and records both conventions.

The old reference stack is incompatible with this repository's Transformers version. The local
implementation therefore reproduces the compact model/trainer equations while pinning the exact
reference sources instead of installing or vendoring their entire environment.

## B. Repository changes

Baseline implementation and entry points:

- `aspd/multimodel/baselines/crosscoder.py`: isolated Crosscoder model, loss, training,
  checkpointing, published-artifact import, blind ranking, exact-context extraction, and analysis.
- `aspd/multimodel/cli/train_crosscoder.py`: single training CLI.
- `aspd/multimodel/cli/analyze_crosscoder.py`: single local/published analysis CLI.
- `aspd/multimodel/baselines/__init__.py`: baseline package export.

Small generic reuse and validation changes:

- `aspd/multimodel/cache.py`, `analysis.py`, `training.py`, `provenance.py`, `config.py`, and
  `feasibility.py`: reusable raw-activation/cache validation, provenance, and capacity checks.
- `aspd/multimodel/cli/train.py` and `validate.py`: generic feasibility/config validation.
- `slurm/multimodel/{cache,train,analyze,extract_examples,project_vocabulary}.sbatch`: consistent
  environment/path handling already needed by multimodel jobs.

Configurations and execution:

- `configs/multimodel/crosscoder/r0_gemma2_{pilot,full}.yaml`.
- `configs/multimodel/crosscoder/r1_qwen3_{pilot,full}.yaml`.
- `slurm/multimodel/crosscoder.sbatch`: one parameterized, resumable cache/train/analyze launcher.

Documentation and tests:

- `docs/multimodel/crosscoder_baseline.md`, `crosscoder_run_status.md`, and this results directory.
- `tests/test_multimodel_crosscoder.py`, plus focused generic regression assertions in
  `tests/test_multimodel_analysis.py` and `tests/test_multimodel_reproducibility.py`.
- `README.md` links the baseline and final result.

No second dataset/token/cache stack, intervention system, or dashboard was added.

## C. Reused infrastructure

R1 reads the existing `qwen3_1_7b_entry_v2` paired cache in place. It reuses the exact token IDs,
train/validation split, tokenizer fingerprints, raw `R` activation shards, cache manifest and
validation, checkpoint conventions, and streaming exact-token top-context accumulator. The ASPD
run directory was not overwritten. ASPD reporting and P1-P5 artifacts are reused directly for the
comparison.

## D. R0 published Gemma reproduction

The audited configuration is Gemma-2-2B versus Gemma-2-2B-it, layer 13 output, 50/50 FineWeb and
LMSYS chat data, 100M cached tokens, two epochs/97,656 steps, 73,728 features, `k=100`, learning
rate `1e-4`, and 10M-token dead-feature threshold.

The official full-run artifacts report FVE 0.876 total (0.881 base, 0.870 chat), L0 99.48, and
dead fraction 0.12. Importing all 73,728 features with the corrected score orientation yields:

| Class | Count | Fraction |
|---|---:|---:|
| Base-specific (`<=0.1`) | 5 | 0.0068% |
| Shared (`0.4–0.6`) | 62,373 | 84.60% |
| Intermediate | 11,216 | 15.21% |
| Chat-specific (`>=0.9`) | 134 | 0.1817% |

Thirty-seven candidates were selected blindly and 34 have published contexts. The model-specific
tails contain interpretable structure: chat-specific features 58070/35811 emphasize model-turn
or response-onset markers, 50586 emphasizes end-of-turn boundaries, and 70149 fires at model
responses to hazardous requests. Base-specific selected contexts are dominated by punctuation
and newline structure (for example 52003 and 5582), consistent with the paper's warning that
model-exclusive regions can contain dense or formatting-related sparsity artifacts.

The local R0 pilot completed all pipeline stages, but its deliberately tiny FVE/L0/dead-feature
numbers are not scientific results. Full local retraining was not attempted because the minimal
float32 cache would require about 1.9 TB while only about 856 GB was available. Therefore R0 is an
exact pinned **published-artifact reproduction plus local pipeline pilot**, not a claim of local
full-run training or validation.

## E. R1 Qwen general diff

The full local run compares `Qwen/Qwen3-1.7B-Base` with `Qwen/Qwen3-1.7B` at the input to layer 14.
It uses the ASPD cache's identical 50/50 FineWeb/UltraChat token grid: sequence length 256,
524,288 cached training tokens, 65,536 held-out tokens, and seed zero. Crosscoder capacity is
65,536 features with BatchTopK `k=100`; 40 cache replays make 10,000 optimizer steps and 20.48M
token exposures, with `lr=1e-4`, 1,000-step warmup, AuxK after 10M tokens, and seed 42.

Job `46447674` trained successfully in 37:08; analysis job `46447676` took 1:01; the final frozen
tail/context reanalysis `46453900` took 0:59. Measurements are:

| Measurement | Result |
|---|---:|
| Total/base/fine-tuned validation FVE | 0.993766 / 0.993692 / 0.993836 |
| Train BatchTopK / measured validation L0 | 100 / 115.896 |
| Dead validation features | 7 / 65,536 (0.0107%) |
| Relative decoder norm range | 0.1229–0.8462 |
| Strict base-specific / shared / intermediate / FT-specific | 0 / 65,357 / 179 / 0 |

The lack of features beyond the paper's 0.1/0.9 thresholds is the measured result, not a failed
plot or a relabeling opportunity. Analysis consequently preserves strict classes while also
freezing 16 lowest-score, 16 shared-center, and 16 highest-score fired features for post-hoc
inspection. Those graded tails surface user/assistant role transitions, chat turn boundaries,
assistant-response onset, and the `</think>`-to-visible-answer boundary. Shared-center features
often have nearly equal decoders and repeated contexts; mirror/redundant features remain a
limitation.

## F. Crosscoder versus ASPD

The detailed evidence table is in `qwen3_crosscoder_aspd_comparison.md`. Globally, Crosscoder has
strong reconstruction and graded representation asymmetry but no strict-exclusive Qwen feature.
ASPD has 3,570 shared-activation/shared-mechanism components, 45 shared-activation/concentrated-
mechanism components, one concentrated-activation/concentrated-mechanism component, and 4,576
mixed/subset-mass components. Its exact top-k invariant is 2,097,152 fires; 4,054/8,192 components
do not fire on validation.

Both methods independently surface hidden-reasoning and chat-boundary representations. The
best-supported matched ASPD examples (736, 654, and 3131) are mechanism-shared, which disagrees
with a simplistic claim that representation asymmetry implies parameter concentration. ASPD 440
and 156 show much stronger fine-tuned parameter concentration, but their retained contexts are
heterogeneous and currently lack stable semantic interpretations. No one-to-one feature identity
or causal equivalence is claimed.

## G. Commands

From the repository root after sourcing the project environment, the reusable pilot pattern is:

```bash
CONFIG=configs/multimodel/crosscoder/r0_gemma2_pilot.yaml
CACHE_JOB=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG",STAGE=cache \
  slurm/multimodel/crosscoder.sbatch)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$CACHE_JOB" \
  --export=ALL,CONFIG="$CONFIG",STAGE=train slurm/multimodel/crosscoder.sbatch)
sbatch --dependency=afterok:"$TRAIN_JOB" \
  --export=ALL,CONFIG="$CONFIG",STAGE=analyze slurm/multimodel/crosscoder.sbatch
```

R1 reuses the existing cache, so both pilot and full execution use only train then analyze:

```bash
CONFIG=configs/multimodel/crosscoder/r1_qwen3_full.yaml
TRAIN_JOB=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG",STAGE=train \
  --job-name=cc-r1-10k-train slurm/multimodel/crosscoder.sbatch)
sbatch --dependency=afterok:"$TRAIN_JOB" --export=ALL,CONFIG="$CONFIG",STAGE=analyze \
  --job-name=cc-r1-10k-analysis slurm/multimodel/crosscoder.sbatch
```

The published R0 import is:

```bash
.venv/bin/python -m aspd.multimodel.cli.analyze_crosscoder \
  configs/multimodel/crosscoder/r0_gemma2_full.yaml \
  --published-feature-csv /path/to/feature_df.csv \
  --published-examples-db /path/to/examples.db \
  --published-tokenizer unsloth/gemma-2-2b
```

Regression verification is:

```bash
.venv/bin/ruff check aspd/multimodel tests/test_multimodel_*.py
bash -n slurm/multimodel/crosscoder.sbatch
.venv/bin/pytest tests/test_multimodel_*.py
git diff --check
```

## H. Artifacts

- R0 official import: `out/multimodel/baselines/crosscoder/r0_gemma2_full/published_analysis/`.
- R0 local pilot: `out/multimodel/baselines/crosscoder/r0_gemma2_pilot/`.
- R1 pilot: `out/multimodel/baselines/crosscoder/r1_qwen3_pilot/`.
- R1 full checkpoint/metrics: `out/multimodel/baselines/crosscoder/r1_qwen3_full/`.
- R1 Crosscoder analysis: `out/multimodel/baselines/crosscoder/r1_qwen3_full/analysis/`.
- Existing ASPD P1-P5 analysis: `out/multimodel/runs/qwen3_1_7b_d0_s1_main/analysis/`.
- Regenerated ASPD report: `out/multimodel/runs/qwen3_1_7b_d0_s1_main/analysis/report/`.
- Structured post-ranking labels: `qwen3_general_diff_annotations.json`.
- Cross-method evidence table: `qwen3_crosscoder_aspd_comparison.md`.

## I. Claim ledger

Verified measurements are the configs, provenance, job exit states, FVE/L0/dead fractions,
relative-norm distributions, class/taxonomy counts, exact fire counts, and retained token contexts.

Qualitative observations are the tentative descriptions of role, chat-boundary, answer-onset, and
hidden-reasoning-boundary contexts. They are explicitly post-ranking and do not modify measured
ranks.

Hypotheses are that semantically analogous Crosscoder and ASPD boundary features may reflect
related computation, and that ASPD 440/156 may expose complementary fine-tuned mechanisms. These
are not established correspondences.

Not yet tested: activation-vector matching across methods, one-to-one feature identity, causal
ablation, or causal logit effects. Causal validation was intentionally not started because the
strongest semantic matches are parameter-shared or low-support, whereas the concentrated ASPD
mechanisms do not yet have stable semantics.

## J. Git cleanliness

Generated checkpoints, caches, logs, large metrics, and plots remain under ignored `out/` and
`slurm/logs/` paths. Only reusable source, configs, tests, and documentation are tracked. Final
verification and the clean/dirty status are recorded in the completion commit and handoff.

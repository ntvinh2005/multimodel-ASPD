# P6-100M local implementation and researcher handoff

Status: `IMPLEMENTED` / `LOCALLY_TESTED` / `HPC_NOT_RUN`.

This handoff concerns only the completed original-ASPD 100M decomposition. It
does not retrain ASPD, run Mix3 or Delta-Crosscoder, mutate source weights, or
establish a causal mechanism.

## Repository provenance

- Integration branch: `p6-100m-integration`, based on `main` at
  `c2d42b13e6d021d66e390aa8728ded2906e495e3`.
- Reviewed remote P6 commit: `origin/multimodel-aspd` at
  `c22b1e374785a8b869ee67944cb9a9cfa480dac7`.
- Merge base: `10af24ef98229f44ff1aa87d16c624d804e77697`.
- The prompt artifact and relevant P6 contracts were selectively adapted from
  the reviewed remote branch. Unrelated Mix3, Delta-Crosscoder, dataset and
  training changes were not merged or cherry-picked.
- The integration is isolated on this branch for researcher review. Every Slurm
  script requires a clean tree and an exact, explicitly reviewed
  `P6_APPROVED_COMMIT`.

## Verified immutable package

The complete 26-artifact manifest was rehashed successfully.

- Package status: `P6_HANDOFF_READY`.
- Risk-screening status: `VERIFIED`.
- Canonical checkpoint:
  `out/multimodel/runs/qwen25_aspd_original_100m/checkpoint_00024414.rank0.pt`.
- Checkpoint SHA-256:
  `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6`.
- Frozen ASPD config file SHA-256:
  `904a91d2ab2c63cafd5f62c9c012bcfbd418f72de8114fffa8f4d14469626c40`.
- Resolved checkpoint/config fingerprint:
  `0f20b77c142f0ba4f1a05e9bf60af2002c1f41de555c92564b5cf379b744c041`.
- Risk shortlist SHA-256:
  `a463575ab15dc0ec4f1b15476fd6e783172d146304e8cbc4ee20df80a373f9b4`.
- Computational shortlist SHA-256:
  `8f913caf46e6f915cf7593f888fd413616260a0e45a7d14e3f6330f21211e34b`.
- Discovery prompts SHA-256:
  `5591b1696c32d9149b45364e3a7418bf634a7efae4ae79198c5c8e22d5009663`.
- Behavioral scoring contract SHA-256:
  `fedb248bae837a168a087486d0147836a0d124f024adfc7c91852fd6b1fe8f64`.
- P6 integration config SHA-256:
  `771d569b734651cb3ef9603135031049427d8fa990f06299ca7042c74fc23d9f`.
- Rank-0 learned inference threshold: `3.0222008228302`.

The P6 config is separate from the ASPD training config. Its strict Pydantic
schema rejects unknown fields, so P6 additions cannot silently alter the frozen
training serialization. Regression tests pin both the original config file hash
and resolved fingerprint.

## Frozen discovery design

Candidates are not stored as constants in the runner or YAML. The strict loader
selects the first three verified risk-screening rows before any P6 outcomes and
reproduces matched-active controls from verified component metrics:

| risk rank | candidate | matched-active control |
|---:|---:|---:|
| 1 | 1720 | 808 |
| 2 | 1173 | 2215 |
| 3 | 2228 | 742 |

The discovery set contains 12 risk prompts, 12 exactly paired sports-safety
prompts and 12 neutral prompts. Both Base and Target are evaluated with alphas
`[0.0, 0.25, 0.5, 1.0]`, random seeds `[0,1,2,3,4]`, greedy decoding and at most
64 new tokens. The complete design has 218 arms: two clean baselines plus all
candidate/control arms for both models. The six-prompt smoke uses 22 arms.

Confirmation remains disabled. It requires a distinct, researcher-approved,
checksummed prompt artifact; discovery prompts cannot be reused.

## Scientific implementation

- At each autoregressive step, clean Base and Target receive identical CPU-canonical
  live-prefix token IDs through device-local clones.
- The shared gate is computed from clean L14 mid-residual inputs with the rank-0
  learned JumpReLU threshold, detached, and frozen before the patched forward.
  This inference policy is explicitly distinct from training-time BatchTopK.
- The hook subtracts the centered, model-specific rank-one contribution from the
  real `model.layers.14.self_attn.o_proj` output. It does not substitute the ASPD
  reconstruction and never assigns global `b_out` to one component.
- Each branch advances with its own generated token and recomputes its clean gate;
  no future response tokens or another branch's continuation are reused.
- Learned, gate-off, matched-active, orthogonal and five seeded random controls
  share intervention timing. Controls are matched per output position in FP32,
  then accepted only when the measured post-cast live-output difference is within
  the frozen tolerance; invalid or BF16-zero doses are excluded from evidence.
- Records include intended/applied L2 and RMS, per-position norms, gate counts,
  modified positions, BF16 zero-rounding, relative output norm, clean-logit
  differences, hook cleanup, deterministic replay and source-weight immutability.
- Component characterization separates effective rank-one operator differences
  (gauge invariant), affine-center effects, input activity and contribution
  differences. It does not equate them with the actual fine-tuning weight delta.

## Behavioral scoring and reporting

The 100M judge adapter sends only prompt and response text to the pinned judge;
model, component, arm, alpha and control metadata remain in a separate mapping and
are rejoined after inference. It scores risk, generic refusal, useful safe sports
advice, coherence/helpfulness and task relevance.

The report computes within-model paired effects and the descriptive differential:

`(Target_intervened - Target_clean) - (Base_intervened - Base_clean)`.

It also compares Target learned attenuation with gate-off, matched-active,
orthogonal and five random same-L2 controls. Exact model/arm/prompt coverage and
parse failures are audited. Missing arms produce
`INCOMPLETE_BEHAVIORAL_MEASUREMENT`, not a favorable mechanism label.

Evidence labels are exploratory: `OBSERVATIONAL_CANDIDATE`,
`CAUSAL_OUTPUT_EFFECT`, `BEHAVIOR_SELECTIVE_CAUSAL_EFFECT`,
`FT_ASSOCIATED_CANDIDATE`, `NOT_SUPPORTED`, or `INCONCLUSIVE`.

## Local verification

Commands used:

```bash
.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml \
  --static

.venv/bin/python -m aspd.multimodel.cli.audit_p6_100m \
  configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml \
  --full-package-rehash

.venv/bin/pytest -q \
  tests/test_multimodel_p6.py \
  tests/test_multimodel_p6_original_causal.py
```

The strict package audit returned `PASS` with
`all_artifact_hashes_verified=true`. The focused suite passed 40 tests. Changed
Python files passed Ruff, all four CLIs imported and exposed `--help`, and all
three Slurm scripts passed `bash -n`.

No 7B model was loaded, CUDA behavior was not exercised, and no Slurm job was
submitted. Real-model hook timing, BF16 applied norms, peak memory, throughput and
deterministic replay remain smoke acceptance gates.

## Manual execution handoff

First run the CPU-only audit:

```bash
.venv/bin/python -m aspd.multimodel.cli.audit_p6_100m \
  configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml \
  --full-package-rehash
```

After reviewing the exact integration commit, Stage A is the only approved first
GPU action:

```bash
export P6_APPROVED_COMMIT="$(git rev-parse HEAD)"
sbatch slurm/multimodel/qwen25_aspd_p6_100m_smoke.sbatch
```

It requests one B200, 16 CPUs, 192 GB and one hour. Expected outputs are under
`out/multimodel/p6_100m/preflight/job_<JOB_ID>.json` and
`out/multimodel/p6_100m/smoke/job_<JOB_ID>/`; logs are
`slurm/logs/q25-aspd-p6-100m-smoke-<JOB_ID>.{out,err}`.

Only after independent review of a provenance-matched `GPU_PREFLIGHT_GO`:

```bash
export P6_APPROVED_COMMIT="$(git rev-parse HEAD)"
export P6_SMOKE_REPORT="out/multimodel/p6_100m/smoke/job_<SMOKE_JOB_ID>/preflight_report.json"
sbatch slurm/multimodel/qwen25_aspd_p6_100m_discovery.sbatch
```

Discovery requests one B200, 16 CPUs, 192 GB and two hours. It writes
`out/multimodel/p6_100m/discovery/job_<JOB_ID>/generation/` and refuses to run if
the smoke runtime projection exceeds the frozen 5,400-second active budget.

After generation review, blinded scoring and report generation are separate:

```bash
export P6_APPROVED_COMMIT="$(git rev-parse HEAD)"
export P6_DISCOVERY_ROOT="out/multimodel/p6_100m/discovery/job_<DISCOVERY_JOB_ID>"
sbatch slurm/multimodel/qwen25_aspd_p6_100m_judge.sbatch
```

The judge requests one B200, 8 CPUs, 60 GB and one hour. It writes run-specific
`behavioral_scores/job_<JOB_ID>/` and `report/job_<JOB_ID>/` directories.

None of these scripts submits another job.

## Scientific boundary

Residual FVU about 0.0038 does not imply faithful L14 `o_proj` reconstruction
(FVU about 0.50) or parameter-delta recovery (NSE about 110.49). This pilot can
test whether a frozen learned intervention has a model-differential, selective
behavioral effect beyond matched perturbation controls. It cannot by itself prove
that a component reconstructs `W_target-W_base`, establish a general mechanism,
or provide independent confirmation. A negative result is retained without
candidate substitution.

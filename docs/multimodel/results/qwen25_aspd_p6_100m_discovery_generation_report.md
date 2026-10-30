# Qwen2.5 ASPD P6-100M bounded causal discovery: generation report

**Status (2026-10-30):** `JOB_COMPLETED_UNVERIFIED`. The real-model GPU
preflight and the frozen bounded discovery generation both completed. This is a
durable reviewer-facing execution record, not a behavioral or mechanistic
claim. Blinded behavioral judging and independent verification have not run.

## A. Executive summary

- The submitted implementation commit was
  `715603a492de18e3e001dea52c7f012a7df7a704`.
- Stage A smoke job `48377787` completed in 2:45 with
  `GPU_PREFLIGHT_GO`, no technical failures, immutable source weights, clean
  hook cleanup, deterministic replay, and finite outputs.
- Stage B discovery job `48382069` completed in 48:22 with exit code `0:0`.
- The discovery produced exactly 7,848 model-arm-prompt records: 218 frozen
  arms over 36 prompts, split equally between Base and Target.
- All 12 listed generation artifacts match their recorded SHA-256 and byte
  sizes. No NaN or Inf was found in the generation records.
- All positive-dose learned interventions were effective under the recorded
  gate/dose contract. However, 167 of 5,832 positive-dose control records
  failed the stricter per-position, post-dtype norm-matching criterion. These
  comparisons must remain invalid and cannot support a favorable evidence
  class.
- Every `alpha=1.0` control comparison passed. The 167 positive-dose failures
  occur only at `alpha=0.25` or `alpha=0.5`, where BF16 rounding is material for
  small perturbations.

The correct current interpretation is that causal experiment data were
generated successfully, but behavioral scoring is `NOT_VERIFIED` and no
component is yet a causal fine-tuning mechanism.

## B. Frozen scientific provenance

| Item | Frozen value |
|---|---|
| Base | `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee` |
| Target | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d` |
| Grounding | Input of `model.layers.14.post_attention_layernorm` |
| Patched computation | Real output of `model.layers.14.self_attn.o_proj` |
| Canonical rank | Rank 0 |
| Checkpoint SHA-256 | `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6` |
| Frozen ASPD config SHA-256 | `904a91d2ab2c63cafd5f62c9c012bcfbd418f72de8114fffa8f4d14469626c40` |
| Resolved config fingerprint | `0f20b77c142f0ba4f1a05e9bf60af2002c1f41de555c92564b5cf379b744c041` |
| Prompt artifact SHA-256 | `5591b1696c32d9149b45364e3a7418bf634a7efae4ae79198c5c8e22d5009663` |
| Behavioral contract SHA-256 | `cda7946ae338ce3c191b3ad9bcabdc291991fcc39286b7c0e9de4d338049b68c` |
| Learned inference threshold | `3.0222008228302` |

Candidates were frozen from the verified risk-screening shortlist before P6
outcomes:

| Risk rank | Candidate | Matched-active control |
|---:|---:|---:|
| 1 | 1720 | 808 |
| 2 | 1173 | 2215 |
| 3 | 2228 | 742 |

The discovery contains 12 risk prompts, 12 paired sports-safety prompts, and 12
neutral prompts. It measures both models, alphas `[0.0, 0.25, 0.5, 1.0]`, and
random-control seeds `[0, 1, 2, 3, 4]` under greedy decoding with at most 64 new
tokens.

## C. Execution record

| Stage | Slurm job | Resources | Result | Runtime |
|---|---:|---|---|---:|
| Real-model smoke | 48377787 | 1×B200, 16 CPU, 192 GB | `GPU_PREFLIGHT_GO` | 2:45 |
| Bounded discovery generation | 48382069 | 1×B200, 4 CPU, 192 GB | `JOB_COMPLETED_UNVERIFIED` | 48:22 |

The smoke measured peak CUDA allocation/reservation of approximately
39.32/40.01 GB. Its revised full-batch runtime projection was 4,313.90 seconds,
below the frozen 5,400-second active budget. The actual discovery finished well
inside the two-hour Slurm walltime.

No ASPD training, checkpoint mutation, source-model weight update, confirmation
run, behavioral judge, or dependent job was performed by these two jobs.

## D. Artifact and coverage audit

The generation directory is locally authoritative:

`out/multimodel/p6_100m/discovery/job_48382069/generation/`

Its size at audit time was approximately 974 MB. Large runtime artifacts remain
under ignored `out/`; they are not copied into Git.

| Check | Result |
|---|---:|
| Recorded artifacts | 12 |
| Hash/size mismatches | 0 |
| Generation records | 7,848 |
| Behavioral-evaluation input records | 7,848 |
| Unique prompts | 36 |
| Base records | 3,924 |
| Target records | 3,924 |
| Non-finite values | 0 |
| Candidate IDs | 1173, 1720, 2228 |

Review-critical artifact identifiers:

| Artifact | SHA-256 |
|---|---|
| Smoke preflight report | `70a3ef858858f9dbe7f155e4614ea79548c92407cdbe6e078fee8b27ec68656e` |
| Discovery run manifest | `3f3d769c7bcc075f1dd47cd458ec951649d516d05fb5179a3b7735eec9096d0d` |
| Discovery checksum manifest | `c985d0bce46782236d5df141c6e94679272bb712c883fcd5792319a0ed0e9fe7` |
| Control norm audit | `7d190ed9cf7b0a0c8f76c4bc3fe8f90a47ff5fa4f15f00b92928772dba8338c6` |
| Frozen selected candidates | `059996c00f567ad8f67aa3a33dcc88d24630d34a2104a1e80827ac332459d6c2` |
| Component characterization | `d9cbfabf99fa6d4f6f47d366b06f2fc4b1974e28911b9f022058946bc9d0b8be` |

## E. Actual post-BF16 control validity

The control audit compares each control perturbation with a hypothetical learned
perturbation recomputed on the same live autoregressive branch and position,
after dtype conversion. It does not compare aggregate norms across already
diverged continuations.

There are 5,832 positive-dose non-clean records. Of these, 167 (2.86%) are
invalid under the frozen 10% per-position relative norm tolerance:

| Candidate/model/dose | Invalid records | Main affected control types |
|---|---:|---|
| 1720 / Target / 0.25 | 36 | Orthogonal matched |
| 2228 / Base / 0.25 | 71 | Orthogonal and random same-L2 |
| 2228 / Base / 0.5 | 48 | Orthogonal and random same-L2 |
| 1173 / Base or Target / 0.25–0.5 | 12 | Orthogonal and random same-L2 |

All positive-dose learned records are valid, all `alpha=1.0` controls are
valid, and no invalid record has a shape mismatch, a nonzero control at a
zero-reference position, or a missing positive reference. The failing
per-position relative errors range from about 0.1001 to 1.0468; their median is
about 0.2931. This pattern is consistent with small post-BF16 perturbations, but
the records remain invalid rather than being retrospectively accepted.

The additional 1,728 invalid records at `alpha=0.0` are intentional zero-dose
rows with no positive learned reference. They are no-op checks and are not
counted among the 167 positive-dose failures.

## F. Observational component characterization

The gauge-invariant Base/Target effective-operator comparisons generated before
behavioral scoring are:

| Component | Operator cosine | Relative operator difference | Interpretation |
|---:|---:|---:|---|
| 1720 | 0.000432 | 0.7966 | Strong Base/Target operator difference |
| 1173 | 0.835997 | 0.2865 | More similar effective operators |
| 2228 | -0.000682 | 0.7742 | Strong Base/Target operator difference |

These are observational matrix-local characterizations. They do not establish
that a component reconstructs the fine-tuning weight delta, nor do they show a
behavioral causal effect.

## G. Reviewer decision boundary

What is verified now:

- exact checkpoint/config/prompt/candidate provenance;
- real Base and Target loading on B200;
- clean-reference gate and real `o_proj` intervention execution;
- source-weight immutability, hook cleanup, deterministic replay, and finite
  generation;
- complete expected arm/prompt coverage and immutable output checksums;
- explicit detection rather than concealment of invalid post-BF16 controls.

What is not verified:

- behavioral risk reduction, refusal, safe-advice quality, coherence, or task
  relevance;
- a favorable learned-versus-control effect;
- independent confirmation;
- reconstruction of `W_target-W_base` or a general fine-tuning mechanism.

A blinded judge may score the immutable outputs, but downstream reporting must
exclude the 167 invalid positive-dose comparisons. In particular, no low-dose
comparison may be promoted because its aggregate norm looks close when its
per-position post-dtype audit failed. The fully valid `alpha=1.0` comparisons
remain eligible for exploratory scoring under the frozen multiple-testing
contract. Any result remains discovery-only and not independently verified.

If the research goal requires every low-dose control to be valid, control
construction must be improved and a new run must be preregistered; the existing
artifacts must not be overwritten or silently repaired.

## H. Current claim ledger

**Verified:** technical GPU feasibility, frozen provenance, exact discovery
coverage, artifact integrity, and generation execution.

**Generated but unverified:** causal intervention continuations and
component/control comparisons.

**Invalid for causal comparison:** the 167 positive-dose records listed above
that fail per-position post-BF16 norm matching.

**Not established:** a behavior-selective causal effect, a fine-tuning-associated
mechanism, independent confirmation, or parameter-delta fidelity.


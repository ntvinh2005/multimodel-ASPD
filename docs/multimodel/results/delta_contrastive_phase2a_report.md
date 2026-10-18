# Delta-Crosscoder Phase-2A contrastive-cache handoff

This closes only the paper-style contrastive data-pipeline engineering task. The pilot and 75-step
smoke are not scientific results, an organism reproduction, or an ASPD comparison.

## A. Commits

- Starting Phase-1 HEAD: `7e1aba04ac3f93aeb75c0d73f20026286c7b99e1`.
- Implementation and both GPU jobs: `d71a8334d927839384ffe409fb9011fd92b1d352`.
- Final documentation commit is recorded in repository history and the final handoff message.

## B. Files created or modified

- Added `aspd/multimodel/delta_contrastive.py` and the dedicated
  `cli/build_delta_contrastive_cache.py`.
- Extended `baselines/delta_crosscoder.py` only at the activation-source/config boundary.
- Added the frozen prompt JSONL and `qwen3_contrastive_smoke.yaml`.
- Reused and extended the existing `slurm/multimodel/delta_crosscoder.sbatch`; no second launcher.
- Added `tests/test_multimodel_delta_contrastive.py` and updated the Delta reference, run status,
  and root README.
- Existing BatchTopK Crosscoder code/artifacts and matched-cache artifacts were not modified.

## C. Exact contrastive construction

For every prompt `x`, Base greedily generates `y_base` and Target greedily generates `y_target`.
This creates `s_base = x || y_base` and `s_target = x || y_target`. Each complete sequence is then
run separately through both models, caching `[h_base(s), h_target(s)]`. The trainer receives only
`Delta(s) = h_target(s) - h_base(s)` for one fixed `s`. It never computes
`h_target(s_target) - h_base(s_base)`.

## D. Tokenization and canonical sequence

The native Qwen vocabularies are exactly equal at 151,669 token-to-ID entries. Their tokenizer
metadata is not equal: Base EOS is `<|endoftext|>`, Target EOS is `<|im_end|>`, and their raw chat
template hashes differ. Target tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` is therefore canonical. One canonical token tensor is
built per complete sequence and copied unchanged to both model forwards. All 32 examples contain
equal Base/Target measurement hashes; the cache validator also compares tensor token IDs against
the JSONL provenance.

## E. Chat/rendering strategy

The canonical Target Qwen template is applied to one user message with
`add_generation_prompt=true` and `enable_thinking=false`. Generated token IDs are appended directly
to that prefix, avoiding decode/re-tokenize drift. Raw prompt, decoded answer, decoded complete text,
exact IDs, full raw template, template hash, tokenizer fingerprints, EOS/stop reason, and revisions
are retained. Base uses its own EOS stopping metadata while generating; Target uses its own.

## F. Prompt source and count

The repo-fixed file contains 16 general-purpose engineering prompts with IDs `general-0001` through
`general-0016`; SHA-256 is
`ac091647774f136319e0d7a18a6fba2967a03d8168f8a7c143fe9dfec9ad3aab`. It was frozen before the
run and is not behavior- or safety-selected. Seeded prompt-level splitting selected four validation
prompts (`0006`, `0007`, `0008`, `0010`), preventing branch leakage across splits.

## G. Generation settings

Seed 42, greedy decoding (`temperature=0`, `do_sample=false`), recorded `top_p=1.0`, maximum 64 new
tokens, and thinking disabled. Base hit the 64-token limit for 16/16 answers; Target hit it for 14/16
and emitted EOS for 2/16. All 16 prompts produced both branches; no two branch token sequences were
identical.

## H. Hook/layer

Both pinned models use `model.layers.14` **input**, tensor index 0, activation width 2,048:

- Base `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`
- Target `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

## I. Cache schema

The separate artifact is under `out/multimodel/cache/delta_contrastive/qwen3_phase2a_16` with
fingerprint `2b9b6d9e...ef2f1f3`. `manifest.json` stores construction/model/tokenizer/template/generation
provenance and split/shard counts. `examples.jsonl` stores per-example text, IDs, generator side,
split, stop/truncation state, same-input hashes, hook, and shard row. Each `.pt` shard stores
`input_ids[B,T]`, `valid_tokens[B,T]`, `activations[B,T,2,2048]`, and `example_indices[B]`.
Right padding is masked, cached padding activations are zero, and the source excludes it. Explicit
right truncation is capped at 256 and records removed tokens; the pilot had zero truncations.

## J. Verification

- Delta-specific plus contrastive tests: 19 passed.
- Full `tests/test_multimodel*.py`: 82 passed.
- Ruff: passed; Slurm `bash -n`: passed; CLI help/config parse: passed.
- Tests cover same-text hashes, two branches, two measured activations, exact Target-minus-Base
  Delta, the cross-response subtraction regression, provenance, round trip, padding exclusion,
  explicit truncation, and deterministic greedy generation.
- Independent post-job cache validation passed; checkpoint reload passed at step 75.

## K. Cache pilot results

| Field | Result |
|---|---:|
| Job / node / Slurm elapsed | `46518291` / `c0906a-s5` / 1m37s |
| Prompts requested / succeeded | 16 / 16 |
| Base / Target responses | 16 / 16 |
| Complete sequences | 32 |
| Train / validation valid tokens | 2,105 / 714 |
| Total valid activation tokens | 2,819 |
| Mean prompt / answer tokens | 25.125 / 62.969 |
| Truncated sequences | 0 |
| Cache size including report | 23,999,515 bytes (22.89 MiB) |
| Generation / extraction / materializer | 25.79s / 0.41s / 58.24s |
| Peak allocated GPU memory | 6,922,556,416 bytes (6.45 GiB) |
| Mean / median Delta norm, Base-generated | 232.089 / 62.664 |
| Mean / median Delta norm, Target-generated | 227.167 / 50.044 |

The Delta norms are pipeline diagnostics only and are not evidence of learned behavior or method
success.

## L. Tiny Delta contrastive smoke

Job `46518503` completed 75 steps and 31,575 token updates in 30s Slurm / 6.61s trainer wall time,
writing and successfully reloading `checkpoint_00000075.pt`. Configuration: 640 features (128
shared, 512 delta), `K_shared=8`, `K_delta=4`, and `lambda_delta=0.005`.

- Literal step 1 → step 75 reconstruction: `240.756 → 129.361`; Delta loss:
  `2,393,134.75 → 2,523.98`. Step 75 is the 57-token remainder batch, so this literal pair is not a
  like-for-like trend.
- Comparable 512-token batches, first epoch steps 1–4 vs last epoch steps 71–74: mean
  reconstruction `241.700 → 236.520`; mean Delta loss `2,393,057.81 → 2,089,729.16`.
- Final validation Delta FVE: `0.137661`; total FVE: `0.109875`.
- Train L0 was exactly shared `8`, delta `4`, total `12` at every step.
- Final validation dead features: 202/640 (10/128 shared, 192/512 delta). Training dead counters were
  zero because the paper default 10M-token death horizon cannot be reached in this tiny smoke.

These values prove wiring and optimization only; the repeated 2,105-token train cache and lagging
threshold calibration make them scientifically uninterpretable.

## M. Unresolved issues

- No official Delta-Crosscoder code is available to resolve the Phase-1 paper ambiguities.
- The pilot prompt set is a frozen repo-local generic set, not the paper's contrastive corpus.
- Greedy outputs commonly reach the 64-token engineering cap; this is recorded, not hidden.
- Threshold-mode validation L0 exceeds the BatchTopK training budget after 75 steps, consistent with
  the already documented short-run threshold-calibration limitation.
- GitHub HTTPS and SSH credentials were unavailable in this execution session, so the local commits
  could not be pushed automatically.

## N. Recommended next step

Stop here. Before any full run, decide and pre-register the real contrastive prompt corpus/mix and
token budget, benchmark cache storage at the intended scale, and separately resolve the organism
checkpoint/layer question. Do not start R0 or 50,000 steps from this engineering artifact.

## O. Repository state

The final handoff message records the final commit SHA, `git status`, and `git diff --stat`. Generated
cache/training artifacts remain under ignored `out/`; existing matched-cache and ASPD artifacts are
unchanged.

# Qwen2.5 Extreme-Sports ASPD preparation

Status: **PREFLIGHT_RESUBMITTED**

The first GPU preflight, job `46638812`, passed static validation but failed during
PEFT equivalence because the unmerged wrapper added a module-path level. Commit
`c0c094d` resolves canonical paths through `PeftModel.get_base_model()` while still
running the forward through the adapter-enabled wrapper, with a regression test.
Job `46690388` was canceled before it started to reduce its resource request. The
launcher now requests 8 CPUs and 60 GB RAM, matching currently running B200 jobs.
The preflight was resubmitted as job `46691095` at
`2026-10-19T12:56:46-04:00` using
`slurm/multimodel/qwen25_extreme_sports_aspd_preflight.sbatch`. Its initial state
was `PENDING (Priority)` on `hpg-b200`; it requests one B200 GPU, 8 CPUs, 60 GB
RAM, and a one-hour walltime.

Expected runtime for job `46691095` is about 5–15 minutes after it starts if the
pinned models are already available in the shared Hugging Face cache. Allow up to
30–60 minutes if model loading or shared-storage I/O is slow. The one-hour Slurm
walltime is the hard limit; queue wait is additional.

This job writes fresh, job-scoped artifacts (no stale JSON is reused):

```text
out/multimodel/diagnostics/aspd_preflight_46691095/static_config.json
out/multimodel/diagnostics/aspd_preflight_46691095/qwen25_peft_equivalence.json
out/multimodel/diagnostics/aspd_preflight_46691095/qwen25_effective_delta_w.json
```

Slurm logs are written to
`slurm/logs/mm-qwen25-aspd-preflight-46691095.out` and
`slurm/logs/mm-qwen25-aspd-preflight-46691095.err`. ASPD cache generation,
training, analysis, causal intervention, and any later job must not begin until
these preflight outputs have been inspected.

Purpose: prepare ASPD to use the same pinned Qwen2.5 Base/Target pair as the
Delta-Crosscoder work, so a later comparison can be designed fairly.

The new engineering pilot is configured in
`configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml`. It uses 256-token
sequences, 131,072 training tokens, 16,384 validation tokens, 8,192 features and
`top_k=32`. Its cache and run directories are new Qwen2.5-specific paths.

The Target is a pinned PEFT LoRA adapter, not a standalone full checkpoint. ASPD's
loader now loads the declared base at the pinned base revision, attaches the exact
adapter revision, calls `merge_and_unload()`, and only then permits hooks, activation
capture, and matrix snapshots. Cache manifests record `model_kind=peft_merged`, the
adapter/base provenance, and `adapter_merged=true`; cache validation rejects an
adapter-backed manifest that is not recorded as merged.

Reproducibility guards have been tightened: the Qwen2.5 config uses the pinned
canonical tokenizer directly and no longer declares an unpinned
`chat_template_tokenizer` override. It also requires model cache schema v2, so stale
schema-v1 manifests fail closed.

The future preflight utility is
`aspd/multimodel/diagnostics/peft_merge_equivalence.py`. It compares the unmerged
PEFT target with the merged target on the same materialized token IDs, including
layer-14 `R`, all seven `X_j` inputs, and final logits. It reports shape, absolute
and RMS differences, relative RMS error, and cosine similarity. The same utility
also audits effective `Delta_W = W_target - W_base` at every configured matrix,
including Frobenius norms, relative norm, cosine, exact equality, adapter-target
expectation, and engineering warnings.

The diagnostic keeps canonical token IDs on CPU and moves per-model clones to the
actual model device before each forward. Its JSON writer creates parent directories,
and its `equivalence`, `delta`, and `all` modes load only the models needed for that
operation; `all` releases the unmerged target before loading Base for the Delta-W
audit.

The installed PEFT package was inspected without loading a model. Its exposed
classes/source do not provide a statically visible `merge_and_unload` signature, so
the loader does not guess at `safe_merge=True`; it fails closed if the runtime
wrapper cannot provide a callable merge operation. Neither diagnostic has been run
on the real 7B pair yet.

Boundaries:

1. No ASPD GPU training has been run.
2. No ASPD scientific result exists.
3. No Delta-vs-ASPD performance claim exists.
4. The new config is an engineering pilot only, not a matched scientific comparison.
5. The target is the pinned PEFT LoRA adapter described by the Delta preflight config.
6. ASPD snapshots the merged effective target weights, not the underlying base weights.
7. A later scientific comparison should match the Base/Target pair, layer/hook,
   held-out behavioral benchmark, candidate-validation procedure, and preferably
   the Delta 45/45/10 FineWeb/LMSYS/contrastive training distribution.

Validation performed without model loading:

```text
qwen25_extreme_sports_aspd_d0_s1_pilot: schema/static validation passed
focused preparation tests: 6 passed
existing config/cache tests: 10 passed
```

The following preflight commands are executed by the submitted Slurm script; no
cache, training, analysis, causal, or Delta V3 command is included:

```bash
# A. Static config validation
.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml --static

# B. Unmerged-vs-merged equivalence preflight
.venv/bin/python -m aspd.multimodel.diagnostics.peft_merge_equivalence \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --mode equivalence --prompt "Give a concise safety checklist for a training exercise." \
  --device cuda --output out/multimodel/diagnostics/qwen25_peft_equivalence.json

# C. Effective Delta-W audit
.venv/bin/python -m aspd.multimodel.diagnostics.peft_merge_equivalence \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --mode delta --device cuda \
  --output out/multimodel/diagnostics/qwen25_effective_delta_w.json

# D. ASPD pilot cache generation
.venv/bin/python -m aspd.multimodel.cli.cache \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml --device cuda

# E. ASPD pilot training
.venv/bin/python -m aspd.multimodel.cli.train \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml --device cuda

# F. ASPD pilot analysis
.venv/bin/python -m aspd.multimodel.cli.analyze \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/checkpoint_00000200.pt \
  --device cuda \
  --output-dir out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis
```

The static validator command is:

```bash
.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --static
```

The submitted preflight does not build an ASPD cache, train ASPD, run ASPD
analysis, run Delta V3, perform causal intervention, or alter existing
Qwen3/Delta artifacts. It only performs static validation, PEFT merge equivalence,
effective Delta-W auditing, and JSON integrity checks.

Scientific status: ASPD remains GPU-unvalidated on the Qwen2.5 Extreme-Sports pair
while replacement job `46691095` is pending. Exact-token provenance and cache-schema guards are
tightened. No real 7B merge equivalence, ASPD cache correctness, ASPD training
result, behavior-selective component, causal effect, or Delta-vs-ASPD comparison has
been established. Do not start cache generation or training until the three fresh
preflight artifacts above have been inspected.

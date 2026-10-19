# Qwen2.5 Extreme-Sports ASPD preparation

Status: **CODE-READY / GPU-NOT-RUN**

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
focused preparation tests: 4 passed
existing config/cache tests: 10 passed
```

The static validator command is:

```bash
.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --static
```

This preparation does not run Slurm, download Qwen2.5, launch either 7B model, run
Delta V3, perform causal intervention, or alter existing Qwen3/Delta artifacts.

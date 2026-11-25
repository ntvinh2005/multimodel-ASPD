# Negation comprehensive same-anchor benchmark

Slurm job `52790878` completed with exit code `0:0`. It measured one Discovery group, `negation-discovery-multiple-0026`, using the affirmative, negative, counterfactual affirmative and counterfactual negative conditions in Gemma 2, SmolLM2 and Granite 3.3. The archived Discovery BatchTopK support came from the original 2,048-anchor pools. Taylor attribution used the same aligned final query anchor, frozen support, the existing selected matrices, and the original `β` representatives (`K_rep=3`, `τ_rep=0.07`). Each measured target used the existing strict `|E| > 0` top-ten parent rule and matrix-output normalization.

## Measured coverage

| Measurement | Result |
|---|---:|
| Active gate–anchor pairs across the four conditions | 208 (48 / 49 / 57 / 54) |
| Distinct active PRE/MID gates | 77 |
| Gate–anchor–model slots with frozen representatives | 570 / 624 |
| Representative targets by model (Gemma / SmolLM2 / Granite) | 550 / 551 / 590 |
| Representative parent edges | 15,170 |
| Measured representative targets with empty parent sets | 174 |
| Supplemental non-representative targets | 12 (four per model) |
| Supplemental parent edges | 60 |

The 54 gate–anchor–model slots without frozen representatives have an explicit unavailable status. A measured target with an empty parent set remains a valid measurement. The supplement selected the strongest observed non-representative parent by normalized `w` for each condition/model, with stable component-ID tie breaking. Each subsequent request names an exact `anchor=component` previously observed as an upstream parent on that anchor. Supplemental edges are stored separately from the official gate-level parent sets and distributions.

One measured path is `smol.L08.q/3207 → smol.L08.mlp_up/6020 → smol.L09.mlp_gate/1619` in the affirmative example. The two normalized edge weights are 0.030170 and 0.004151. Both use anchor `07ef1c5628698a1af549e6c54b5c2d45bb1f1fefc0e4bb3d1e99cb2afc070e55`. The middle component is supplemental. This path records native Taylor dependencies at the query. Its role in negation remains a question for causal validation.

## Resources and approval boundary

| Resource | Observed benchmark |
|---|---:|
| Slurm wall time on one B200 | 4 min 51 s (0.081 GPU-h) |
| Backend initialization | 255.58 s |
| Representative measurements, including cache insertion | 11.05 s |
| Supplemental measurements, including cache insertion | 0.17 s |
| Peak allocated / reserved GPU memory | 52.41 / 52.43 GiB |
| Peak process RSS | 5.80 GiB |
| Raw group file / SQLite cache / compact UI asset | 10.92 / 1.42 / 4.48 MB |

The four-group plan uses the lower median by representative target count within each Discovery family. The benchmark is the highest-workload group among those four medians. The scope family contributes its four factorial conditions to this plan.

| Discovery group | Active gate–anchor pairs | Representative targets |
|---|---:|---:|
| `attribute-0005` | 198 | 1,580 |
| `binary-0032` | 202 | 1,657 |
| `multiple-0026` (cached) | 208 | 1,691 |
| `scope-0008` | 207 | 1,688 |
| Total | 815 | 6,616 |

With the benchmark group cached, a single job would measure 4,925 additional representative targets and pay model initialization once. The benchmark estimate was **6–8 minutes on one B200 (0.10–0.13 GPU-h), excluding queue time**. The proposed allocation requested 32 GiB host memory. The estimated four-group raw export, shared cache and frontend addon totaled about 67 MB. The estimate covered representative targets and existing supplemental records. Further recursive targets depend on which parents are requested.

## Approved four-group run

Slurm job `52792763` completed with exit code `0:0` in **5 min 11 s on one B200 (0.086 GPU-h)**. It measured all 4,925 new representative targets and reused all 1,691 representative targets and 12 supplemental targets in the benchmark cache. The exported benchmark group is byte-identical to its original artifact.

| Group | Active gate–anchor pairs | Representative targets | Parent edges | Active gate–anchor–model slots without β representatives |
|---|---:|---:|---:|---:|
| `attribute-0005` | 198 | 1,580 | 14,030 | 53 |
| `binary-0032` | 202 | 1,657 | 14,840 | 46 |
| `multiple-0026` | 208 | 1,691 | 15,170 | 54 |
| `scope-0008` | 207 | 1,688 | 15,230 | 44 |
| **Total** | **815** | **6,616** | **59,270** | **197** |

The four groups contain 16 final-query anchors and 126 distinct active PRE/MID gates. Of the 6,616 measured representative targets, 689 have empty parent sets. Across all models, 197 of 2,445 active gate–anchor–model slots lack a frozen representative and are marked unavailable. The 12 non-representative targets and their 60 edges are from `multiple-0026`. Other groups support recursive exploration where the exact upstream component was itself measured as a representative. An unmeasured non-representative parent requires a targeted request.

Backend initialization took 255.52 s. Measuring and caching the three new groups took 32.24 s in total. Peak allocated GPU memory was 52.44 GiB, peak reserved GPU memory was 52.49 GiB and peak process RSS was 5.82 GiB. The raw group exports total 42.56 MB and the shared SQLite cache is 5.51 MB.

The raw four-group run and manifest are in `comprehensive-four-groups-edbeb9f/`. The verified static deployment is in `comprehensive-four-groups-edbeb9f-workbench/`, under the same artifact root below. The expanded addon is 17.44 MB. Its index points the benchmark group to its original GPU job and the three new groups to job `52792763`. The frontend data loader accepted the measured records for each active gate, anchor and model. All 166 base build files matched the benchmark deployment byte for byte. The 60 Python tests, 25 frontend tests and TypeScript check were rerun successfully. In the Workbench, use **Negation → Dependency coverage → Comprehensive groups** to select one of the four groups.

## Verification and access

- A CPU-only replay had zero cache misses, initialized no model backend, and emitted a byte-identical group artifact (SHA-256 `6f356113657317b2cb345f3fd1fb8627ee71b345e085aecf7ca2e87acbf63221`).
- The frontend installer verified the source group hashes, exact example/pool/anchor/model-position joins, natural support, frozen targets, signed edges, normalized weights and supplemental reachability.
- 60 selected Python tests and 25 frontend tests passed. The TypeScript check and production build passed. Browser interaction was not tested. Ruff flagged an unused loop variable and four references to loop variables in a synchronously called closure in the pinned compute runner. The other lint checks passed.
- Use **Negation → Dependency coverage → Comprehensive groups → multiple-0026**. Select a condition and model, then a parent component. **Expand this component’s measured parents** appends exact same-anchor edges when available. The displayed parent distribution stays tied to the selected gate’s frozen representatives.

The immutable artifacts are under `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/negation-evaluations/v1-c1fb958`:

- `comprehensive-benchmark-edbeb9f/` contains the Slurm measurement and manifest.
- `comprehensive-cache-v1/targets.sqlite` contains the exact-anchor cache.
- `comprehensive-benchmark-edbeb9f-cache-replay/` contains the byte-identical CPU replay.
- `comprehensive-benchmark-edbeb9f-workbench/` is the built dashboard.

The compute revision is `edbeb9f12f19609838ea032a9d1a664673a995cc`. Checkpoint step 61,036 has manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`. The Discovery manifest hash is `4edd09f521c4791092fb4dc717ae7cc509afa6fc1a3fa631f46f8603d436e1ca`.

From the Negation worktree, the following installs a completed run in a fresh frontend addon directory:

```bash
ASPD_ROOT=/blue/thai/share/multimodel-ASPD/repo
NEGEVAL="$ASPD_ROOT/.codex_runtime/negation-evaluations/v1-c1fb958"
PYTHONPATH=. "$ASPD_ROOT/.venv/bin/python" multimodel_dashboard/scripts/install_negation_comprehensive.py \
  --workbench multimodel_dashboard/public/negation \
  --run "$NEGEVAL/comprehensive-benchmark-edbeb9f" \
  --output multimodel_dashboard/public/negation/expanded
```

The Slurm template is `slurm/multimodel/aspd_v3_negation_comprehensive.sbatch`. Set `REPO`, `REVISION`, `PREPARATION`, `DISCOVERY`, `OUTPUT` and `CACHE` as job environment variables. Use `--mode benchmark --group negation-discovery-multiple-0026 --benchmark-trace` for the reported benchmark. A follow-up uses `--mode trace --group GROUP --target ANCHOR=COMPONENT`, the same cache, and a fresh output directory. A completed follow-up includes all cached reachable supplemental targets for that group and can be installed in a fresh addon with the other group exports.

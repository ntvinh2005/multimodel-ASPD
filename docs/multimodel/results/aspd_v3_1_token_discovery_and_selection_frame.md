# Token discovery and Selection Frame controls

## Workbench

Research notes autosave in the browser by exact checkpoint and annotation scope. The three scopes are shared gate, model-specific gate and individual component. The editor separates observations, hypotheses and evidence. Researcher status appears in the graph hover text and inspector. Research Notes in the header opens the filterable table and JSON/CSV export. Import JSON restores notes on another browser origin or local tunnel port.

The initial import seeds nine SmolLM2-specific notes once per browser origin and preserves any existing note at the same key. All nine have the researcher status `Hypothesis proposed`. The working Selection Frame hypothesis is visible in the library. The original report used `953` without a site. Across the 16 examples in the four measured Negation groups, `PRE:953` had 171 aligned firing positions including the described words and punctuation, while `MID:953` had none. The import uses `PRE:953` and records this resolution.

**Explore by token** in Induction and Negation loads complete archived activation sequences on demand. Each selectable span ends at an exact shared canonical boundary and has a model-specific native token index. The final aligned token is the default. Select a span to see every active PRE/MID gate at its anchor. Existing measured final-query graphs appear at their original anchors. At the eight fully measured nonfinal anchors, the inspector supports recursive expansion of stored parents and children. Representative-based parent distributions and cross-model comparisons use the original representatives. The additional graph retains parents per individual component under the same top-ten and strict `|E| > 0` parent rule. All edges refer to the selected anchor.

## Benchmark release coverage

The Selection Frame controls consist of eight matched four-condition groups (32 prompts). The comparisons include choice verbs, selection versus removal, source phrases, definite versus indefinite articles, active versus passive constructions, polarity and unrelated natural-language controls. The frozen checkpoint produced natural activation records for all 290 exact aligned positions in one 2,048-anchor pool. The preparation and the activation artifact record exact source revisions, pool membership and aligned anchors. The original datasets were preserved.

| Dataset | Groups | Activation anchors measured / expected | Active gate–anchor pairs | Anchors with complete component graphs | Measured component graphs / expected, all models |
|---|---:|---:|---:|---:|---:|
| Induction Smoke | 32 | 2,258 / 2,258 | 146,373 | 0 | 0 / 12,290,976 |
| Induction Discovery | 256 | 16,784 / 16,784 | 1,096,169 | 0 | 3,132 / 92,060,880 |
| Negation Smoke | 32 | 3,992 / 3,992 | 249,470 | 0 | 0 / 20,975,832 |
| Negation Discovery | 128 | 16,024 / 16,024 | 998,021 | 8 | 49,030 / 83,905,080 |
| Selection Frame controls | 8 | 290 / 290 | 17,015 | 0 | 0 / 1,429,344 |
| **Total** | **456** | **39,348 / 39,348** | **2,507,048** | **8** | **52,162 / 210,662,112** |

Gate–anchor pairs count shared support once across models. The eight complete anchors cover 468 active shared gate–anchor pairs. The per-dataset, per-condition and per-model counts are in the deployment's `tokens/coverage.json`. This file reports incomplete coverage and separates targets with measured empty parent sets from missing targets. Benchmark missing or failed measurements were zero across its eight anchors. Held-out splits were not opened. The browser bundle for Natural Validation lacks exact full sequences and all-position shared support, so it is outside these denominators. Adding Natural Validation requires recovery or extraction of its canonical source anchors.

For a measured example, choose **Explore by token → Negation Discovery → negation-discovery-multiple-0026**. The first `from` in each condition is fully measured. The second measured position is `will` in the affirmative conditions and `not` in the negative conditions. The eight measured anchors have 39,504 component targets, 374,700 retained parent edges and 2,034 measured empty parent sets. This coverage does not establish a causal or universal Selection Frame mechanism.

## Benchmark and scale estimate

Slurm job `52801196` completed `0:0` on one B200 in **8 min 36 s**. Backend initialization took 258.10 s. The two dependency batches took 122.15 and 96.05 s. Peak GPU allocation was **56.75 GiB** (57.01 GiB reserved) and peak process RSS was **5.70 GiB**. The two raw compressed batches total 5.71 MB and their eight compressed browser assets total 8.10 MB.

Extrapolating by component count gives 323 B200 GPU-hours of measurement. Extrapolating by distinct backward calls gives 444 hours because production batches mix more varied prompts with fewer repeated targets. Allow approximately **335–455 B200 GPU-hours** including cold starts when batching about 64 four-document batches per job. These are planning estimates from two batches on one Negation group. The Selection Frame controls alone are estimated at **2.3–4.7 B200 GPU-hours**. The five-dataset projection requires about **74 GB** combined for compressed measurement and browser bundles at the sampled compression ratio. The two local project roots measured about 870.19 GB by allocated blocks before scaling. The repository ceiling is 1 TB, and additional cache and temporary outputs count toward it.

The deterministic shard runner uses at most four distinct documents per batch. It saves each batch with exact anchor IDs, source and configuration hashes, a completion marker and an artifact digest. Completed batches are skipped on resume. The eight complete benchmark anchors can be reused for Negation Discovery via `--reuse-benchmark`. The shard planner was checked against the archived Discovery artifacts for both Induction and Negation and against the new Selection Frame artifact. The approved complete Selection Frame run is reported below.

## Artifacts and verification

All evaluated artifacts are under `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/negation-evaluations/v1-c1fb958/`.

- Preparation in `selection-frame-v1-preparation/`
- Benchmark, run manifest and Selection Frame activation records in `token-dependencies-benchmark-b30ee0d/`
- Built Workbench and machine-readable coverage in `token-discovery-v1-compressed-workbench/`

The benchmark ran compute revision `b30ee0d0f114aeb42ec520c1a742d45d15fdf1f6` and checkpoint manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`. The frontend deployment verified all token asset hashes and retained the prior Induction and Negation graph bundles byte for byte. The 63 selected Python tests, 35 frontend tests and production build passed. Browser interaction was not tested in a live session.

## Complete Selection Frame run

After approval for up to five B200 GPU-hours, array `52805330` ran the pinned compute revision `9141c6d8e3d1202eb32882dd8e3f068e2748bc1b`. The half-open batch ranges were `[0, 32)`, `[32, 53)` and `[53, 73)`, balanced by the number of distinct backward requests. Each task requested one B200, eight CPU cores and 16 GiB of host memory. Each task had a 100-minute wall limit.

| Slurm task | Elapsed | Exit code |
|---|---:|---:|
| `52805330_0` | 1 h 12 min 18 s | `0:0` |
| `52805330_1` | 59 min 39 s | `0:0` |
| `52805330_2` | 58 min 23 s | `0:0` |
| **Total B200 use** | **3 h 10 min 20 s (3.17 GPU-hours)** | |

| Selection Frame measurement | Verified |
|---|---:|
| Matched groups / prompts | 8 / 32 |
| Completed batches | 73 / 73 |
| Aligned anchors with complete graphs | 290 / 290 |
| Active gate–anchor pairs with complete graphs | 17,015 / 17,015 |
| Component dependency targets on Gemma / SmolLM2 / Granite | 476,448 / 476,448 / 476,448 |
| Component dependency targets across the three models | 1,429,344 / 1,429,344 |
| Stored signed parent edges | 13,528,080 |
| Measured empty parent sets | 76,536 |
| Representative targets used in the separate official summaries | 140,503 |
| Missing component dependency targets | 0 |

The frozen activation archive supplied exact canonical positions and natural BatchTopK support. The run measured every naturally active component in the selected matrices at each of those positions. Validation checked each batch hash, exact target identities, native model indices, within-model compute order, finite signed effects, matrix-output scales and the approved maximum of ten parents. The displayed child links invert the stored parent edges at the same anchor. The complete graph coverage supports manual comparisons and leaves the preliminary hypotheses at `Hypothesis proposed`.

The 73 compressed raw batches total 206.76 MB. A CPU-only replay verified all 73 batches from cache with no backend initialization. The 290 compressed Workbench graph assets total 297.74 MB. Open **Negation or Induction → Explore by token → Selection Frame controls** in the new Workbench. For a matched example, choose `selection_vs_removal_by_polarity` and select `from` in both “Choose the book from the shelf.” and “Take the book from the shelf.” The verified assets each cover all three models, with 69 and 78 active shared gates at those respective positions.

Under the artifact root above, `selection-frame-full-9141c6d-v1/` contains the raw batches, their completion markers, `verification.json` with coverage per group, condition and model, and a `COMPLETE` record for the verified deployment. `selection-frame-full-9141c6d-v1-workbench/` contains the static Workbench and `tokens/coverage.json`. Deployment validation checked the hash of all 290 new graph assets and verified that the prior Induction and Negation assets were retained.

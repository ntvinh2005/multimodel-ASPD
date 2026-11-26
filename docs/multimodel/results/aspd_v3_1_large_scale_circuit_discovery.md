# CPU discovery of larger ASPD circuits

## Scope and resource allocation

This study reuses archived v3.1 measurements for Gemma 2, SmolLM2 and Granite. All graphs preserve each source model, exact canonical anchor, known activation pool, layer, matrix and rank-one identity. Signed Taylor effects and normalized parent weights use the existing same-anchor formulation. No checkpoint tensors, gates, source weights, parent rules or representative definitions were changed.

The live `thai` QoS reported 48 CPUs and 375 GiB of RAM. At the initial and pre-scale snapshots, another researcher had a two-CPU job requesting 15 GiB. The project used 873.41 GB by allocated blocks across its shared and personal roots against the 1 TB ceiling. The four-CPU pilot requested 12 GiB; the full run requested 12 CPUs and 20 GiB. The two jobs were sequential. There was no GPU allocation or model inference.

The first pilot, `52954603`, stopped on an archived inactive gate placeholder. The importer was corrected to recognize a recorded zero only when the natural support lacks that gate. Positive support must still match the artifact. Job `52955533` completed on the corrected importer. The full run was submitted as `52955706`.

## Pilot benchmark

The pilot included a deterministic sample of Natural Validation anchors plus existing circuit witnesses, and measured groups from Selection Frame, Negation and Induction. It indexed 3,863,460 signed edges, 366,350 target scalar records and 970 anchor–model scopes in 30.70 seconds. The index occupied 67.2 MB. The indexing process recorded 1.77 GiB peak RSS.

| Workers | Search wall time | Aggregate process CPU | Scopes/s |
|---:|---:|---:|---:|
| 1 | 35.21 s | 33.57 s | 26.78 |
| 2 | 17.59 s | 32.64 s | 53.62 |
| 4 | 10.07 s | 33.91 s | 93.67 |

The four-worker run used 84.2% of four CPU slots during search, with roughly 394 MiB peak RSS per worker. It processed 943 discovery scopes and produced 14,279 bottom-up, 536 top-down and 35 composition proposals before exact deduplication. The three directions used 26.52 s, 1.22 s and 0.45 s of search CPU respectively. Workers share one memory-mapped index and take deterministic batches dynamically. Each batch builds its anchor-local adjacency once for the three directions, so CPU use follows measured workload across directions. The pilot's exact deduplication produced 13,834 proposals and 100 screened candidates. The full run uses a wider screen of 10 candidates per source, model and direction.

## Sources and methods

The Natural Validation archive contributes the retained same-anchor parent rows, with source hashes verified against `COMPLETE`. The archive lacks document text, document identity and activation-pool assignments for those parent rows. These fields remain unavailable, and this source contributes structural discovery only. The independently archived natural firing examples remain labeled as separate observations. Three newly measured Natural test anchors are included as exploratory observations. The full Induction, Negation and Selection Frame browser indices provide verified coordinates, gate activations and the available active component coefficients and normalized contributions. Coverage is complete for Selection Frame's 290 exported anchors and partial for many Induction and Negation scopes. Absence of a recorded parent in a partial scope contributes no control observation.

The read-only index packs component identities into integers and stores signed edges and scalar records in single memory-mapped files. Hashes, checkpoint, tokenizer and model revisions are pinned in the manifest. Each exact scope has an indexed slice. Controlled examples have immutable group splits. Where a family contains multiple measured groups, the split holds out groups within that family. Selection Frame has one measured group per template family, so entire templates are held out. This is a resplit of existing measurement artifacts. Upstream data collection was exploratory and the holdout is internal to this search.

The three searches are:

- **Bottom-up.** Archived starting components and fan-in or fan-out nodes seed connected beam growth at 5, 8, 12 and 15 nodes. An induced subgraph contains only measured edges from the same graph. Two hops are required. Each proposed extension competes by connectivity, depth and strength.
- **Top-down.** Matched annotated positions provide signed changes in the internal component coefficient and changes in normalized contribution strength. Where complete coverage confirms a gate is inactive, its gated coefficient and contribution are zero. A contrast skips unavailable targets from partial graphs. Candidate sinks are selected from discovery groups and expanded recursively upstream. Output-logit change is unnecessary.
- **Composition.** The five checked-in saved circuits and exact snapshots for the existing Natural candidates supply elementary templates. Both templates must occur in the same exact graph. The search accepts overlap, a directed bridge of up to two edges or a shared downstream target. Additional researcher circuit exports can be supplied with another `--circuits` argument.

The proposal ledger stores exact component edges and merges duplicates across directions. Inverted postings are built only for shortlisted edges; their intersections give the complete exact recurrence within the search index, including the held-out grouping. Group and prefix recurrence are distinct. The prefix audit checks whether a held-out anchor presents a causal text prefix already seen at a discovery anchor. Natural Validation rows with unavailable text are excluded from this audit.

Each ranked candidate has structural dimensions, recurrence, witness strengths and signs, novelty relative to existing gate motifs, and a separate matched-null outcome. The null samples substitute active vertices from the same native layer and matrix, log-degree bin and observed gate-prevalence bin. With insufficient matches the empirical null remains unavailable. Its witness-based hit rate is exploratory. The Pareto ranking keeps size and depth visible while rewarding recurrence, novelty, evidence at multiple exact anchors and separation from available null controls. It retains the fixed shortlist as a screening stage. A directed isomorphism test distinguishes the recurrence of a gate-labeled topology across models from unlabeled structural similarity. These summaries carry no causal or functional equivalence claim.

## Reproduction

Use the pinned CPU checkout and new output paths. The Slurm wrapper checks its source revision and CPU count, and defaults to no GPU request. Each completed search shard writes atomically and can be reused with the identical index and plan.

```bash
export MODE=pilot
export REPO=/path/to/pinned-checkout
export REVISION=<full-computation-commit>
export TOKENS=/path/to/workbench/tokens
export NATURAL=/path/to/natural-validation-posthoc
export NATURAL_TEST=/path/to/natural-test-graph-directory
export NATURAL_PILOT=/path/to/pilot.json
export OUTPUT=/path/to/new-output
sbatch slurm/multimodel/aspd_v3_circuit_discovery.sbatch

# Full run after the benchmark and resource recheck.
MODE=full OUTPUT=/path/to/new-full-output sbatch --cpus-per-task=12 --mem=20G \
  --time=00:35:00 slurm/multimodel/aspd_v3_circuit_discovery.sbatch

python -m aspd.multimodel.cli.audit_discovered_circuits \
  --run /path/to/new-full-output --output /path/to/new-publication \
  --review docs/multimodel/results/aspd_v3_1_circuit_discovery_hypotheses.json
```

The Workbench reads `circuit-discovery/catalog.json` from the static artifact root in **Natural Validation → Discovered circuit candidates**. Filters use discovery direction, size, recurrence, model evidence and hypothesis text. Each opened graph is one exact archived witness. The existing inspector, circuit draft, composition tools and notes use the same identities and local storage.


## Full run and evidence

Job `52955706` completed with exit code `0:0` in 5 min 20 s at source revision `b4ee10d30ae2aa2f5e760c009e14d907fc1b6fc1`. Search indexed 38,970,368 edges and 1,495,450 available component scalar rows across 87,625 anchor–model scopes. The index build took 133.75 s. Search used 12 workers for 99.93 s and 1,031.27 aggregate CPU seconds, or 86.0% of the allocated search slots. Maximum batch RSS was 6.98 GiB from Slurm accounting. The full run occupied 942.56 MB, including the shared index and 1,810 atomic proposal shards.

| Direction | Generated proposals | Expansion states or composition pairs | Catalog memberships |
|---|---:|---:|---:|
| Bottom-up | 709,379 | 26,195,640 | 127 |
| Top-down | 1,890 | 45,852 | 60 |
| Composition | 161 | 258 | 30 |

The 711,430 proposals deduplicated to 684,919 exact component-edge sets. The stratified screen kept 199 distinct candidates with 5–15 nodes. Eighteen were returned by two algorithms using the same measurements. The prefix audit found 101 candidates in held-out groups, of which 71 recurred on at least one previously unseen causal prefix and 30 used only repeated discovery prefixes. Thirteen candidates with at least eight components recurred on an unseen held-out prefix. This audit uses the already sealed discovery ranking and does not re-select the screened candidates.

The initial vertex-substitution null was feasible for 3 of 199 candidates. An additional observational control kept the measured naturally active vertices fixed and swapped directed edges within native matrix pairs. This preserves per-node degrees in each pair, all firing identities, layer position and matrix type. A nontrivial swap was possible for 135 candidates. Of these, 30 exact structures were absent in 64 constrained null snapshots and 105 occurred at least once. These outcomes describe topology under this null at the preserved witness, conditional on archival coverage and candidate screening. They do not quantify an out-of-sample error rate. The catalog records the witness ID, accepted swaps, trials and hits. It contains eligible same-role context denominators for 109 candidates. A partial graph contributes an absence only when all required targets were measured and the corresponding gates were naturally active.

### Cases for manual inspection

| Case | Measured evidence | Interpretation and limitation |
|---|---|---|
| `CXD-9760e17982a8c61a` | Gemma, eight nodes and eight edges, 22 exact anchors and four unseen held-out prefixes. An L14 MID:6020 route reached L15 Q/K. L15 Q and local L16 Gate/Up components were parents of a MID:6020 Down target. The constrained null reproduced the exact layout in 18/64 samples at the audited witness. | A candidate for determiner-triggered integration at the final choice query. Both affirmative and negative conditions and both answer swaps contained the held-out path. Scope-specific function remains open. |
| `CXD-1d7322926a89a591` and `CXD-d6c066817f4bbfe3` | Five-node, seven-edge gate-labeled topology in Granite L16–L17 and SmolLM2 L13–L14 at three exact discovery anchors. Q and Up feed O and Down targets. Granite had one unseen held-out prefix and the SmolLM2 variant had zero. | This is a cross-model structural match. The same template can supply the matched computation or a shared formatting cue. The held-out witnesses do not establish cross-model function. |
| `CXD-cfaacda2b24c56e3` | SmolLM2, six nodes and six edges, four Natural Validation anchors. The O/7352 route connects a saved demonstrative circuit to the Q/2379 route of NVX-034. Constrained null hit rate 0/64 at the audited anchor. | The bridge motivates independent tests of demonstrative number and discourse reference. Exact anchor text is unavailable in this archive. |
| `CXD-650bfa991afe88c6` | SmolLM2, five nodes with two diamonds. Two Down targets feed Q/2152, recurring at 27 exact anchors and seven unseen held-out prefixes. Constrained null hit rate 1/64. | Active and passive carrying examples provide counterevidence for a selection-specific reading. The terminal punctuation and clause-closure hypotheses survive. |
| `CXD-a3e04e475d3f184f` | Granite, 215 exact anchors and 35 unseen held-out prefixes. In Induction held-out groups, the circuit occurred in 2/3 eligible induction, 2/3 no-match and 2/3 counterfactual anchors. It also recurs in Selection Frame. | A specificity control. The strongest raw recurrence can be explained by a common routing pattern across several tasks. Known gate overlap with workbook row X01 was kept out of the new sheet. |

The full catalog contains 65 candidates with an unlabeled topology match across models. Only the two paired records above share a gate-labeled topology at an exact anchor. The hand-audited hypotheses and registered alternatives are in the accompanying review JSON. Structural input, relay and downstream-readout roles are assigned from measured adjacency. Their semantic roles remain hypotheses.

## Delivery

- Complete CPU run including method rankings, exact graph evidence and resumable shards at `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/large-aspd-discovery-full-v1/`.
- Publication catalog with prefix audit, matched context denominators, witness-specific graph nulls and curated hypotheses at `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/large-aspd-discovery-publication-v3/catalog.json`.
- Static Workbench bundle at `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/large-aspd-discovery-workbench-v3/`. Serve this build from a CPU allocation and open the discovery panel under Natural Validation.
- The existing workbook has six deduplicated review rows in **Complex circuit discovery**, including the two model variants in a single row. Cell values, wrapping and status validation were verified through the native Sheets API. The original candidate range was identical before and after the update.


The TypeScript check, 74 frontend tests, production build and browserless DOM interaction check passed. The latter loaded the real Gemma eight-node candidate, opened its component computation inspector and existing notes, followed the cross-model peer and returned to the established Natural Validation candidate view. The local Python checks use `pytest -q tests/test_multimodel_discovery_search.py tests/test_multimodel_posthoc_edges.py`.

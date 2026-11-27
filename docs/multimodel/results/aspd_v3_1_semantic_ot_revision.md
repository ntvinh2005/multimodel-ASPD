# Semantic OT revision on Natural Validation

## Status

The updated collector, scoring pipeline, release verifier, and Workbench are implemented. The frozen checkpoint and post-hoc reference files were verified. The CPU archive diagnostic below has run. The fresh ASPD inference, held-out candidate results, and new live catalog depend on the GPU pilot. The existing catalog remains served during this period.

## What the original results show

The old experiment selected 132 components from the archived Taylor measurement envelope. Only 5 of its 82 grouped pairs had scoreable held-out OT, with the earlier minimum of three contexts per endpoint. Three pairs had scoreable matched controls. This revision requires at least 10 independent measured nonzero held-out contexts per endpoint and a mean of at least 20 across retained components.

A CPU diagnostic reused 20 unique causal prefixes from the old catalog. Original prefix pooling reproduced the archived OT distances for all three selected pairs. The original catalog hash is `d7b3840c2e23e1bac6df9e51bbd0c248dde0617e0d63cb32e56d3ca1b70158d0`. The diagnostic took 13.63 seconds and 0.87 GiB peak host RSS with one CPU thread, including pinned MiniLM loading and embedding. The six endpoints and all distances use archived coefficients.

| Archived pair | Held-out observations, left/right | Distinct documents, left/right | Documents shared | Original prefix OT | Full prefix OT | Local OT | Anchor OT | Combined OT | Without terminal unit OT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOT-010 | 3/3 | 1/1 | 1 | 0.0254 | 0.0030 | 0.0491 | 0.0234 | 0.0362 | 0.0476 |
| SOT-016 | 3/8 | 2/2 | 0 | 0.9220 | 0.9141 | 0.8909 | 0.8882 | 0.8955 | 0.9056 |
| SOT-022 | 6/6 | 1/1 | 1 | 0.0095 | 0.0071 | 0.0219 | 0.0042 | 0.0131 | 0.0235 |

These three comparisons show why a low distance can survive an embedding change when both components draw their mass from the same document. SOT-016 differs in the opposite direction. The revised embeddings reduce its distance slightly, and the available evidence still comes from only two documents per component. All six archived endpoints fall short of the revised document minimum.

## Fresh measurement contract

- The frozen source models are Gemma 2 2B IT, SmolLM2 1.7B Instruct, and Granite 3.3 2B Instruct. The checkpoint is step 61,036, manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`. The backend verifies the original configuration, model and tokenizer revisions, matrix inventory, 8,192 features per site, and natural BatchTopK (K=32) on complete 2,048-anchor pools.
- The fixed cohort is 1,747 rank-one components across 197 gates. It contains 586 Gemma, 581 SmolLM2, and 580 Granite components. The strata draw 24 gates from each PRE/MID reference-frequency quartile plus five named reference gates. Representative components and matrix scales are read from sealed Natural Validation importance and representative files. The reference gates provide continuity with prior inspections. Their functional labels remain research hypotheses.
- Collection starts at zero-based pool 75, after the original importance and edge pools. The deterministic cumulative budgets are 4, 64, 128, and 192 pools. The first document is excluded from analysis because it may straddle the archive boundary. Each pool retains its original complete composition when computing support. Expansion decisions use coverage and resources before held-out OT scoring.
- For component (c) in matrix (j), the collector captures its native matrix input and records (e_{ci}=g_{ci}(v_c^	op x_{ji})). The exported magnitude is (m_{ci}=|e_{ci}|\lVert u_c\rVert_2/\sigma_j). The fixed scale cancels within a component's OT mass normalization. Signed coefficients, positive and negative magnitude totals, and actual normalized magnitudes remain available to inspect. Factor sign is checkpoint-specific.
- Archive zeros come from measured rows, including inactive gates. A missing or corrupt shard fails verification. The exporter verifies native token ID, position, and end offset against the original aligned record. Unaligned portions of the prefix appear unshaded and marked unavailable. Anchor outlines and heatmap shading carry separate meanings.

## Comparison design

The open-ended search uses the existing balanced OT solver with cosine ground cost in the common, hash-verified MiniLM space. It compares mean-pooled 128-token causal suffixes from the original method, full prefixes up to the encoder’s 512-token limit, 40-token local pooling, the last non-special contextualized wordpiece, and an equally weighted local-plus-anchor concatenation. The last wordpiece can differ from the source-model tokenizer's token. It lies at the end of the same causal text prefix. Terminal-removal sensitivity re-embeds the local prefix after dropping the terminal text unit.

Document hashing assigns discovery and held-out membership. Per component, one nonzero anchor per document is retained, using the strongest observed example after deduplication. Four-unit shingles on the final 64 text units screen near-duplicate prefixes. The held-out support also excludes near-duplicates of discovery examples. The browser exposes all retained examples. OT samples up to 24 documents per component and split with a fixed anchor hash, independent of semantic scores. Counts include all retained observed examples.

The control for each direction is a set of measured-zero contributions from the opposite component. Each zero is from a separate document when available, matched by source and native token ID and position first. The fallback tiers use a shared terminal text unit and position, terminal unit alone, or source and position. Prefix length breaks ties. This tests whether the candidate is more specific than measured-zero contexts with similar surface properties. A measured zero does not assign a negative semantic label. Difficult lexical controls are flagged by the matching tier. A blinded test of known functional matches and externally verified semantic negatives remains needed.

Discovery uses up to four centroid neighbors per target model and ranks by advantage over the closer of the two matched references. The shortlist gives each model pairing up to eight priority slots and fills any remaining slots by rank. At most 48 pairs are frozen before scoring held-out documents. Retention requires a positive lower bound on the exploratory 95% document-bootstrap interval, more than half of transport mass linking distinct anchors, less than half linking the same document, and positive advantage after removing the terminal text unit. These intervals are descriptive and do not correct for multiple comparisons. Compare the same shortlisted pairs against centroid distance and anchor-overlap baselines. The UI also exposes the exact-anchor Jaccard across all eligible examples rather than only the OT subsample. Group triangles require three actual passing pair comparisons.

## Artifacts and release

Each measured pool is sealed with hashes and can be resumed only under the same extraction contract. The coverage report records component counts, corpus sources, and original-to-deduplicated support. Scoring emits a frozen discovery list, all evaluated pairs, held-out method summaries, and separately sealed component example files. The catalog states that exact-anchor Taylor graphs and circuit links are unavailable for the fresh anchors. Older graph and circuit files are kept independently for reproducibility.

The installer checks the catalog hash, per-endpoint coverage, independent document counts, split lineage, signed and normalized token measurements, and component artifact hashes. It installs a versioned bundle and updates `semantic-discovery-release.json` atomically. Frontend requests for the release pointer bypass stale browser caches. All example files are scoped by run identity. A release can only be activated after successful verification.

## Commands and resource bounds

Run from the isolated checkout. The Slurm template requests one B200, eight CPUs, and 32 GiB RAM for a 20-minute initial pilot. It checks the pinned Git revision. The collector rejects later waves if pilot projections exceed six GPU-hours or 3 GiB for compressed measurements. Peak GPU and CPU memory and per-pool time are logged by the collector. These are planned limits, pending a GPU benchmark. At 192 pools, the main coefficient matrix before compression is about 2.56 GiB. Read-only CPU analysis temporarily holds its original blocks and concatenated representation. Actual analysis memory and compressed storage will be measured.

```bash
# Invoke inside the authorized Slurm allocation.
python -m aspd.multimodel.cli.collect_aspd_semantic_contributions \
  --config configs/multimodel/posthoc/v31_500m_validation_v1.yaml \
  --reference "$REFERENCE" --output "$MEASUREMENTS" --pools 4

# Check document coverage before deciding on the next predefined wave.
python -m aspd.multimodel.cli.analyze_aspd_semantic_contributions \
  --measurements "$MEASUREMENTS" --output "$COVERAGE_OUTPUT" --coverage-only

# Score once the measurement budget is frozen.
python -m aspd.multimodel.cli.analyze_aspd_semantic_contributions \
  --measurements "$MEASUREMENTS" --snapshot "$MINILM_SNAPSHOT" \
  --output "$ANALYSIS_OUTPUT" --threads 2

# Verify first. Activate only in the intended release tree after verification.
python -m aspd.multimodel.cli.install_aspd_semantic_discovery "$ANALYSIS_OUTPUT/frontend"
```

No fresh candidate or graph metrics are available as of the CPU diagnostic. The original study remains available in its immutable archive.

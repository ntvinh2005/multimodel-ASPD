# Natural Validation circuit discovery and Semantic OT

This experiment uses the frozen ASPD v3.1 checkpoint at step 61,036 on Gemma 2, SmolLM2 and Granite. Its sole input dataset is Natural Validation, the document-hash validation split of `monology/pile-uncopyrighted` at revision `3be90335b66f24456a5d6659d9c8d208c0357119`. Its new document split is internal to the pilot. All attribution comes from existing same-anchor measurements. The run uses CPU tokenization and embedding only.

## Formulation and recovery

The inspected repository history, Multimodel ASPD formulation and post-hoc formulation map specify component coefficients, contribution norms, Taylor parent edges and Jaccard/TV for parent sets. No previous Semantic OT definition was located there. The original mass and graph definitions are reused. OT is a separate comparison of distributions of observed contexts.

For each measured component-context row, the nonnegative mass is

$$
m_{ci} = \frac{|e_{ci}|\,\lVert u_c\rVert_2}{\sigma_j}, \qquad
p_{ci} = \frac{m_{ci}}{\sum_k m_{ck}}.
$$

Here `e` is the saved gated coefficient, the downstream write norm is from the edge selection artifact, and `sigma` is the original matrix output scale from the importance artifact. The signed coefficient, total normalized magnitude and mass on either sign are retained. Factor orientation is specific to the checkpoint. A sign comparison does not infer matching write directions across models.

Only saved downstream coefficient rows enter the support. The archive is conditioned on representative selection and sampled firings. A missing row remains unavailable. Exact anchor overlap compares probability distributions over *observed* nonzero supports. It is not an estimate of activation similarity over unmeasured anchors. The sampling envelope requires at least one measured coefficient in every model and uses a common recovered anchor inventory.

The raw edge shards store anchor IDs and coefficients but not text. The recovery command streams the beginning of the original pinned Pile shard, reuses the original validation split and three pinned tokenizers, and runs the original aligned-pool builder without source model inference. All **28,524/28,524** measured anchor IDs were recovered in pools 50 through 74 from 53,037 input rows. Identity is checked using the document hash, corpus group, canonical character boundary and prefix hash. Only the prefix through that boundary is saved. Per-model token positions and token IDs are retained. No right-side text enters embedding or display.

The earlier H5 embeddings represent rendered firing contexts from a separate extraction. They cannot be attached to the same anchors without independently verifying prefix identity and component measurements. The pilot uses the shared CPU encoder [all-MiniLM-L6-v2 at revision 1110a243](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41). Mean pooling excludes padding and vectors are L2-normalized. Long prefixes use the final 128 encoder tokens because the measured position is at the end. The mask experiment removes the trailing word-like or punctuation unit from that same prefix. This is a lexical sensitivity check, using text segmentation rather than a model-specific token identity.

The ground cost is `C[i,j] = 1 - cosine(z[i], z[j])`. We solve the balanced transport linear program with sparse marginal constraints and an exact HiGHS solver for shortlisted pairs. Prefiltering uses centroid neighbors per model and includes same-gate pairs in the sampled component cohort. As a consequence, untested pairs cannot qualify for a cluster. The clustering step requires every within-cluster pair to be tested and below the discovery cutoff. The cutoff is the 10th percentile of distances to prequalified control components on discovery. Controls are chosen from other gates within the target model and a bounded support count ratio, using similarities in archived token identity, position bucket, punctuation and corpus-source histograms. Layer and matrix break ties. This is approximate matching rather than exact control on every covariate. Exact-anchor overlap, centroid cosine and suffix-removed OT are recorded alongside full-prefix OT.

Discovery and held-out membership are set by a fixed document hash with seed 61036. Held-out prefixes already present in discovery are excluded. Cluster construction, the threshold and the control selection use discovery. Held-out pairs require at least three observed positive-mass contexts on each side. The tests are exploratory, especially when control components share contexts across pairs.

## Natural Validation structural searches

The archive originally supplied 63 candidates whose source was Natural Validation. The exported Natural archive retains 186 exact Natural Validation witnesses with restored prefixes, and its recurrence counts are restricted to this dataset. The mixed original catalog is still opt-in.

The separate structural rerun evaluates 426 model-position pairs. Bottom-up uses the existing connected growth and composition uses the existing saved-circuit matching. Natural text contains no preassigned affirmative/negative or induction/no-match conditions. For its top-down seed selection, the rerun instead compares the same component's measured coefficient across independent documents with the same source, native token ID and token-position bucket. These are observational magnitude contrasts. Upstream search then uses the existing measured graph and growth rule.

| Method | Proposals before merging | Shortlisted candidates |
|---|---:|---:|
| Bottom-up | 2,727 | 15 |
| Top-down | 105 | 15 |
| Composition | 3 | 3 |

The rerun yielded 33 candidates. None recurred as the same component-edge set within the sampled held-out positions. The 63 original archived candidates have broader graph coverage and separate counts. For example, one Granite composition reuses the archived period motif at the exact recovered prefix ending “Single MDa ions can be stored for several dozen milliseconds.” This verifies a period in that instance. It leaves abbreviation, punctuation and clause-final controls open.

## Semantic results

The initial pilot measured 36 components and shortlisted 347 OT comparisons from 647 distinct prefixes. It produced 9 complete-link clusters, 6 spanning models. Only 3 of 18 grouped pairs had enough held-out contexts. The single scoreable held-out pair with a matched control did not beat that control. These figures motivated a broader cohort and the suffix-removal check. The expanded run keeps separate draws from the archive-eligible components and an enrichment for endpoints of the previous Natural Validation candidates. Drawing candidates by eligibility and hash keeps discovery semantic scores out of sample selection.

The expanded run uses 36 hash-selected and eight prior graph endpoints per model, for **132 components**. The common measurement envelope contains 20,597 discovery and 5,869 held-out anchors after prefix and coverage filters. This is the comparable envelope, not the support size of an individual component. The pilot reuses all 647 cached embeddings. The expanded run covers 2,129 unique exact prefixes and computes 3,611 additional vectors including the suffix-removed view. It scores 1,357 discovery pairs and returns **28 clusters**, 20 spanning models. Two clusters share at least two exact components with a candidate in the original Natural Validation catalog.

| Matched-control comparison | Discovery | Held-out |
|---|---:|---:|
| Grouped pairs with enough contexts | 82 / 82 | 5 / 82 |
| Pairs with a scoreable matched control | 81 | 3 |
| Lower full-prefix OT than control | 74 / 81 | 2 / 3 |
| Lower mean-embedding distance than control | 74 / 81 | 2 / 3 |
| Lower suffix-removed OT than control | 74 / 81 | 2 / 3 |
| Higher exact-anchor mass overlap than control | 19 / 81 | 1 / 3 |

Cluster membership is selected by discovery OT, so discovery performance is expected to favor similar comparisons. The sparse held-out comparisons provide **insufficient evidence that Semantic OT improves upon pooled embeddings**, or that a low OT distance implies a shared computation.

### Cases from the measured catalog

- **SOT-022, Granite.** L21 Down/90 and L23 Gate/90 have held-out OT 0.0095 against a control at 0.889. Their exact-anchor mass overlap of 0.954 is already very high. The components share MID:90, and a measured directed edge connects them at 11 discovery positions and six held-out positions. They share two components with CXD-3ff399470516b394. The match can largely be explained by shared anchor support. One held-out USPTO prefix concerns customer satisfaction and ends with “customer contacts a service.”
- **SOT-010, Gemma and Granite.** The two MLP Gate components for MID:4856 have held-out OT 0.0254 and exact-anchor overlap 0.909. Held-out comparison with the selected control was unavailable. The high shared support explains much of this low distance. It gives no independent test of a common internal operation.
- **SOT-016, Gemma and Granite.** This different-gate pair had discovery similarity but the held-out transport pairs included a StackExchange question about weak topology and a GitHub prefix importing React Navigation. Held-out OT 0.922 is higher than the matched control's 0.785. Removing the trailing unit leaves that failure in place. This is a concrete example of a discovery match failing to generalize.
- **Links between different groups.** The exact same-anchor scan checked 78 discovery and nine held-out model-positions with at least two eligible grouped components. It found four directed component pairs, two joining different groups. For instance, SmolLM2 L08 Q/7237 connects by a saved two-edge path to L12 V/2046 at three discovery positions and zero held-out positions in this scan. A path does not identify which information passes between those groups.

### CPU benchmark

The group QOS reports 48 CPU and 375 GB RAM. The 50% limit is 24 CPU across user activity. The bounded processes each used one CPU thread. Expansion ran after the first pilot. Recovery and experiments used zero GPUs.

| Stage | Elapsed | Peak RSS |
|---|---:|---:|
| Exact archive replay | 6m 29s | 0.69 GiB |
| 36-component pilot | 54s | 1.08 GiB |
| 132-component expanded pilot | 2m 30s | 1.24 GiB |
| Natural structural rerun | 21s | 1.16 GiB |

The recovered prefix data and replay ledger occupy 51 MiB. The expanded scored catalog, embeddings and CSV occupy about 8.3 MiB. Recovery was dominated by streaming and SQLite on shared storage. All-pairs OT across the full eligible component population was not run. A larger experiment needs a coverage-first design and a separately reported comparison that holds source, template and terminal token fixed. The results above support review of individual groupings and graphs, with mechanism labels remaining open.

## Workbench and reproduction

Choose **Natural Validation → Circuit Discovery** for the dataset-restricted rerun and archive. The mixed historical catalog is selectable in that panel. Choose **Natural Validation → Semantic Discovery** to inspect paired prefixes, less similar transport matches, coefficient mass by sign, held-out comparisons and native same-anchor graphs. A displayed semantic pair is based on its context distributions, including pairs with different gates. The graphs are only drawn from records at one recovered model and anchor.

The run artifacts are in `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/semantic-ot-natural-validation-v1/`. `recovery/COMPLETE.json` pins the corpus bytes, exact anchor inventory and tokenizer files. `pilot/COMPLETE.json` and `expanded/COMPLETE.json` pin scored results, embedding vectors and the review table. The recovered corpus text and tensors remain on HiPerGator. Profiles and checkpoint were read-only.

Use a new output root, the frozen validation archive, the shared tokenizer cache and a single CPU thread for the pilot. The stream input is the first 134,217,728 bytes of the pinned `train/00.jsonl.zst`. Use the listed MiniLM revision and pack `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and `model.safetensors` in one directory.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m aspd.multimodel.cli.recover_aspd_natural_validation \
  --archive "$ARCHIVE" --pile "$PILE_PREFIX" --tokenizer-cache "$HF_HUB" \
  --training-config configs/multimodel/aspd_v3/main500m_preparation_v1.yaml \
  --output "$OUT/recovery"
python -m aspd.multimodel.cli.discover_aspd_semantic_ot \
  --archive "$ARCHIVE" --recovery "$OUT/recovery" --snapshot "$MINILM" \
  --graph-index "$GRAPH_INDEX" --structural-catalog "$ORIGINAL_CATALOG" \
  --components-per-model 12 --output "$OUT/pilot"
python -m aspd.multimodel.cli.discover_aspd_semantic_ot \
  --archive "$ARCHIVE" --recovery "$OUT/recovery" --snapshot "$MINILM" \
  --graph-index "$GRAPH_INDEX" --structural-catalog "$ORIGINAL_CATALOG" \
  --components-per-model 36 --structural-seeds-per-model 8 --suffix-check \
  --embedding-cache "$OUT/pilot" --output "$OUT/expanded"
python -m aspd.multimodel.cli.discover_aspd_natural_circuits \
  --archive "$ARCHIVE" --recovery "$OUT/recovery" --graph-index "$GRAPH_INDEX" \
  --circuits multimodel_dashboard/src/workbench/data/bottomUpCircuits.json \
  --evidence multimodel_dashboard/src/workbench/data/expandedNaturalValidationGraphs.json \
  --output "$OUT/structural-pilot"
python -m aspd.multimodel.cli.export_aspd_natural_candidates \
  --catalog "$ORIGINAL_CATALOG" --recovery "$OUT/recovery" \
  --output "$OUT/natural-archive/archive.json"
python -m aspd.multimodel.cli.link_aspd_semantic_circuits \
  --semantic "$OUT/expanded" --graph-index "$GRAPH_INDEX" \
  --output "$OUT/circuit-links.json"
```

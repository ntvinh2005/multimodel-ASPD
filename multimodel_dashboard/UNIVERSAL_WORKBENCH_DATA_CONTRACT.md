# Universal Mechanism Discovery Workbench data contract

The workbench consumes post-hoc results without loading training models. The default dataset is
`aspd-v3.1-500m-measured-workbench-v1`, a bounded measured export from the completed ASPD v3.1 500M
post-hoc validation artifact. Its manifest sets `synthetic: false`; the older deterministic fixture
remains source-only and is not imported by the application.

Every dataset includes a stable dataset ID, schema version, synthetic flag, exact source commit,
training branch/config, screening settings, edge scoring method, and normalization method. The
measured export records the post-hoc source revision, checkpoint step, immutable COMPLETE-chain
hashes, selection rule, browser graph bound, source artifact root, and explicit limitations.

TypeScript interfaces live in `src/workbench/types.ts`. They cover the dataset manifest, models,
shared gates, model-specific rank-one components, aligned anchors with model-specific tokenization,
anchor-specific directed edges, parent distributions, natural gate activation examples, and research
hypotheses.

Nullable numeric fields mean unavailable measurements, never numeric zero. A parent distribution
with zero total strength has status `undefined` and `values: null`; TV and Jaccard remain undefined.
The graph renderer receives only a bounded subset produced by `componentsForSelection` and
`edgesForAnchor`.

`src/workbench/measuredDataset.ts` validates the generated JSON into `WorkbenchDataset`, preserves
checkpoint and anchor provenance, rejects synthetic or cross-model edges, and fails closed on broken
references or non-finite measured values.

The export is regenerated with:

```bash
python3 multimodel_dashboard/scripts/export_measured_workbench.py \
  --artifact-root /blue/thai/share/multimodel-ASPD/evaluations/minhdatle/posthoc/v31_500m_validation_v1_20261122T223718 \
  --activation-context-root /blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1/extraction/full/job_51194062 \
  --output multimodel_dashboard/src/workbench/data/measuredWorkbenchDataset.json
```

The default bound selects the two lowest feature-index eligible gates per PRE/MID site and the first
stored seeded-random natural anchor for each. Parent sets retain the frozen run rule `K_par=10`,
`tau=0`; graph rendering retains edges from the top six parent gates by mean parent-distribution mass
across models. Full parent distributions for the selected scenarios remain in the bundle. This is a
deterministic visualization subset, not an estimate for the full dictionary.

Canonical context text and continuous `g^s` were not retained in the immutable edge artifact. The
Examples panel therefore uses a separate, hash-verified natural-firing extraction from checkpoint
step 61,036 on the document-disjoint test split. It exports up to four existing contexts for every
gate present in the bounded graph, including the saved activation/firing window and measured center
`g^s`. These examples are not the contribution-graph anchors and do not by themselves validate a
semantic interpretation. The manifest preserves both source hashes and this relationship explicitly.
`signedEffect` is measured first-order Taylor `E`; `upstreamCoefficient` and `gradient` remain null
because the artifact does not store them separately.

The scientific specification is `docs/Universal_Mechanism.pdf`, SHA-256
`a963c3616a498ca1b299fad6f033a94a2ca27a7ac8029de6dc5c7e67ebf31f27`. The PDF defines aggregated
parent scores as the mean over representative downstream components; generated and imported adapters
must preserve that factor so total normalized edge strength remains on the specified scale.

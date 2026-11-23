# Universal Mechanism Discovery Workbench data contract

The workbench consumes post-hoc results without loading training models. The initial fixture is
deterministic synthetic data and is never presented as a pretrained-model measurement.

Every dataset includes a stable dataset ID, schema version, synthetic flag, exact source commit,
training branch/config, screening settings, edge scoring method, and normalization method. The
fixture is synchronized to revision `5ca3b43326d77d4d46fa8605394c5e726f5b48b9`, branch
`aspd-v3.1-main500m-checkpoint-timeout-fix`, and
`configs/multimodel/aspd_v3/main500m_preparation_v1.yaml`.

TypeScript interfaces live in `src/workbench/types.ts`. They cover the dataset manifest, models,
shared gates, model-specific rank-one components, aligned anchors with model-specific tokenization,
anchor-specific directed edges, parent distributions, and research hypotheses.

Nullable numeric fields mean unavailable measurements, never numeric zero. A parent distribution
with zero total strength has status `undefined` and `values: null`; TV and Jaccard remain undefined.
The graph renderer receives only a bounded subset produced by `componentsForSelection` and
`edgesForAnchor`.

Future JSON/CSV adapters must validate external records into `WorkbenchDataset`, preserve checkpoint
and anchor provenance, avoid inferring component matches from location, and never create dependency
edges across models.

The scientific specification is `docs/Universal_Mechanism.pdf`, SHA-256
`a963c3616a498ca1b299fad6f033a94a2ca27a7ac8029de6dc5c7e67ebf31f27`. The PDF defines aggregated
parent scores as the mean over representative downstream components; fixtures and imported adapters
must preserve that factor so total normalized edge strength remains on the specified scale.

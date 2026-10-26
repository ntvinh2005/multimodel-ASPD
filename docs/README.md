# Documentation policy

Durable project documentation belongs under `docs/` and is committed with the code it describes.
This includes runbooks, experiment status, final reports, decision records, and compact numerical
summaries needed for review.

Large generated artifacts remain outside Git under `out/` (for example checkpoints, full metrics
tables, caches, and plots). A durable report in `docs/` must summarize the important results, state
the interpretation boundary, and point to the authoritative artifact path and checksums when
relevant.

Multi-model reports use these locations:

- `docs/multimodel/results/`: completed experiment and analysis reports.
- `docs/multimodel/eval/`: versioned evaluation manifests and prompt sets.
- `docs/multimodel/`: methods, runbooks, handoffs, and live run-status documents.

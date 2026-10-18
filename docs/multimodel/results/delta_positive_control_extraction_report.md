# Delta positive-control extraction report

Status: **BORDERLINE / insufficient behavioral evidence**. This phase did not retrain
the completed checkpoint and did not build an ASPD correspondence.

## Audit and checkpoint

- Code commit used for training: `6c38648d564b84cbd57d9c91e98af7149f140`.
- Extraction implementation commit: `54377bc`.
- Checkpoint: `checkpoint_00002048.pt`, Delta training job `46568022`.
- Orientation is now literal: `R_base = ||d_base||/(||d_base||+||d_target||)`;
  low values are target-heavy and high values are base-heavy. The previous analysis
  helper had the tail names inverted; candidate ranking and adapter ranking were fixed.

## Measured extraction

Extraction job `46572404` completed in 1m36s on a B200. It replayed 20 validation
batches (81,920 exact token positions) through the frozen Base/Target pair and trained
encoder checkpoint. Candidate groups:

| group | count |
|---|---:|
| target-heavy Delta | 40 |
| base-heavy Delta control | 20 |
| balanced Delta control | 20 |
| shared control | 20 |
| total unique features | 100 |

98/100 candidates have at least one saved context. Contexts retain exact token IDs,
token strings, center position, measured `z[t,c]`, source group, model revisions and
layer-14 input hook metadata. The top-20 rule is sequence-diverse: at most one context
per source sequence per feature, sorted by measured center activation.

Top target-heavy examples are not restricted to the 32 contrastive training prompts:
for example feature 7887 (`R_base=0.3851`) fired 9 FineWeb, 395 LMSYS and 103
contrastive validation positions; feature 7224 (`R_base=0.3866`) fired 12 FineWeb,
1,345 LMSYS and 206 contrastive positions. This supports a `supported` held-out status
in the current source split, but it is not behavioral evidence.

Manual inspection of the strongest saved contexts currently points mostly to LMSYS
chat-boundary/template and generic assistant-response patterns (for example
`<|im_start|>user`, Alibaba Cloud system text, arithmetic corrections and multilingual
answer formatting), not a stable extreme-sports concept. This is a useful control result:
the checkpoint has measurable, held-out firing, but the present top features should not
be labeled as fine-tuning-relevant risk features.

## Interpretation boundary

The current validation artifact has FineWeb, LMSYS and repeated 32-prompt contrastive
groups only. It does **not** yet contain frozen risk/extreme-sports versus neutral
behavioral prompts, so risk-fire rate, neutral-fire rate, enrichment and effect-size
fields remain unavailable. No causal or target-behavior claim is made.

The correct decision is **BORDERLINE**: measured contexts generalize beyond the tiny
contrastive cache into LMSYS/FineWeb, but the required held-out behavioral check is
still missing. The next phase should freeze behavioral prompts/responses and run the
same exact-token extraction before considering causal validation or same-pair ASPD.

## Artifacts

```text
out/multimodel/baselines/delta_crosscoder/
  qwen25_extreme_sports_positive_control/analysis/
    feature_metrics.json
    source_activation_metrics.json
    top_activation_examples.json
    candidate_summary.json
```

The dashboard now labels `R_base` orientation explicitly and renders measured contexts
for selected Delta features. No large activation tensors are committed.

## Tests

```text
24 passed (Delta, streaming, semantic-triage suites)
Ruff: passed
Python compileall: passed
```

TypeScript/Vite build still requires a Node executable in the compute environment;
the source changes are present but the production bundle was not rebuilt here.

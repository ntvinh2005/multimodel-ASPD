# Natural Validation token pilot for NVX-025

Open **Natural Validation**, choose **NVX-025**, and open **Examples**. Click a token to see its exact test anchor and naturally selected PRE and MID gates. The component selector keeps the same exact model, layer, matrix and gate while moving between tokens or examples. Use **Token color** to switch between shared activation, ungated reading, signed coefficient and contribution magnitude. Each metric uses one scale across the pilot examples for the chosen component or gate. Teal and purple denote positive and negative signs. Hover for the raw value and available native token positions. Hatched spans indicate unavailable measurements.

Use **Return to saved validation graph** to inspect the measured **Core motif** or **Same-gate extension**. Those saved graphs have a different source split and anchor. On a test token, the graph area reports availability for that token. A compatible future asset can report a full or partial graph, or an empty measured edge set. Cross-model comparisons show exact previously recorded endpoints sharing the selected gate and matrix, with one reading and contribution per available model. The notes editor uses the same checkpoint and component keys as the existing Workbench.

## Archive coverage

| Level | Evidence recovered in this pilot | Evidence still required |
|---|---|---|
| P0 · text, positions, `gˢ`, binary `g` | 10 held-out windows, 513 of 536 displayed native positions mapped to 489 distinct exact shared anchors across eight archived test pools. All 489 have verified Gemma, SmolLM2 and Granite token positions. | 23 displayed native positions lack a verified shared anchor in the extraction. |
| P0 · `vᵀx`, `e`, `‖eu‖₂`, normalized magnitude | Archived target coefficients and static factor statistics at validation witnesses. At an archived test token with `g=0`, `e` and contribution magnitude are zero by definition. | Test-anchor native forward capture, including ungated scores at both active and inactive tokens. |
| P1 · signed `E` and normalized `w` | Two separate NVX-025 validation graph witnesses contain six and nine displayed component edges. The test manifest contains 100,352 anchors and neither witness occurs in that inventory. | Exact-anchor same-token dependency measurement for the 489 held-out pilot anchors. |
| P2 · native computation | Archived Up → Down Taylor-based affine predictions and static parameters for the witness case study. | Exact native replay under fixed gates for the recovered test contexts. |
| Controlled calendar contrasts | Six prepared texts with a verified shared target boundary in all three tokenizers, covering day-first dates, month-first dates, month changes, month-only uses and lowercase verbal use. | New natural BatchTopK evaluation pools for these texts. The prepared examples have no activation or graph measurements. |

The exporter starts from the immutable quick Intruder test archive, verifies its completion marker and SHA-256 hashes, and recovers candidates for the native Gemma prefix by overlapping saved native token windows starting at BOS. It then pins all three fast tokenizers and reruns the repository’s exact canonical alignment. It retains a window position only when the canonical anchor digest matches the archived anchor and the native window matches its recovered placement. A position outside the retained set cannot inherit measurements from another anchor. Continuous scores come directly from the archived 2,048-anchor BatchTopK pools. Binary support is the presence of a strictly positive archived score. The replay command independently checks the requested native token positions, original sparse support and frozen checkpoint.

### Observed pilot examples

| Held-out text at the selected token | MID:1117 `gˢ` | MID:7417 `gˢ` | PRE:1529 `gˢ` |
|---|---:|---:|---:|
| `date: '31 [March] 2006.'` | 0.17369 | 0.42534 | 0.37818 |
| `Hillard Bell Huntington (21 [December] 1910` | 0.13334 | 0.41988 | 0.38567 |
| `Day: [August] 1, 2017` | 0.31119 | 0 | 0 |
| `History\nIn [August] 2015` | 0.28941 | 0 | 0 |

The ranking of these held-out windows was conditioned on firing. Counts across the 489 distinct retained anchors are 12 for MID:1117, eight for MID:7417 and nine for PRE:1529. These observations motivate testing date order and lexical class with the prepared contrasts. Component trajectories, unconditional frequencies and equivalence of computation across models remain open.

The exact MLP Up → Down endpoints from the separate validation witness are `gemma.L15.mlp_up/7417 → gemma.L15.mlp_down/7417`, `smol.L08.mlp_up/7417 → smol.L08.mlp_down/7417` and `granite.L20.mlp_up/7417 → granite.L20.mlp_down/7417`. The [existing component case study](aspd_v3_1_component_computation_inspector.md) reports the corresponding affine predictions and all exact endpoint identities.

## Resource estimate and next measurement

The CPU archive export took 23.37 seconds and 772 MiB peak RSS. The browser pilot bundle uses about 1.02 MB (decimal). The hash-bound native replay input and controlled contrast manifests are about 58 kB and 2.1 kB.

A single native forward per model and document batch can gather one selected component’s reading, signed coefficient and contribution for every aligned token in a window. The proposed first forward replay uses the 60 verified positions in the 31 March window, with the paired Up and Down endpoints in each model. The planning allocation is 5–15 minutes per source model including cold load, 12–16 GiB of GPU memory, 16 GiB of host RAM and under 2 MB of scalar output for the three-model pilot. Preflight verified all 60 positions, the frozen natural support and the two exact Up and Down endpoints for every model. The forward pass and exact local MLP computation remain unexecuted on the archived test texts. The archived binary support is fixed during replay. Causal prefix inputs end at the last requested native token, and attention remains native to each source model.

Exhaustive dependencies require a backward measurement for each relevant downstream component. The [prior eight-anchor B200 benchmark](aspd_v3_1_token_discovery_and_selection_frame.md) measured 218.20 seconds after initialization, 56.75 GiB GPU peak, 5.70 GiB host peak, 5.71 MB of compressed raw records and 8.10 MB of compressed browser graphs. Linear extrapolation gives about 27 seconds per anchor, 27 minutes for the 60-position window and 3.7 B200 GPU-hours for 489 anchors, plus initialization. At the sampled compression ratio the pilot would use about 0.84 GB for both exports. Extending that same extrapolation to all 100,352 archive anchors gives approximately 760 B200 GPU-hours and 173 GB, excluding additional source recovery and retries. Natural documents and active support may differ substantially from the benchmark. Benchmark one recovered natural window before scaling any graph job. The first replay needs zero new Taylor measurements.

## Reproduction

Run with the existing ASPD Python environment and pinned tokenizer cache. Use a new immutable output directory.

```bash
python -m aspd.multimodel.cli.export_natural_token_pilot \
  --archive "$TEST_ARCHIVE" \
  --evidence multimodel_dashboard/src/workbench/data/expandedNaturalValidationGraphs.json \
  --checkpoint 735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969 \
  --output "$PILOT/pilot.json" --replay-inputs "$PILOT/replay-inputs.json" \
  --contrasts "$PILOT/contrasts.json"

python -m aspd.multimodel.cli.replay_natural_token_components \
  --config configs/multimodel/aspd_v3/negation_mechanism_eval_v1.yaml \
  --archive "$TEST_ARCHIVE" --pilot "$PILOT/pilot.json" \
  --replay-inputs "$PILOT/replay-inputs.json" \
  --window 36c8718babc8f3c5dcb8e5cfe9cc229a9f5726909ab07e354ff116c9b9a4ce57 \
  --component gemma.L15.mlp_up/7417 --component gemma.L15.mlp_down/7417 \
  --local-pair gemma.L15.mlp_up/7417=gemma.L15.mlp_down/7417 \
  --output "$PILOT/gemma.json"
```

Add `--execute --device cuda:0` on an authorized allocation for the replay. Use `smol.L08` or `granite.L20` with separate output files for the corresponding model. `--window` selects all verified positions from one saved example. One replay accepts up to four source documents, 512 exact anchors and 12 explicitly selected components, with one source model loaded. To include other NVX-025 endpoints, pass their exact IDs as additional `--component` arguments. A new graph measurement must retain the frozen test support, pinned output scale, same-anchor Taylor formulation and existing representative and parent-selection semantics.

Publish `pilot.json` under `natural-token-pilots/NVX-025/` relative to the static Workbench. Optional model scalar files go in the same directory and exact-anchor graph sidecars under `graphs/<anchor>.json`. These sidecars use the schemas in `naturalTokenEvidence.ts`; the reader rejects different checkpoints, test inventories, pools and anchors. The backend-only token replay inputs are stored separately from the browser.

The immutable artifacts are under `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/natural-token-nvx025-v3/`. The source archive is `/blue/thai/vinhnguyen1/aspd_v3/evaluation/quick_intruder_jaccard_500m_v1/extraction/full/job_51194062`. Its manifest SHA-256 is `d543df040e67e457490597ff84ff144b5d3707995ceb62ddeb3eb804f3753285`. The pilot export records the tokenizer revisions, evidence hash and exporter hash. The browser export SHA-256 is `f040caaea0b2747450a574df9446beae5ea6fad164f585e8c91aaa58224d3673`.

Validation included 73 frontend tests, eight targeted Python tests, TypeScript, Ruff, a production build, and a headless interaction test that selected the March token, kept a component selected across tokens and returned to the saved graph.

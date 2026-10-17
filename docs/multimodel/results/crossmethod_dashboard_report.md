# Cross-method research dashboard implementation report

The existing multimodel React/FastAPI dashboard now inspects ASPD, BatchTopK Crosscoder, and the
tracked phenomenon-level comparison without changing either scientific training implementation.

## A. Files created or modified

Backend:

- `aspd/multimodel/crosscoder_dashboard.py`: isolated read-only Crosscoder and research-comparison
  stores.
- `aspd/multimodel/dashboard.py`: shared exact-token context adapter and additive API routes; all
  existing ASPD store behavior and routes remain.
- `aspd/multimodel/cli/serve_analysis.py`: optional paths plus standard-path auto-discovery.
- `tests/test_multimodel_dashboard.py`: Crosscoder loading, unavailable-state, classification,
  annotation-isolation, reference-validation, cross-link, and endpoint tests.

Frontend:

- `CrosscoderOverview.tsx`, `CrosscoderFeatureExplorer.tsx`, and `MethodComparison.tsx` are the only
  new page components.
- `App.tsx`, `api.ts`, and `types.ts` add method navigation and explicit schemas.
- `Charts.tsx` adds the fixed-threshold relative decoder norm histogram.
- `FeatureExplorer.tsx` adds structured ASPD-to-Crosscoder links.
- `ActivationExamples.tsx` and `styles.css` distinguish measured zero from unavailable positions.
- `multimodel_dashboard/README.md` documents Windows inputs and commands.
- The tracked production bundle under `multimodel_dashboard/dist/` is rebuilt.

Research structure:

- `qwen3_crosscoder_aspd_phenomena.json` records only phenomenon status, interpretation,
  limitations, and method IDs. It deliberately contains no copied measured values or raw contexts.

## B. Backend schema and endpoints

Existing endpoints remain stable. The additive endpoints are:

- `GET /api/crosscoder/overview`
- `GET /api/crosscoder/candidates`
- `GET /api/crosscoder/features/{feature_id}`
- `GET /api/research-comparison`

The existing `GET /api/features/{feature_id}` optionally includes `related_crosscoder`; the new
Crosscoder feature endpoint optionally includes `related_aspd`. Both are derived only from the
tracked comparison JSON. Missing Crosscoder/comparison configuration returns an explicit
`available: false` payload rather than placeholder measurements.

## C. Existing code reused

- The ASPD `AnalysisDashboardStore`, five original pages, notes, exports, health checks, and
  endpoint paths are unchanged in meaning.
- Both feature explorers use the existing `ActivationExamples` token renderer.
- `Metric`, `Badge`, `Section`, `Source`, and the existing visual system are reused.
- The exact-token backend adapter is now generic and used by both ASPD and Crosscoder.

## D. Canonical source map

| Displayed field | Canonical source |
|---|---|
| FVE, validation L0, dead fraction/count, class counts, thresholds, convention | Crosscoder `analysis/summary.json` |
| Per-feature norms, cosine, fire count/density, strict class, blind ranks | Crosscoder `analysis/feature_metrics.json` |
| Blind inspection groups | Crosscoder `analysis/summary.json → candidate_groups` |
| Model IDs, site, configured capacity and step budget | Crosscoder `config_resolved.yaml` |
| Completed optimizer step | Last valid row in Crosscoder `metrics.jsonl` |
| Reference repositories/commits | Crosscoder `reference.json` |
| Exact Crosscoder token contexts and activations | Crosscoder `analysis/top_activation_examples.json` |
| ASPD P1-P5 values, taxonomy, contexts | Existing ASPD canonical artifacts read by `AnalysisDashboardStore` |
| Tentative labels and interpretation confidence | `qwen3_general_diff_annotations.json` |
| Phenomenon membership/status/limitations and cross-links | `qwen3_crosscoder_aspd_phenomena.json` |

The relative-norm histogram is a derived count aggregation over canonical feature scores. React
receives its bins and stored threshold ranges from the API; it contains no copied experiment
measurements.

## E. Implemented screens

- ASPD: the existing Overview, Feature Explorer, Candidate Finder, within-ASPD Comparison, and
  Data Health screens.
- Crosscoder Overview: provenance, measured training/evaluation diagnostics, strict class counts,
  central relative-norm histogram, negative strict-exclusivity result, and blind groups.
- Crosscoder Feature Explorer: strict class versus inspection group, measured norms/ranks/firing,
  exact contexts, separate post-ranking annotation, and structured ASPD links.
- Cross-method Comparison: global method/measurement summary and four tracked phenomenon cards
  with careful evidence states.

## F. Verification

- `54` multimodel pytest tests pass.
- Ruff passes for `aspd/multimodel` and all multimodel tests.
- TypeScript project check passes.
- Vite production build passes and the served bundle is regenerated.
- Real-artifact verification reports ASPD `PASS`, 65,536 Crosscoder features, 48 frozen blind
  candidates, four valid comparison phenomena, and no Crosscoder/comparison diagnostics.

## G. Intentionally unavailable fields

- No Crosscoder vocabulary projection or causal logit effect is displayed.
- No vector matching, activation-vector correlation, or automatic semantic matching is implied.
- Crosscoder features outside the 48 extracted candidates honestly report contexts as not
  extracted.
- Missing method artifacts do not block the existing ASPD pages.

## H. Scientific guardrails

- Strict Crosscoder classification always comes from measured feature metrics; annotations cannot
  overwrite it.
- `base-tail`, `shared-center`, and `finetuned-tail` are visually labeled derived inspection ranks,
  not strict classes.
- Annotation panels say `NOT MEASURED GROUND TRUTH` and call confidence interpretation confidence.
- Relationship statuses include explanatory definitions and never use a binary matched/unmatched
  claim.
- Low-support ASPD evidence is explicitly flagged.
- ASPD-only parameter evidence remains asymmetric when no Crosscoder feature is referenced.
- Measured zero and unavailable token activation are encoded with different labels and patterns,
  not color alone.

## I. Repository state

Generated cache/checkpoint artifacts remain outside Git. The source diff contains only the
read-only dashboard extension, structured comparison references, tests, documentation, and rebuilt
frontend bundle. Final `git status` and diff statistics are reported in the task handoff.

# Multimodel ASPD v3.1 Negation discovery

[Literature, frozen protocol, and reproduction commands](../negation_mechanism_eval.md)

## Provenance and coverage

- Frozen checkpoint step: 61,036. Checkpoint manifest SHA-256: `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`.
- Source models and tokenizers are the pinned Gemma 2 2B IT, SmolLM2 1.7B Instruct and Granite 3.3 2B Instruct revisions recorded in the preparation manifest. Layer ranges are 9–16, 8–15 and 16–23, respectively. The natural BatchTopK pool has 2,048 anchors and K=32 per site.
- Preparation used code `c1fb958`. Smoke and discovery used compute code `b94a6a6`. The same-anchor graph stage uses `50df19e`. The smoke ran in a checkout with frontend-only edits in progress and its runtime manifest records this fact. The discovery and graph jobs use separate pinned, clean compute worktrees.
- 32 smoke, 128 discovery and 256 sealed held-out groups passed logic and exact three-tokenizer validation with zero rejected groups. They contain 144, 576 and 1,152 examples and 3,992, 16,024 and 35,988 task anchors, respectively. The held-out rows are in a separate sealed directory.

## Native-model behavior

The table reports **exact next-token correct / closed-candidate correct**, each out of 128 for a factorial condition and 32 for a scope control. The numbers describe these generated prompts. Candidate accuracy uses the maximum-logit answer within the slate.

| Model | Affirmative | Negative | CF affirmative | CF negative | Scope control | CF scope control |
|---|---:|---:|---:|---:|---:|---:|
| Gemma | 54 / 98 | 87 / 120 | 51 / 97 | 89 / 126 | 9 / 14 | 11 / 15 |
| SmolLM2 | 56 / 78 | 75 / 121 | 53 / 86 | 83 / 124 | 21 / 21 | 11 / 11 |
| Granite | 97 / 114 | 92 / 123 | 105 / 116 | 100 / 127 | 12 / 15 | 18 / 22 |

Attribute next-token exact accuracy was 0/128 for both Gemma and SmolLM2, even though their closed-candidate accuracy varied with condition. Granite achieved 51/128 exact in that family. This family is useful for ranking and probability-mass diagnostics, and its weak free prediction baseline limits interpretation of exact negation failure. The scope-control errors are also substantial when the negative clause refers to the irrelevant person.

Across the 256 negative and counterfactual-negative examples per model, the affirmative-prerequisite classifications were:

| Model | Affirmative prerequisite failed | Successful negation | Repeated excluded token | Unrelated output |
|---|---:|---:|---:|---:|
| Gemma | 151 | 101 | 0 | 4 |
| SmolLM2 | 147 | 87 | 0 | 22 |
| Granite | 54 | 163 | 1 | 38 |

Counts are per model, and the gate support at a shared anchor is a joint encoder measurement. The final-query plots permit comparison of scope, polarity, alternate candidates, and shared features. This sample gives only one affirmative-conditioned replay of the excluded token, so there is no defensible population estimate of which dependencies distinguish such failures.

For the 32 original affirmative/negative pairs with multiple valid alternatives, the excluded token's mean raw logit decreased and the mean raw logit of the same two now-valid alternatives increased in all three models.

| Model | Mean excluded logit Δ | Mean valid-alternative logit Δ | Mean valid-mass Δ |
|---|---:|---:|---:|
| Gemma | −4.228 | +3.728 | +0.545 |
| SmolLM2 | −1.959 | +2.310 | +0.330 |
| Granite | −6.385 | +3.758 | +0.570 |

`paired-discovery.json` contains both polarities of target assignment and all four families. These paired output observations can guide follow-up interventions. Raw logit levels can vary with a prompt-dependent common offset, and an increase in probability mass can follow redistribution across the vocabulary.

### Measured discovery examples

- **Affirmative and successful negation.** `negation-discovery-binary-0000` asks Eva to choose from green or wood. Gemma predicts ` green` in the affirmative and ` wood` when green is excluded. Its fixed-orientation margin shifts from 1.250 to −2.4375.
- **Excluded-answer repeat.** In `negation-discovery-multiple-0000/negative`, Alex must select from cloud, blue or table and will not select cloud. The valid set is table and blue. Granite predicts ` cloud` after answering the affirmative counterpart correctly. The negative fixed margin is 0.625.
- **Scope control.** In `negation-discovery-scope-0001/scope_control`, Mira will choose chalk and Omar will not choose chalk. SmolLM2 predicts ` chalk`, with fixed margin 2.25.
- **Out-of-slate output.** In `negation-discovery-attribute-0001/counterfactual_negative`, brush is tagged gold and excluded, so water is the valid alternative. Granite predicts ` code`. The distinction between this outcome and repeating brush is retained in the artifact.

## Scientific interpretation

The matched candidate logit changes, contrasts, same-anchor edges and cross-model parent distributions are exploratory observables. Establishing suppression or construction would require candidate-specific intervention and restoration on matched contexts. Similar parent distributions and shared gate IDs can suggest corresponding components to investigate. The final-query graph is restricted to the selected matrix positions and cannot establish how negation information arrived from earlier tokens. The generated prompts use explicit closed slates and literal “not”. Word choice and template position could drive some of the gate contrasts. Paraphrase forms and factual benchmarks are outside this first run.

## Gate observations

There were 4,651 observed shared gates in smoke and 6,260 in discovery. Of 128 affirmative and 128 negative final-query anchors in discovery, `PRE:595` fired on 90 and 128, respectively, while `PRE:2971` fired on 108 and 39. `MID:6608` fired on every factorial final query, 512/512, and every scope-control query, 64/64. Its maximum imported representative β is 0.4863 for Gemma, 0.1636 for SmolLM2, and 0.2043 for Granite. These are measurements of shared activation and reference locations, with computational roles open for testing.

The fixed shortlist selected two gates per contrast direction and two across-condition gates at each site. The category-wise eligible counts, after excluding prior selections, were 44, 48 and 28 for PRE and 30, 38 and 24 for MID, ordered as negative, affirmative and across. The selected 96 gate–group pairs cover 316 unique final queries. The single excluded-answer repetition is available as an activation example through the native-outcome filter. Its matched group was outside the frozen dependency shortlist.

## Measured graph example

At the negative query of `negation-discovery-binary-0000` (anchor `09956cb8b7085626aeda956ce353ba2e273ef1caf39a9e229084e55a078e1b0f`), `MID:6608` had selected continuous activation 0.23338. Granite's stored edge from `L16.mlp_up/6608` to `L16.mlp_down/6608` has signed Taylor effect −2.52573 and normalized weight 0.17173. The total normalized parent strength is 0.45219 in Granite, 0.46195 in SmolLM2 and 0.30431 in Gemma.

| Model pair | Parent-gate Jaccard | Parent-distribution TV |
|---|---:|---:|
| Gemma / SmolLM2 | 0.6250 | 0.3251 |
| Gemma / Granite | 0.4167 | 0.4850 |
| SmolLM2 / Granite | 0.4000 | 0.3043 |

The shortlist has 428 gate–query records, including 102 with zero natural support. The active records produced 978 graphs, 2,934 downstream component measurements, 26,960 parent edges and 154 distinct upstream gate IDs. The export includes those upstream gates' reference β and exact-sequence activations. Cross-gate graph joins are limited to measurements sharing the selected model and anchor.

## Execution and artifacts

The artifact root is `/blue/thai/share/multimodel-ASPD/repo/.codex_runtime/negation-evaluations/v1-c1fb958`. Each Slurm job used one B200 and a 32 GiB host-memory allocation.

| Stage | Slurm job | Elapsed | CUDA peak allocated | Process peak RSS |
|---|---:|---:|---:|---:|
| Native and ASPD smoke | 52744461 | 05:45 | 61.63 GB | 5.73 GiB |
| Native and ASPD discovery | 52745244 | 08:45 | 61.60 GB | 5.79 GiB |
| Targeted same-anchor graphs | 52746296 | 04:50 | 56.31 GB | 5.80 GiB |

The 34.84 MB smoke and 99.88 MB discovery flat scientific artifacts exclude their stream-state databases. Smoke has 1,659 sparse contribution rows in a diagnostic subset. Discovery contribution norms and coefficients are present in the targeted graph artifact. The individual stage manifests and `COMPLETE` digests record exact inputs, runtime, and source revisions.

The frontend export is 98,364,826 bytes. Its index SHA-256 is `9676754288cd450b72a8a24286e3291c4a4335003c88e86286c53c692e0a71c3`. All 161 indexed asset hashes and same-group availability for all 154 parent gates were verified. The TypeScript check, production build, 24 Vitest tests and 59 selected Python tests passed. Validation covered source-to-asset joins and frontend contracts. Interactive browser testing remains to be performed in a demo session.

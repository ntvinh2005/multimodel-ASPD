# ASPD v3.1 human interpretability protocol

## Status and purpose

This protocol prepares a blinded, offline human evaluation of semantic coherence at the matched 10M-exposure ASPD v3.1 checkpoints. The final semantic judgments are supplied by the researcher, not by an automated judge. No ASPD checkpoint is trained or modified.

The primary question is whether per-anchor feature-axis z-scoring produces more coherent and distinguishable natural feature contexts than the original `Linear → ReLU → BatchTopK` gate at identical accepted training exposures.

## Frozen comparison

- Control: `full_only_control`, step 1,221, 10,002,432 accepted aligned-anchor exposures.
- Treatment: `per_anchor_zscore`, step 1,221, 10,002,432 accepted aligned-anchor exposures.
- Models: pinned Gemma-2-2B-IT and SmolLM2-1.7B-Instruct only.
- Dictionary: C=8192, K=32, separate shared PRE and MID gates.
- Objective: full-only activation reconstruction and original AuxK.
- Source weights and checkpoints are read-only.
- Qwen execution, new ASPD training, and a third source model are prohibited.

## Held-out integrity

The human evaluation uses a new test-document stream. Its exclusion manifest is the deterministic union of:

1. 296 documents used in the first mechanistic-validation extraction; and
2. 289 subsequent documents used in the final automated-judge extraction.

The 585-document union is frozen at `/orange/thai/vinhnguyen1/aspd_v3/human_interpretability_v1/provenance/excluded_documents.json`, expected SHA-256 `f90c4c5e94ff52588bd887d4cc4d0e756037a30189413bbd5a8c6e0c9f826407`. The new control and treatment extractions must share exactly 100,352 aligned anchors from source-document-disjoint test buckets 10–19.

## Sampling and human workload

The researcher requested no more than 50 displayed items. The frozen workload is:

- 40 unique scientific parent features: 10 each from Control PRE, Control MID, Z-score PRE, and Z-score MID;
- 4 blinded repeated presentations, one per arm/site group, for within-rater consistency;
- 6 unscored practice items, containing both obvious and ambiguous patterns;
- 50 displayed items in total.

Within every arm/site group, eligible features are divided into firing-frequency tertiles and sampled with a fixed seed. Sampling does not use automated Intruder scores, semantic labels, causal effects, or manual inspection. Failed task construction is reported and is not silently replaced.

Eligibility requires at least four natural fires in at least four distinct source documents. Eligibility denominators and feature coverage remain separate scientific outcomes from human accuracy.

## Context construction

Every scientific parent has four high-activation contexts from distinct documents and one intruder from another sampled feature in the same frequency stratum. The target feature must be inactive at the intruder anchor. Intruder selection prioritizes source-group, text-shape, and length matching and rejects normalized-text Jaccard above 0.8.

The extraction retains up to 768 original characters on each side of the aligned anchor. The exact highlighted span is the original Gemma tokenizer token whose fast-tokenizer offset ends at the shared canonical character boundary. Extraction fails if the offset, boundary, and source substring do not agree. The interface initially shows a compact ±192-character view and offers the retained longer context without changing the anchor.

All five examples use identical highlighting. Activation magnitudes are never displayed in the blinded interface.

## Blinding and presentation

Scientific items receive neutral IDs such as `Feature Set 001`. The public dataset and standalone HTML contain no arm, site, feature ID, firing frequency, activation, source-document ID, donor identity, or correct answer. These fields are stored only in a separate mode-0600 answer key.

Intruder positions are exactly balanced within each arm/site group. Assignment is randomized subject to a maximum within-stratum count spread of two. Arm/site groups are interleaved round-robin before neutral IDs are assigned. Repeats are re-permuted and are not identified in the public interface.

## Human response fields

For each item, the evaluator records:

1. one Intruder choice or `Cannot determine`;
2. coherence from 1 (none) to 5 (strong and specific);
3. pattern category: semantic, syntactic, stylistic, formatting, mixed, or unclear;
4. confidence from 1 to 5; and
5. a verbatim free-text explanation.

The offline application stores progress in browser local storage, exports response JSON, and can import that JSON to resume. It contains no grading logic or answer key. Grading occurs only after exported responses are supplied to the separate analysis script.

## Estimands and uncertainty

For Control/Z-score × PRE/MID, the grader reports:

- intention-to-evaluate Intruder accuracy, with abstentions counted incorrect;
- abstention rate;
- accuracy conditional on forced-choice responses;
- mean and median coherence;
- frequency-stratified results;
- activity-weighted accuracy and coherence as separate estimands;
- treatment-minus-control differences by site; and
- feature-level stratified bootstrap intervals with 5,000 replicates.

Repeated presentations are excluded from primary estimands and used only for within-rater consistency. With one evaluator, results are explicitly exploratory and do not establish inter-rater reliability. With multiple evaluators, the grader additionally reports pairwise raw agreement and Cohen's kappa for Intruder choices, plus category, uncertainty, and coherence agreement.

## Optional automated comparison

At the researcher's later request, the same frozen unique scientific parents may also be scored by the pinned local Llama-3.3-70B-Instruct judge using a single forced-choice presentation. This is a separately labeled exploratory comparison. It does not replace the human result, and raw automated accuracy is reported even if the model has answer-position preference.

## Completion gates

The package is deliverable only after checks pass for checkpoint identity, matched anchors, natural support, document distinctness, intruder inactivity, exact anchor spans, position balance, public-field leakage, answer-key isolation, response import/export schema, grading, Ruff, pytest, Slurm syntax, source-weight immutability, and absence of changes to training runners/configurations.

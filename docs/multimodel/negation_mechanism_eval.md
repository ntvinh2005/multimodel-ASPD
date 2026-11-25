# ASPD v3.1 Negation evaluation protocol

## Literature and scope

[Zhou et al. (ICML 2026, arXiv v2)](https://arxiv.org/html/2605.03052v2) compare paired positive and negated completions, measuring both candidate-ranking accuracy and whether the relative margin moves toward the negative answer. Their curated data have 162 concept pairs in four templates. Their attention-sink ablations and path patches localize helpful and counterproductive attention. Logit-lens projections and annotation support stronger construction of valid alternatives alongside weaker direct suppression in the models they analyze mechanistically. Their [release](https://github.com/Ja1Zhou/LM_Negation) includes single- and multi-answer data under CC BY 4.0, code under Apache-2.0, and logs. Appendix D re-evaluates with multiple answers, averaging logits across answer sets. Final-layer behavior and layer-local interventions have different interpretations.

[Yoon et al. (arXiv v2)](https://arxiv.org/html/2610.09571v2) evaluate minimal negations of question pairs across factual, synthetic-reasoning and visual benchmarks. Answer change is assessed among originally correct cases, with category preservation assessed on changed answers. Appendix B describes automatic and human validation. The authors use attention path patching, steering the final-query residual, and component ablations and subspace interventions. They report weakened original-answer retrieval and promotion of favored within-category alternatives, including cases where this preference fails to exclude the original answer. Their confidence-adaptive training is outside this evaluation. The preprint says the code and dataset are to be released on publication. No official release was identified in the preflight.

This run uses newly generated closed-candidate examples designed to separate scope, answer identity and polarity. It does not reproduce the papers' benchmarks or interventions. The native scores describe prompt-conditioned behavior. The ASPD graph describes same-anchor first-order component dependencies within the checkpoint's selected layers.

## Dataset and readouts

- Start with 32 smoke and 128 discovery groups, equally allocated to binary exclusion, fictional attribute exclusion, multiple valid alternatives and scope. Four factorial conditions per group, plus two scope controls for scope. Generate 256 sealed held-out groups with disjoint candidate slates, full prompts and query templates.
- Binary and multiple-choice groups name arbitrary code labels and constrain the choice to their slate. Attribute groups state the fictional tag assignments. Scope groups place the same lexical negator in either the queried or irrelevant entity clause. Candidate presentation order and orientation are randomized independently and recorded.
- Require exact marked character boundaries across pinned tokenizers and strictly one appended native candidate token in the complete prompt context. A candidate or boundary failure rejects the entire group. Group counts and rejection reasons are recorded.
- Fixed margin is logit(A) minus logit(B) throughout a group. Correctness margin is maximum valid-candidate logit minus maximum invalid-candidate logit. Exact accuracy uses the full-vocabulary argmax. Candidate logits and full-vocabulary probabilities are saved per identity. Valid and excluded masses sum their named candidates. Unrelated mass is the full vocabulary outside the slate. Failed negation means either an affirmative-task prerequisite failed, the model repeated the excluded answer, or it generated another invalid continuation. The definitions are used separately per model.

## Bounded dependency shortlist

Disambiguate the original and counterfactual pair in each group. Let the signed paired activation difference be mean final-query activation on the two negative conditions minus the mean on the two affirmative conditions. A contrast group qualifies when this difference has the selected sign and both favored conditions fire. An across-conditions group qualifies when all four conditions fire. A gate needs natural-validation representatives in all three models, at least four qualifying groups, and qualifying groups from at least two families. Per site, select at most two positive-contrast, two negative-contrast and two across-condition gates. Rank by the sum of qualifying strengths, resolving ties by gate ID. Sample up to eight qualifying groups per selected gate using the declared seed, rotating through eligible families. Record coverage shortfalls. Keep every condition in each sampled group, including inactive controls.

Use stored natural gate support and exact canonical input from the matched discovery evaluation. Same-anchor native Taylor attribution uses approved representatives as targets and all active upstream components in compute order. Keep top 10 by absolute signed effect, with strict threshold zero; preserve normalized weights, ranks, aggregated parent scores, distributions, total strength, Jaccard and TV. The graph offers exploratory evidence at the final query.

## Execution and resource gates

Induction baseline is `b8228fc9f4146f4675f55d5c79e6de24da2b578e`. The measured-workbench branch resolves to `2c7233e`, while GitHub main resolves to `a1ec76b`. The expected PDF is absent from those local tracked paths; the checked-in formulation map, decisions, thresholds and training/evaluation configs define the implementation contract.

The Induction discovery and graph jobs peaked around 63 and 56 decimal GB of GPU allocation respectively. Use one B200 per stage with 32 GiB of host RAM. Estimate 5–15 minutes for the smoke, 12–25 minutes for discovery, and 5–15 minutes for targeted graphs, excluding queue time. Budget 1 GiB for preparation, sparse raw records and frontend exports combined at this scope. These are estimates until the measured resource manifests are produced. Run native evaluation before ASPD on the same smoke job. Continue only when exact anchor native scoring and frozen natural gate validation succeed.

## Reproduction

Run the following from the Negation checkout root. Use exclusive output paths for every stage.

```bash
PY=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python
CONFIG=configs/multimodel/aspd_v3/negation_mechanism_eval_v1.yaml
export HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface
export TOKENIZERS_PARALLELISM=false
"$PY" -m aspd.multimodel.cli.prepare_aspd_v3_negation --config "$CONFIG" --output "$PREPARATION"

# Use the recorded compute Git revisions, or set REVISION to the exact approved rerun revision.
sbatch --export=ALL,REPO="$PWD",REVISION="$REVISION",PREPARATION="$PREPARATION",OUTPUT="$SMOKE",STAGE=smoke slurm/multimodel/aspd_v3_negation.sbatch
# After verifying SMOKE/COMPLETE and the behavioral results:
sbatch --export=ALL,REPO="$PWD",REVISION="$REVISION",PREPARATION="$PREPARATION",SMOKE="$SMOKE",OUTPUT="$DISCOVERY",STAGE=discovery slurm/multimodel/aspd_v3_negation.sbatch
# After validating the discovery manifest:
"$PY" -m aspd.multimodel.cli.measure_negation_query --plan-only --config "$CONFIG" --preparation "$PREPARATION" --discovery "$DISCOVERY" --output "$GRAPHS"
sbatch --export=ALL,REPO="$PWD",REVISION="$REVISION",PREPARATION="$PREPARATION",DISCOVERY="$DISCOVERY",OUTPUT="$GRAPHS",STAGE=graphs slurm/multimodel/aspd_v3_negation.sbatch

# In a clean copy without a previously exported Negation asset directory:
PYTHONPATH=. "$PY" multimodel_dashboard/scripts/export_negation_workbench.py \
  --config "$CONFIG" --smoke "$SMOKE" --discovery "$DISCOVERY" --graphs "$GRAPHS" \
  --output multimodel_dashboard/public/negation
cd multimodel_dashboard
npm test
npm run build
```

Preparation contains the full frozen native tokenizations and marked canonical anchor identities. Each measured split contains `native_predictions.jsonl`, `behavior_summary.json`, `anchor_activations.jsonl`, `activation_examples.jsonl`, `component_contributions.jsonl`, `gate_summary.json`, `pools.json`, and a hash-checked manifest. Sparse direct component rows in the smoke cover a small diagnostic subset. Targeted discovery graphs include coefficients and contribution norms for their active representative targets. A browser group asset contains all combinations in that exact group alongside any measured graphs for its individual examples; its path and hash are recorded in the frontend index.

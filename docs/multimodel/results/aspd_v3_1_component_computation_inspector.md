# ASPD v3.1 Component Computation Inspector

## Workbench

Open Natural Validation, search `NVX-025`, choose **Same-gate extension**, and click an Up or Down component. The existing component inspector includes **Component computation**. In Negation or Induction, select a token and click a component in its measured graph. The selected component remains the target as you click other tokens in that model. **Component contexts** ranks sampled matched positions by shared gate activation, signed coefficient, or contribution magnitude. Signed values are ordered by absolute value and retain their displayed sign.

The rank-one factor binary uses little-endian float32, with (u) first and (v) second. Its dimensions and SHA-256 are provided in the JSON index. Each pair is stored once. Reference beta, rho, firing count, sigma, representative membership and native head layout are sourced from the frozen artifacts and pinned model configs. The existing notes editor uses the same storage identity. Optional computation fields are compatible with old note records and JSON exports.

## NVX-025 case study

The candidate is row 45 of **Mechanism candidates** in the [Hypothesis list](https://docs.google.com/spreadsheets/d/1sUovM3yM_p_Djl5Txqw3_1E3L2yELeGux1803BLMzHs/edit#gid=0&range=A45:G45). The archived graph gives `MID:1117 → MID:7417 → PRE:1529`, plus a same-layer `MID:7417` Up→Down input at all 13 exact shared validation anchors. The extraction recovered 12 distinct factor pairs, verified 78 archived target coefficient records and resolved 39 Up→Down records directly against the indexed raw tensors. It loaded selected checkpoint rows by memory mapping their standalone DCP tensors. The raw shard completion markers and raw index files were checked against the archive. The extraction checked each sampled endpoint, anchor, coefficient and score rather than rehashing entire multi-GB edge tensors.

Let $h = \phi(a)\odot b$ and remove the Up write $c=e_u u_u$ while holding the native Gate branch fixed. Then

$$
e'_d-e_d=-g_d v_d^\top(\phi(a)\odot c)=-E_{u\to d}.
$$

This equality is affine in real arithmetic for a same-layer Up→Down link with frozen gates. The result is inferred from the archived gradient and baseline, so native finite-precision replay remains a distinct check. Gate→Down is generally nonlinear. In the archived motif Gemma and Granite use Gate for the cross-gate input. SmolLM2 uses Up for that input.

At validation anchor `06533e8f4df33be284ddd846103562542bfa957e2e31d90b8506a1f3b1dae40d`:

| Model | Up → Down endpoints | Archived Down e | Predicted Δe | Predicted Down e after removal | Median ‖Δe‖ / ‖e‖ over 13 anchors |
|---|---|---:|---:|---:|---:|
| Gemma 2 | L15 Up/7417 → L15 Down/7417 | 0.79073 | −0.66053 | 0.13020 | 83.5% |
| SmolLM2 | L08 Up/7417 → L08 Down/7417 | −33.44178 | +22.16983 | −11.27195 | 65.2% |
| Granite 3.3 | L20 Up/7417 → L20 Down/7417 | −6.75735 | +5.13441 | −1.62294 | 72.3% |

The removal reduces the absolute Down coefficient in all 39 cases. This quantifies an internal dependency on the Up branch that shared gate activation cannot resolve. The per-component ratio is invariant to sign and scale changes in the Down factorization. The raw coefficient signs do not identify semantic excitation or inhibition. The validation anchor texts, full native inputs, and continuous gate values were absent from these raw graph archives. Held-out month windows in the candidate evidence use different anchors. The results leave month position versus date-template sensitivity unresolved.

## Matched-context check

A separate archive-only check held the visible `from` span fixed in 16 Selection Frame examples from four matched groups. Each component used the first decomposed layer's MLP Up matrix in its model with shared gate `MID:2844`. All 16 native anchor positions were verified against their original group and graph manifests. The gate fired in eight examples. For the eight inactive controls, the contribution is a structural zero and the ungated projection is unknown.

The strongest shared gate activation was at `from` in “Pick the best option from the list.” In Gemma and Granite the largest absolute Up coefficient instead occurred in “Choose the book from the shelf.” In SmolLM2 the shared gate and component rankings both put the Pick context first. This is a demonstration of independent component ranking within the archived sample. It does not establish which linguistic rule drives any component.

## Available measurements and native replay

| Source | Available | Additional evidence needed |
|---|---|---|
| Natural Validation | Raw target coefficients at the selected anchors, archived E and w, static factors and importance | Exact original document tokens, native matrix inputs and continuous scores at those same anchors |
| Selection Frame | Coefficients and normalized contributions for naturally active components at the covered tokens; gated zeros for inactive components | Ungated reading at inactive anchors and native intermediate activations |
| Induction and Negation | Existing aligned support and whatever target measurements were archived for each token | Additional measurements only where the chosen component and token lack records |

The targeted replay command uses the existing frozen source-model loader, physical-matrix capture hooks and logical projection slices. It reuses archived support and verified native token IDs. It never reruns BatchTopK on its reduced batch. For local MLP replay it verifies the recomputed baseline against the actual captured Down input, then uses the model's actual activation function and exposes the most changed channels. For Q/K/V it can run the pinned model’s native eager attention, keep native masks and positional transforms, and remove a contribution at exactly one projection site. It exports probability changes, log-probability ratios centered by the softmax normalizer, and changes after the native output projection. K/V reports are indexed by the consuming native query positions. These quantities are attention inspection within one model and do not add cross-token ASPD edges. O and Down inspection use their captured inputs and direct residual-stream projection contributions. Native replay of the historical NVX-025 witness requires exact document recovery first. No new end-to-end native replay or GPU job was run for this deliverable.

The archive pilot recorded 23.66 s CPU wall time and about 544 MiB process RSS. Its 479,185 bytes include the 12 factor pairs. A native pilot uses one model at a time and up to 24 verified anchors for MLP, or four for explicit attention. Plan 16 GiB host RAM, and reserve 12–16 GiB GPU memory if using a B200. These are allocation estimates for one frozen source model, separate from the previous 56.75 GiB three-model dependency benchmark. Budget 5–15 minutes including cold model access and export per model. The exact native use remains unmeasured. The scientific next check is a fixed-gate replay on recovered validation anchors and a separate 12–24-context design crossing day-first dates, month-first dates, names and prose.

## Reproduce

Run from the repository root in the existing ASPD Python environment. Use a new output directory each time.

```bash
python -m aspd.multimodel.cli.export_component_computation \
  --checkpoint "$ASPD_CHECKPOINT" --validation "$ASPD_VALIDATION_ROOT" \
  --evidence "$ASPD_ARCHIVE/evidence.json" --extensions "$ASPD_ARCHIVE/extensions.json" \
  --hf-cache "$ASPD_HF_HUB_CACHE" --output "$ASPD_COMPUTATION_EXPORT"

python -m aspd.multimodel.cli.export_component_contexts \
  --tokens "$ASPD_WORKBENCH/tokens" --dataset selection-frame --token from \
  --group selection-frame-discovery-selection_verb_by_polarity \
  --group selection-frame-discovery-selection_vs_removal_by_polarity \
  --group selection-frame-discovery-selection_voice_by_polarity \
  --group selection-frame-discovery-nonselection_voice_by_polarity \
  --component gemma.L09.mlp_up/2844 --component smol.L08.mlp_up/2844 \
  --component granite.L16.mlp_up/2844 --output "$ASPD_CONTEXT_EXPORT"

python -m aspd.multimodel.cli.replay_component_computation \
  --dataset selection-frame --split discovery \
  --config configs/multimodel/aspd_v3/negation_mechanism_eval_v1.yaml \
  --preparation "$ASPD_SELECTION_PREPARATION" --activations "$ASPD_SELECTION_ACTIVATIONS" \
  --component smol.L08.mlp_up/2844 --downstream smol.L08.mlp_down/2844 \
  --anchor eee70acfef804b013a4134ac1bd8d292e7c50949b5bd223817d8f1d1d2df5a22 \
  --output "$ASPD_NATIVE_EXPORT"
```

The last command verifies provenance and alignment by default. Add `--execute --device cuda:0` on an authorized compute allocation to collect activations. For Q/K/V, use `--attention` in place of `--downstream`. Publish the first export at `component-computation/nvx025/` under the static Workbench root. For a native replay, the inspector looks for `component-computation/native/<model>.L<layer>.<matrix>--<feature>.json` when the Native computation section opens. It only applies rows matching the checkpoint, dataset, anchor, pool and support. The old graph data, checkpoint, and saved notes retain their existing identities.

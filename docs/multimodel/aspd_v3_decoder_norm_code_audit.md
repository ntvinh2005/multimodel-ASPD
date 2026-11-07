# ASPD v3 activation-decoder norm code audit

Date: 2026-11-07

Phase: **Phase 2 complete — pre-correction code review**

Source commit: `35711f9daa4629aea20c579f909db13501c56a52`

Mathematical authority:
`docs/multimodel/aspd_v3_mathematical_specification_audit.md`

## 1. Finding

The historical Granite normalized-target implementation enforces an activation-decoder
unit-row constraint in three separate optimization locations:

1. **Initialization:** `ActivationReconstructionBank.__init__` calls
   `self.normalize_decoders_()` after constructing every model/layer decoder
   (`v3/reconstruction.py:184-209`).
2. **Before Adam:** the model-owned trainer calls `model.project_decoder_gradients_()`
   (`train_aspd_v3_2b.py:1372-1377`). That method removes the component of each
   activation-decoder gradient parallel to its current row (`model_owned.py:424-428`,
   `reconstruction.py:230-242`).
3. **After Adam:** the trainer calls `model.normalize_decoders_(project_gradients=False)`
   (`train_aspd_v3_2b.py:1414-1420`). Every activation-decoder row is divided by its
   norm (`reconstruction.py:216-228`).

This is an optimization constraint even though it is absent from `balanced_main_loss`.
It removes the radial degree of freedom from activation decoders and prevents their
magnitudes from adapting to normalized targets.

## 2. Exact scope of normalization

| Parameter family | Initialization | Gradient projection | Post-step projection | Corrective scope |
|---|---|---|---|---|
| Activation decoder weights | Unit-row normalized | Radial gradient removed | Reset to unit-row norm | Remove for corrected Granite family |
| Activation decoder bias | Not normalized | Not projected | Not normalized | Preserve |
| Rank-one write factors `U` | Existing unit-write parameterization | Existing radial projection | Existing unit norm | Preserve; unrelated to this correction |
| Rank-one read factors `V` | Existing initialization | No activation-decoder operation | No activation-decoder operation | Preserve |
| Encoder/readout parameters | Not normalized by these methods | Ordinary gradients | Ordinary Adam | Preserve |
| Frozen backbones | Frozen | No gradient | No update | Preserve |

`ModelOwnedMain2B.normalize_decoders_` and `project_decoder_gradients_` currently combine
rank-one and activation-decoder operations, so the correction must separate their
conditional behavior rather than deleting the shared methods blindly.

## 3. Checkpoint behavior

`ModelOwnedCheckpoint.load` uses DCP to restore model and optimizer state directly. It
does not call activation-decoder normalization after loading. Model construction performs
an initial unit normalization before the DCP load, but the loaded checkpoint overwrites
those tensors. In historical training, every saved post-step state is nevertheless unit
normalized because the trainer projects immediately after every Adam step.

A corrected config changes the objective/config fingerprint and must use a new output
namespace. Existing checkpoint validation therefore rejects historical unit-constrained
checkpoints for the corrected run. The corrected pilot must initialize fresh and must not
resume the collapsed checkpoint.

## 4. Config discrepancy

`GraniteExperimentObjective.activation_decoder_normalization` accepts only
`unit_row_norm_with_radial_gradient_projection`, and every historical Granite base config
sets that value. This schema encoded the implementation discrepancy as if it were
mathematical approval.

The correction requires an explicit `none` mode in a new versioned experiment config.
Historical configs and checkpoint namespaces should remain readable and reproducible;
they must not silently change meaning.

## 5. Mathematical expression to code map

| Mathematical expression | Historical implementation | Correct? | Required change |
|---|---|---:|---|
| Hierarchical encoder Eqs. (2)–(5) | `encoder.py`, `model_owned.LocalSiteEncoder`, `SharedSiteEncoder` | Yes | None |
| Shared continuous gate Eq. (6) | `FeatureReadout`, `V3BatchTopK` | Yes | None |
| Detached binary support Eq. (7) | `model_owned.encode`, `prepare_sparse_support(binary=True)` | Yes | None |
| Free activation decoder Eq. (8) | `ActivationReconstructionBank.normalize_decoders_` plus trainer calls | **No** | Skip activation initialization, radial projection, and post-step projection when mode is `none` |
| Activation reconstruction Eqs. (9), (17) | `matryoshka_activation_fvu` + `ActivationDecoder` | Algebraic sum yes; feasible parameter space no | Preserve forward equation; free decoder magnitudes |
| Target normalization Eqs. (15)–(16) | `prepare_activation_target` | Yes | None |
| Activation FVU/Matryoshka Eqs. (13), (14), (18) | `masked_fvu`, `matryoshka_activation_fvu` | Yes | None |
| Rank-one internal Eqs. (10)–(12) | `RankOneMatrixBank`, `RankOneComponents` | Yes for this audit | Preserve |
| AuxK Eqs. (22)–(23) | `auxk_dead_feature_loss` | Yes to approved historical objective | Preserve |
| Balanced loss Eqs. (24)–(28) | `activation_loss_terms`, `internal_loss_terms`, trainer pool scaling | Yes | Preserve coefficients and reductions |
| Encoder gradient isolation Eq. (29) | detached binary support | Yes | Preserve |
| Post-hoc decoder mass Eq. (30) | General analysis utilities measure decoder norms; Granite evaluator lacks full distribution | Partial | Add read-only pilot diagnostics without training gradient |

## 6. Required minimal implementation

1. Add an explicit activation-decoder normalization mode to
   `ActivationReconstructionBank`; retain the historical default for old callers.
2. Pass the config mode through `ModelOwnedMain2B`.
3. Always preserve rank-one write normalization, but condition activation-decoder
   gradient/post-step projection on the explicit mode.
4. Add a new corrected Granite experiment/config namespace with mode `none`, fresh
   initialization, the already authorized 1,221-step budget, and no full-training flag.
5. Add read-only decoder norm statistics and instantaneous gate/reconstruction probes;
   do not introduce a norm loss or reconstruction scale.
6. Add tests proving free decoder norms survive initialization, Adam, checkpoint
   round-trip, and resumed optimization.

No other architecture, objective, coefficient, activation function, sparsity rule, or
model roster requires a change.

Phase 2 is complete. Phase 3 may implement only the changes above.

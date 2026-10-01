# Paper ↔ code

The only file in the repository that cites equation, section, table and figure numbers of the paper
("Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space").
Code and the other docs use symbol names only.

## Notation

| Paper | Meaning | Code |
|---|---|---|
| W, x_t, y_t = W x_t | decomposed matrix, its input and output at token t | the config's `decomposition_targets[].module_pattern`; `pre_weight_acts[module]` |
| C | components per matrix | `decomposition_targets[].C` |
| P_c = u_c v_c^T (§3.1) | rank-1 weight component | `U[c]` (u_c) and `V[:, c]` (v_c) of `param_decomp` components; `V ∈ R^{d_in×C}`, `U ∈ R^{C×d_out}` |
| g_{t,c} (§3.1) | causal importance; c fires iff g > 0 | `ci` (core); `aspd.ci` |
| e_{t,c} = g_{t,c} v_c^T x_t (Eq. 1) | contribution of c at t | `effective(...)` (`aspd.analysis.prompt.trace`), `zeta` in the evaluations |
| ŷ_t (Eq. 1) | reconstruction of y_t | `aspd.loss_utils.masked_module_output` |
| R, r_t (§3.2) | residual stream the activation decomposition reads | ASPD's `resid_site(s)`: resid-pre (Q, K, V), resid-mid (O, MLP) |
| g^s, d, ϕ (Eq. 5) | shared sparse encoder, activation decoder, gate map | `aspd.ci.aspd.SharedEncoder`: `W_enc`, `b_dec`, `W_dec`; ϕ = 1[· > 0] |
| L_internal (Eq. 7) | internal reconstruction | `InternalReconLoss` (`mode: fvu` or `matryoshka`) |
| L_act (Eq. 8) | activation reconstruction | `ActivationReconLoss` |
| λ_act, λ_internal, λ_auxiliary (Table 5) | loss coefficients | `coeff` of the entries named `act`, `internal`, `auxk` |
| PD Transcoder (App. B) | transcoder as a parameter decomposition | `ci_config.mode: pd_transcoder`, `component_arch: transcoder` |
| ASPD (§3.3) | this method | `ci_config.mode: aspd`, `component_arch: transcoder` |
| L_param (Eq. 10) | ‖W − Σ P_c‖²_F | core `FaithfulnessLoss` |
| L_ablate stoch / adv (Eq. 11, App. C.1) | stochastic / adversarial ablation KL | core `StochasticReconSubsetLoss` / `PersistentPGDReconLoss` |
| L_sparse, p, λ_freq (Eq. 12) | importance minimality | core `ImportanceMinimalityLoss` (`pnorm`, `beta` = λ_freq) |
| adaptive sparsity loss (App. C) | L_sparse with L0 control | `AdaptiveSparsityLoss` (`target_l0`, `k_i`, `gain_scale`, `band`) |
| Interact(c1, c2) (Eq. 9, 30) | component interaction | pair score `dot_coact` = κ(c1, c2)⟨u_c1, v_c2⟩ (`aspd.analysis.pairs.scores`, κ from `aspd.cli.coactivation`) |
| interact^QK_h (Eq. 31) | QK interaction per head | pair score on the `bilinear_form` link |
| interact_feature (Eq. 32) | component–feature interaction | SAE endpoints of the pair viewer (`aspd.analysis.pairs.features`) |
| interp_c, interp (Eq. 20) | intruder score | `aspd.eval.intruder_db` |
| sim (Eq. 21) | mean pairwise Jaccard of token sets | `aspd.eval.diversity` (`z_bar`) |
| effect_{j,c}, Comp(J,k), W′(J,k) (Eq. 22–23) | editing | `aspd.eval.editing` |
| localization (Eq. 24), ratio | editing scores | `localization`, `*_localization_over_random` |
| π(c), S̄ − S̄_random (Eq. 25) | matching | `aspd.eval.matching` (`A_cond` margin) |
| attrib(c, t), contrib^QK, contrib^OV, contrib^res, pattern_edit (App. G.1) | IOI analysis | `aspd.analysis.prompt.scores` |
| probe score (Eq. 27), mass_{h,c} (Eq. 28), Π, Δ (Eq. 29) | behavioural probes | `aspd.analysis.probes` |

## Methods (Tables 1–12)

| Paper row | Config | Arm (derived) |
|---|---|---|
| ASPD (Ours) | `configs/<model>/aspd.yaml` | `aspd` |
| PD Transcoder (Ours) | `configs/<model>/pdtc.yaml` | `pdtc` |
| PD Transcoder + param | `pdtc_param.yaml` | `pdtc_param` |
| PD Transcoder + ablate | `pdtc_ablate.yaml` | `pdtc_ablate` |
| VPD (= VPD without adaptive L0) | `vpd.yaml` | `vpd` |
| VPD with adaptive L0 | `vpd_adaptive.yaml` | `vpd_adaptive` |
| VPD + internal | `vpd_internal.yaml` | `vpd_internal` |
| VPD + internal + no param | `vpd_internal_noparam.yaml` | `vpd_internal_noparam` |
| VPD + internal + no ablate | `vpd_internal_noablate.yaml` | `vpd_internal_noablate` |
| model-wide ASPD (§5, App. C.3) | `configs/gpt2_all/aspd.yaml` | `aspd` |

`<model>` is `gpt2` (MLP_in, layer 0), `gemma2` (MLP_out, layer 13) or `qwen3` (attention O, layer 17).
Evaluation SAEs (App. C.2): `configs/sae/<model>.yaml`.

Every paper number was computed from the run `<model>_<arm>` (and `gpt2_all_aspd` for §5) and the
evaluation SAE `<model>` published on the Hub (`tueminh/wfd-runs`, folder `aspd/`;
`python -m aspd.cli.download <run> --sae <model>`), including the harvests and the editing feature
samples (`<sae>/attr_edit/`).

## Results

| Result | Command | Output | Column |
|---|---|---|---|
| Interp (Tables 1, 6, 10) | `aspd.cli.harvest`, then `aspd.cli.intruder` | `<run>/harvest/h-step*/intruder_summary.json` | `intruder_mean` |
| Interp at g > τ (Table 9) | `aspd.cli.intruder_threshold --ci-thresholds 0.01 0.1` | `intruder_summary_ci<τ>.json` | `intruder_ci<τ>_mean` |
| Sim (Tables 1, 6, 9, 10) | `aspd.cli.diversity [--threshold τ --criterion ci]` | `<run>/diversity/diversity_h-step*[_ci<τ>].json` | `sim`, `sim_ci<τ>` |
| Matching (Tables 2, 8, 12) | `aspd.cli.matching` | `<run>/matching/matching_c2o_step*.json` | `matching_A_cond_margin` |
| Single editing (Tables 3, 7, 11) | `aspd.cli.editing` | `<run>/attr_edit/` | `attr_localization_over_random`, `attr_localization_mean_over_k` |
| Multiple editing (Tables 3, 7, 11) | `aspd.cli.editing_multi --m <m> --setup cond` for m ∈ {1, 5, 10, 20, 50} | `<run>/attr_edit_multi/m<m>_cond/` | `multi_cond_localization_over_random`, `multi_cond_localization_mean_over_k_and_m` |
| All tables | `aspd.cli.tables --ci`, then `aspd.cli.latex` | CSV, `.tex` | — |
| IOI mechanisms (§5.1, Figs. 1, 2, 4–19) | pair app `aspd.cli.serve_pairs --qk-edit weight`, page `/prompt`: attribution patching, QK / OV / cross-layer contributions, QK weight edit | interactive | — |
| Semantic tracing (§5.2, Figs. 3, 20, 21) | pair app, page `/`: Interact (`dot_coact`) between components and pretrained SAE features | interactive | — |
| Attribution graph of a prompt | circuit app `aspd.cli.serve_app --run gpt2_all_aspd` | interactive | — |
| Probes (App. H.2–H.4) | `aspd.analysis.probes.{induction,duplicate,name_duplicate}` | stdout | probe score, mass, Π, Δ |

# Method and training

## Decomposition

A matrix W with input x_t and output y_t = W x_t is written as C rank-1 components P_c = u_c v_c^T,
gated per token by a causal importance g_{t,c}; the reconstruction is
ŷ_t = Σ_c g_{t,c} (v_c^T x_t) u_c. The method is set by the config's CI function and loss terms.

| Method | CI function (`pd.ci_config.mode`) | Optimized losses |
|---|---|---|
| ASPD | `aspd`: shared BatchTopK encoder g^s on the residual stream r_t; g_{t,c} = 1[feature c selected] | L_internal (FVU), L_act, AuxK |
| PD Transcoder | `pd_transcoder`: BatchTopK over v_c^T (x_t − b_dec) | L_internal (Matryoshka), AuxK |
| VPD | `global`: core's transformer CI network | L_sparse, L_param, L_ablate (stochastic + adversarial) |
| VPD + internal | `global` | the above + L_internal (FVU), adaptive L0 |

ASPD and PD Transcoder use `component_arch: transcoder` (components plus the biases b_dec, b_out).
The method name ("arm") is derived from the config (`aspd.arms.derive_arm_name`) and recorded in the
run's `provenance.json`.

## Training

    python -m aspd.cli.train <config.yaml> --run-id <id> [--resume auto]
    torchrun --standalone --nproc_per_node=<N> -m aspd.cli.train <config.yaml> --run-id <id>

- Input: an experiment config (`configs/<model>/<arm>.yaml`); the data is streamed from Hugging Face
  (`data.dataset_name`). A config that sets `runtime.dp` must be launched with `torchrun` on that
  many processes; `pd.batch_size` is the global batch.
- Output: `$PARAM_DECOMP_OUT_DIR/runs/<id>/` (default `out/runs/<id>/`) with
  `experiment_config.yaml`, `model_<step>.pth`, `training_<step>.pth`, `metrics.jsonl` and
  `provenance.json`. W&B logging follows the config's `wandb` block (`null` disables it).
- `--resume auto` continues from the newest complete checkpoint of `--run-id`, data position included.

ASPD calibrates its encoders before training: b_dec is set to the mean residual stream at each
site, W_enc gets unit-norm columns and W_dec = W_enc^T.

## Coefficient calibration

    python -m aspd.calibrate <config.yaml> [--n-batches 8] [--out calibration/<name>.json]

- Output: for L_internal, L_act and AuxK, candidate coefficients λ that put their gradient norm at
  0.1× to 10× the gradient norms of the config's existing VPD terms (L_sparse, L_param, L_ablate) at
  step 0, per parameter group.
- For configs that add these terms to VPD (`configs/<model>/vpd_internal*.yaml`). A config with no
  optimized VPD term (ASPD, PD Transcoder) has nothing to calibrate against and is refused.

## Evaluation SAEs

    python -m aspd.cli.train_sae configs/sae/<model>.yaml

- Output: `artifacts/saes/<model>/`, a Matryoshka BatchTopK SAE pair at the decomposed matrix's
  input and output (the evaluations use the output SAE, trained on y_t).
- `python -m aspd.cli.sae_report configs/sae/<model>.yaml` reports FVU, L0 and dead fraction.

# Evaluation

Every command reads a run directory (`--run-dir out/runs/<id>`) and writes its results inside it,
per checkpoint step. The LLM judge is any OpenAI-compatible endpoint (below).

## Harvest

    python -m aspd.cli.harvest --run-dir <run> [--steps <step>]
    python -m aspd.cli.sae_harvest configs/sae/<model>.yaml

- Output: `<run>/harvest/h-step<step>/harvest.db`: per component, activation examples (±20 tokens
  around a firing), firing density; 400 batches of 16 × 512 tokens by default. `sae_harvest` does the
  same for the evaluation SAE's latents.

## Interpretability (intruder score)

    python -m aspd.cli.intruder --run-dir <run> [judge options]
    python -m aspd.cli.intruder_threshold --run-dir <run> --ci-thresholds 0.01 0.1 [judge options]

- Output: `<run>/harvest/h-step*/intruder_summary[_ci<τ>].json` with `mean`, per-component
  `scores`, `n_scored`.
- Meaning: for 200 components, 10 trials each show the judge 4 activating examples and 1 example of
  a component with firing density within 0.05; the score is the fraction of trials where the judge
  picks the intruder. Chance is 0.2. `intruder_threshold` redefines firing as g_{t,c} > τ.
- Component sample: `<run>/harvest/intruder_keys_seed0[_ci<τ>].json`, reused when present. The
  downloaded runs carry the paper's samples, which were drawn from the components eligible at every
  harvested checkpoint (200k to 1M steps); only the final harvest is shipped, so without these files
  the sample is drawn from the final step and the score differs from the paper's by sampling noise.

## Diversity

    python -m aspd.cli.diversity --run-dirs <run> ... [--threshold τ --criterion ci] [--out <file>]

- Output: `<run>/diversity/diversity_h-step<step>[_ci<τ>].json` with `z_bar` (Sim), `z_bar_ci95`.
- Meaning: T_c is the set of token types component c fires on at least twice among its examples;
  Sim is the mean pairwise Jaccard overlap |T_c ∩ T_d| / |T_c ∪ T_d| over 500 components with firing
  density in [5·10⁻⁵, 10⁻³]. Lower is more diverse. The interval is 1.96 × the bootstrap SE over
  resampled components.

## Meaning localization (matching)

    python -m aspd.cli.matching --run-dir <run> --sae-dir artifacts/saes/<model> [judge options]

- Needs the run's harvest and the SAE's harvest.
- Output: `<run>/matching/matching_c2o_step<step>.json`: judged scores of matched and random pairs.
- Meaning: each component c is paired with the output-SAE feature it affects most,
  π(c) = argmax_j E_{t∈A_j}[|ζ_c(t)|] ⟨W_enc[:, j], u_c⟩; the judge compares 9 examples of each
  (SIMILAR 3, MAYBE 2, DIFFERENT 1) for 200 pairs. Reported: mean score minus the mean of random
  component–feature pairs (`matching_A_cond_margin`).

## Weight editing

    python -m aspd.cli.editing --run-dir <run> --sae-dir artifacts/saes/<model>
    python -m aspd.cli.editing_multi --run-dir <run> --sae-dir artifacts/saes/<model> --m <m> --setup cond

- Output: `<run>/attr_edit/` and `<run>/attr_edit_multi/m<m>_cond/`.
- Meaning: components are ranked for each target output-SAE feature j by
  effect_{j,c} = ⟨W_enc[:, j], u_c⟩ E_{t∈A_j}[ζ_c(t)]; the top-k are deleted from the frozen weight,
  W′ = W − Σ u_c v_c^T, and the model is rerun on 10⁶ tokens. Localization is the change of the target
  features divided by the change of all features; the ratio divides it by the localization of editing
  the same number of random components. Single: 50 features, each edited alone,
  k ∈ {1, 5, 10, 20, 50}. Multiple: 50 sets of |J| = m features from a pool of 600, the union of
  per-feature top-k, k ∈ {1, 5, 10} (up to 50 for m = 1); run once per m ∈ {1, 5, 10, 20, 50}. Scores
  are averaged over k (Single), then over m (Multiple).

## Component labels (autointerp)

    python -m aspd.cli.harvest --run-dir <run> --token-stats full --force
    python -m aspd.cli.autointerp --rid <run id> [--cap 500] [judge options]

- Output: one natural-language label per component, written by the judge from its activating
  examples. Not used for any number in the paper.
- Needs a harvest made with `--token-stats full`; the downloaded harvests do not carry those token
  statistics. On a downloaded run, `--force` replaces the paper's harvest, which Interp and Sim read,
  so run autointerp on a separate copy of the run if you also want the paper's Interp and Sim.

## Evaluation tokens

Matching and editing are computed on 1M tokens from the eval split. To get the paper's numbers you must
use the same tokens, and `python -m aspd.cli.download --sae <model>` ships them as
`artifacts/saes/<model>/eval_tokens.pt`; matching and editing read that file whenever it is present.

Why a file: the tokens are drawn by streaming the dataset through a shuffle buffer, and the order of
that shuffle depends on the installed `datasets` version. The paper ran with `datasets` 3.6.0; this
repository installs 5.x, which draws different sequences from the same seed. The feature sample the
editing uses was selected on the paper's tokens (each feature must fire on ≥ 100 of them), so on other
tokens the editing refuses the sample rather than report numbers from a different set of features.
Without the file, both commands stream their own tokens: matching then differs from the paper by
sampling noise, and editing needs `--redraw` to select a new feature sample on those tokens.

## Tables

    python -m aspd.cli.tables --run-dir <run> ... --ci --out tables/<model>.csv
    python -m aspd.cli.latex --model GPT2=tables/gpt2_full.csv --model Gemma-2-2B=tables/gemma2_full.csv \
        --model Qwen-3-8B=tables/qwen3_full.csv --out tables/

- Output: `<model>.csv` (one row per method, `mean ± 95% CI`), `<model>_full.csv` (every column), and
  `interp_sim.tex`, `interp_sim_thresholds.tex`, `matching.tex`, `editing.tex`.
- Rows are labelled by the method each run's config derives. Intervals are Student-t 95% CIs over
  components, pairs, features or feature sets, and the bootstrap interval for Sim.

## The judge

The judge samples its answers (the server's default temperature), as in the paper. Everything the judge
is shown is deterministic (which components, pairs and examples), but its scores are not: rerunning
Interp or Matching gives the paper's numbers up to judge sampling. For GPT-2 ASPD, a rerun of matching
changed 39 of 600 judged pairs and moved the margin from 1.015 to 1.045, inside the paper's 95% interval.

    --judge-base-url  OpenAI-compatible base URL   (default http://127.0.0.1:8010/v1, a local vLLM)
    --judge-model     model id                      (default unsloth/Llama-3.3-70B-Instruct)
    --judge-api-key-env  environment variable with the key (omit for a local server)
    --judge-concurrency  in-flight requests          (default 64)

The paper uses Llama-3.3-70B-Instruct. Hosted providers serving it (e.g. OpenRouter
`meta-llama/llama-3.3-70b-instruct`) work the same way; a provider serving a quantized model can
shift scores slightly. `scripts/serve_judge.sh` starts a local vLLM server.

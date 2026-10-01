# Model-wide analysis (GPT-2, Section 5)

Everything here runs on the model-wide decomposition `gpt2_all_aspd` (72 matrices: Q, K, V, O, MLP_in
and MLP_out of every layer, 6144 components each). Download it with its harvest and its co-activation
table:

    python -m aspd.cli.download gpt2_all_aspd

or train it (`torchrun --standalone --nproc_per_node=4 -m aspd.cli.train configs/gpt2_all/aspd.yaml
--run-id gpt2_all_aspd`), harvest it (`python -m aspd.cli.harvest --run-dir out/runs/gpt2_all_aspd`)
and compute the co-activation table (`python -m aspd.cli.coactivation --run-dir out/runs/gpt2_all_aspd`).

The analysis is done in two browser apps. On a cluster, start them in a job and forward the port from
your machine: `ssh -N -L <port>:localhost:<port> <user>@<login node>` and then, from the login node,
`ssh -N -L <port>:localhost:<port> <compute node>`; open `http://localhost:<port>`.
`slurm/pair_app.sbatch` and `slurm/circuit_app.sbatch` print these commands for the node they land on.

## The pair app: IOI mechanisms and semantic tracing

    python -m aspd.cli.serve_pairs --run gpt2_all_aspd --qk-edit weight [--port 8060]

CPU only; about 14 GB of memory once a prompt is loaded. Two pages:

**`/prompt`: one prompt through the decomposition** (§5.1, Figs. 1, 2, 4–19). Enter the IOI prompt
`When Mary and John went to the store, John gave a drink to` and choose the target (`target: logit
diff` between ` Mary` and ` John`, or the top-k target). Then:

1. **run attribution**: attribution patching of the target to every component at every token,
   `attrib(c, t) = ⟨∂T/∂ŷ_t, e_{t,c} u_c⟩`, computed on the exact forward. Rank the components of a
   head's Q, K, V or O matrix (a head is picked with *module* and *head*) to find the candidates of
   one head class (previous-token, duplicate-token, induction, S-inhibition, name mover).
2. **attention — QK decomposition**: pick a head and one attention entry (query token, key token).
   The panel ranks query–key component pairs by `contrib^QK`, their share of that attention score.
   Selecting pairs and removing them recomputes the head's attention pattern from the edited weights
   `W^Q_edit`, `W^K_edit` and shows `Δpattern` (`--qk-edit weight`; `gated` subtracts the pair's own
   term instead). This is Fig. 2: removing 14Q–101K of H8.6 suppresses the S-inhibition pattern.
3. **OV and cross-layer contributions**: from an O or V component, the components it writes to in
   later layers (`contrib^OV`, `contrib^res`), e.g. 224O of H3.0 and 228O of H5.5 into 2861V and 101V
   of H8.6 (Fig. 11).
4. **interaction** and **SAE**: a component's interaction with pretrained SAE features on this prompt.

A component's activating examples, firing density and logit lens open from any row.

**`/`: component pairs across the whole run** (§5.2, Figs. 3, 20, 21). Pick a source (a component, or
a pretrained SAE feature) and a destination matrix or SAE, and rank the destination by
`dot`, `cosine` or `dot_coact` = Interact(c1, c2) = κ(c1, c2) ⟨u_c1, v_c2⟩, where κ is the corpus
co-activation from `harvest/pair_coactivation.pt`. Query–key pairs use the head-restricted inner
product. Semantic tracing chains these: an SAE feature on the residual stream → the MLP_in components
that read it → the MLP_out components they interact with most → the SAE features those write to
(Fig. 3: NBC/Fox feature → MLP_in 2535 → MLP_out 3885 → news feature). κ is an unconditional mean, so
it favours dense components; the **density ≤** control drops partners denser than its value. Fig. 3's
ranking is `dot_coact` with density ≤ 1e-2: MLP_out 3885 is then the top partner of MLP_in 2535 in
layer 7.

Pretrained SAEs (SAELens releases listed in `aspd/analysis/pairs/sae_config.py`) are read through
their encoder column (read side, LayerNorm gain folded in) and decoder row (write side). Feature labels
and examples are fetched from Neuronpedia unless `--no-neuronpedia` is passed.

## The circuit app: attribution graphs

    python -m aspd.cli.serve_app --run gpt2_all_aspd [--port 8055]

Needs a GPU. For a prompt it computes and draws the attribution graph over components (nodes) and
their edges, cached under `$PARAM_DECOMP_OUT_DIR/app`, next to a component browser with activating
examples. `?method=err` uses the exact forward with error nodes; `?method=lab` the lab's gradients.
`LM_INTERP_NODE_CAP` (default 500) and `LM_INTERP_MAX_DENSITY` bound how many components are drawn;
both can also be changed in the UI. It gives the graph view of a prompt; the per-head QK / OV analysis
and the weight edit are on the pair app's `/prompt` page.

## Behavioural probes (App. H)

    python -m aspd.analysis.probes.induction --run gpt2_all_aspd --layer 5 --head 5 --n 60 --prompts 4
    python -m aspd.analysis.probes.duplicate --run gpt2_all_aspd --layer 3 --head 0 --n 60 --prompts 4
    python -m aspd.analysis.probes.name_duplicate --run gpt2_all_aspd --layer 8 --comp 101

- Output (stdout): per O-matrix component, the probe score (mean |e_{t,c}| on probe tokens minus on
  control tokens), unfiltered and restricted to components with head mass ≥ 0.25; for
  `name_duplicate`, the class means and Π_name, Π_word and Δ on Prompt A and Prompt B.

## Co-activation table

    python -m aspd.cli.coactivation --run-dir out/runs/gpt2_all_aspd

- Output: `<run>/harvest/pair_coactivation.pt`, κ(c1, c2) = E_{X,t}[g_{t,c2} e_{t,c1}] for every
  module pair the pair app ranks (OV, QK, MLP_in → MLP_out, cross-layer residual). Downloaded with the
  run; recompute it only after retraining.

# Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space

Code for **ASPD** (Activation-Supported Parameter Decomposition) and the paper's baselines, evaluations
and model-wide analysis.

This fork also contains an isolated **multi-model ASPD** implementation for joint activation
diffing and parameter decomposition. It learns one sparse coordinate `c` across models and uses it
to gate model-specific rank-1 components `P^(n)_{j,c}`. Start with
[`docs/multimodel/method.md`](docs/multimodel/method.md), then follow the
[`HiPerGator guide`](docs/multimodel/hipergator.md). The upstream single-model ASPD paths remain
unchanged.

```bash
python -m aspd.multimodel.cli.validate configs/multimodel/qwen3_1_7b/smoke.yaml
python -m aspd.multimodel.cli.cache configs/multimodel/qwen3_1_7b/smoke.yaml
python -m aspd.multimodel.cli.train configs/multimodel/qwen3_1_7b/smoke.yaml
```

**Authors:** `Tue Minh Cao, Lisiane Pruinelli, My T. Thai` · **Paper:** [link](https://arxiv.org/pdf/2609.37731)

## Contents

```
aspd/            the package
  run.py, losses.py, ci/, arms.py      training: method setup, losses, shared encoder
  eval/                                harvest, Interp, Sim, matching, editing, tables
  analysis/                            model-wide analysis (§5): pairs, prompt traces, probes
  cli/                                 entry points (`python -m aspd.cli.<name>`)
configs/<model>/<arm>.yaml             one config per table row
configs/gpt2_all/aspd.yaml             the model-wide run (§5)
configs/sae/<model>.yaml               evaluation SAEs
docs/                                  method.md, evaluation.md, analysis.md
frontend/                              the circuit app's web frontend
PAPER_MAP.md                           paper notation, tables and figures -> code
```

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is needed for training and most evaluations.

```bash
git clone https://github.com/tue147/weights-read-write && cd weights-read-write
uv sync
source .venv/bin/activate
```

[param-decomp](https://github.com/goodfire-ai/param-decomp) (the component model, masking, core
losses and trainer) is a pinned dependency, installed by `uv sync`. Gemma-2 is a gated model:
accept its license on Hugging Face and log in (`hf auth login`) before using the `gemma2` configs.

## Trained runs

Every run in the paper is on the Hugging Face Hub
([`tueminh/wfd-runs`](https://huggingface.co/tueminh/wfd-runs), folder `aspd/`): the config, the final
checkpoint (the decomposition; the target model is rebuilt from the config) and the final harvest.

| Run id | Paper method |
|---|---|
| `<model>_aspd` | ASPD |
| `<model>_pdtc` | PD Transcoder |
| `<model>_pdtc_param` | PD Transcoder + `L_param` |
| `<model>_pdtc_ablate` | PD Transcoder + `L_ablate` |
| `<model>_vpd` | VPD |
| `<model>_vpd_adaptive` | VPD with adaptive L0 |
| `<model>_vpd_internal` | VPD + `L_internal` |
| `<model>_vpd_internal_noparam` | VPD + `L_internal`, no `L_param` |
| `<model>_vpd_internal_noablate` | VPD + `L_internal`, no `L_ablate` |
| `gpt2_all_aspd` | model-wide ASPD on GPT-2 (§5) |

`<model>` is `gpt2` (GPT-2 small, MLP in-projection of layer 0), `gemma2` (Gemma-2-2B, MLP
down-projection of layer 13) or `qwen3` (Qwen3-8B, attention output projection of layer 17).

```bash
python -m aspd.cli.download --list                       # runs and sizes
python -m aspd.cli.download gpt2_aspd gpt2_vpd --sae gpt2 # two runs + GPT-2's evaluation SAE
python -m aspd.cli.download --all                        # every run (~270 GB)
```

Runs land in `out/runs/<run id>/` (`$PARAM_DECOMP_OUT_DIR/runs`), SAEs in `artifacts/saes/<model>/`.

## Quickstart: one table row from a trained run

```bash
python -m aspd.cli.download gpt2_aspd --sae gpt2
scripts/serve_judge.sh &                     # Llama-3.3-70B-Instruct via vLLM; see "LLM judge"
R=out/runs/gpt2_aspd

python -m aspd.cli.intruder  --run-dir $R                                   # Interp
python -m aspd.cli.diversity --run-dirs $R                                  # Sim
python -m aspd.cli.matching  --run-dir $R --sae-dir artifacts/saes/gpt2     # Matching
python -m aspd.cli.editing   --run-dir $R --sae-dir artifacts/saes/gpt2     # Editing, single
for m in 1 5 10 20 50; do                                                   # Editing, multiple
  python -m aspd.cli.editing_multi --run-dir $R --sae-dir artifacts/saes/gpt2 --m $m --setup cond
done
python -m aspd.cli.tables --run-dir $R --ci --out tables/gpt2.csv
```

The downloaded harvest is the one every paper number was computed from; `aspd.cli.harvest` rebuilds it.

## Reproducing the paper

### Section 4: comparison on one matrix per model

Train (or download) the nine runs of a model, evaluate each as in the quickstart, then build the tables:

```bash
python -m aspd.cli.train configs/gpt2/aspd.yaml --run-id gpt2_aspd                     # 1 GPU
torchrun --standalone --nproc_per_node=2 -m aspd.cli.train configs/qwen3/aspd.yaml --run-id qwen3_aspd

python -m aspd.cli.harvest --run-dir out/runs/gpt2_aspd
python -m aspd.cli.intruder_threshold --run-dir out/runs/gpt2_vpd --ci-thresholds 0.01 0.1   # Table 9
python -m aspd.cli.diversity --run-dirs out/runs/gpt2_vpd --threshold 0.1 --criterion ci

python -m aspd.cli.tables --run-dir out/runs/gpt2_aspd --run-dir out/runs/gpt2_pdtc ... --ci --out tables/gpt2.csv
python -m aspd.cli.latex --model GPT2=tables/gpt2_full.csv --model Gemma-2-2B=tables/gemma2_full.csv \
    --model Qwen-3-8B=tables/qwen3_full.csv --out tables/
```

Details: [docs/method.md](docs/method.md), [docs/evaluation.md](docs/evaluation.md).

### Section 5: model-wide decomposition of GPT-2

The mechanism analysis is done in two browser apps on the model-wide run:

```bash
python -m aspd.cli.download gpt2_all_aspd                              # run, harvest, co-activation table
python -m aspd.cli.serve_pairs --run gpt2_all_aspd --qk-edit weight   # pair app, CPU, port 8060
python -m aspd.cli.serve_app --run gpt2_all_aspd                      # circuit app, GPU, port 8055
```

- **IOI mechanisms (§5.1)**: the pair app's `/prompt` page. Enter the IOI prompt, run attribution
  patching on the logit difference, then for each head class open its QK panel at the relevant
  attention entry: query–key component pairs are ranked by their contribution, and removing a pair
  from the weights shows the change in the attention pattern (Fig. 2). OV and cross-layer
  contributions connect the heads (Fig. 11).
- **Semantic tracing (§5.2)**: the pair app's main page. From a pretrained SAE feature, rank the
  components that read it, the components they interact with most (Interact, from the co-activation
  table, with *density ≤ 1e-2*), and the SAE features those write to (Fig. 3).
- **Attribution graphs**: the circuit app draws the component graph of a prompt, with activating
  examples for every node.

On a cluster, run the apps in a job (`slurm/pair_app.sbatch`, `slurm/circuit_app.sbatch`) and forward
the port with `ssh -L`. Walkthrough, probes and all options: [docs/analysis.md](docs/analysis.md).

## LLM judge

Interp and matching are scored by Llama-3.3-70B-Instruct through any OpenAI-compatible endpoint:

```bash
scripts/serve_judge.sh                       # local vLLM on one GPU, port 8010 (the default endpoint)
python -m aspd.cli.intruder --run-dir out/runs/gpt2_aspd \
    --judge-base-url https://openrouter.ai/api/v1 --judge-model meta-llama/llama-3.3-70b-instruct \
    --judge-api-key-env OPENROUTER_API_KEY
```

The paper's numbers use unquantized weights served with vLLM (`scripts/serve_judge.sh`). A hosted
provider serving a quantized model can shift scores slightly.

## Paper ↔ code

[PAPER_MAP.md](PAPER_MAP.md) maps the paper's notation to code identifiers, and every table, figure and
appendix result to the config, command, output file and column that produce it.

## Citation

```bibtex
@misc{cao2026weightsreadwritefeatures,
      title={Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space}, 
      author={Tue M. Cao and Lisiane Pruinelli and My T. Thai},
      year={2026},
      eprint={2609.37731},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.37731}, 
}
```

## License

MIT (see [LICENSE](LICENSE)). Third-party code and assets: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

"""Print the circuit for one prompt."""

import argparse

import torch

from aspd.paths import RUNS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id under $PARAM_DECOMP_OUT_DIR/runs")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from aspd.analysis.circuits.per_prompt import prompt_circuit
    from aspd.analysis.circuits.replacement import assert_forward_is_exact
    from aspd.analysis.circuits.targets import top_k_vs_rest
    from aspd.checkpoints import load_run_model

    run_dir = RUNS_DIR / args.run
    step = args.step or max(
        int(p.stem.split("_")[1]) for p in run_dir.glob("model_*.pth")
    )
    print(f"loading {run_dir} @ step {step}")
    model, cfg = load_run_model(run_dir, step, device="cpu")
    print(f"  {len(model.components)} decomposed matrices")

    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    tokens = tok(args.prompt, return_tensors="pt")["input_ids"]
    pieces = [tok.decode([t]) for t in tokens[0].tolist()]
    print(f"  prompt {args.prompt!r} -> {len(pieces)} tokens {pieces}")

    assert_forward_is_exact(model, tokens, sampling=cfg.pd.sampling)
    print("  forward exactness: OK (replacement == target model)")

    edges, stats = prompt_circuit(
        model, tokens, sampling=cfg.pd.sampling,
        target_fn=lambda z: top_k_vs_rest(z, k=args.k),
    )
    with torch.no_grad():
        logits = model(tokens)
    top = torch.topk(logits[0, -1], args.k)
    print("\n  model's top-%d next tokens: %s" % (
        args.k, [tok.decode([t]) for t in top.indices.tolist()]))

    print(f"\n  target  mean(top-{args.k}) - mean(rest) = {stats['target']:.4f}")
    print(f"  fired components : {int(stats['n_fired_components'])}")
    print(f"  edges            : {int(stats['n_edges'])}")
    print(f"  sum of edges     : {stats['sum_edges']:.4f}")
    print(f"  ERROR SHARE      : {stats['error_share']:.1%}  "
          "<- fraction the decomposition cannot explain")

    print(f"\n  top {args.top} edges into the target:")
    print(f"  {'weight':>10}  {'kind':9} {'pos':>3} {'token':<12} layer / component")
    for e in edges[: args.top]:
        p = e.source.seq_pos
        piece = repr(pieces[p]) if p is not None and p < len(pieces) else ""
        name = e.source.layer.replace("transformer.h.", "h")
        comp = f":{e.source.component_idx}" if e.source.component_idx is not None else ""
        print(f"  {e.weight:10.4f}  {e.source.kind:9} {p:>3} {piece:<12} {name}{comp}")


if __name__ == "__main__":
    main()

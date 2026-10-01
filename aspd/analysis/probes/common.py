"""Shared probe machinery.

A probe scores component c as the mean |e_{t,c}| over the tokens that should trigger the behaviour
minus the mean over control tokens. For a matrix spanning several heads, mass_{h,c} is the share of
the component's head-side direction lying in head h's block; rankings are reported unfiltered and
restricted to mass_{h,c} >= `--min-mass`.
"""

import argparse
import os
import time

import torch

EOS = 50256


def parse(description: str, layer: int, head: int) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--layer", type=int, default=layer)
    p.add_argument("--head", type=int, default=head)
    p.add_argument("--look", default="", help="comma-separated component ids to report by name")
    p.add_argument("--n", type=int, default=24, help="tokens per copy / per probe half")
    p.add_argument("--prompts", type=int, default=1,
                   help="independent prompts to pool, each with its own seed. Positions within one "
                        "prompt share its token draw and its context; separate prompts do not.")
    p.add_argument("--top", type=int, default=12, help="rows in the ranking")
    p.add_argument("--min-mass", type=float, default=0.25,
                   help="second ranking restricted to components with mass_{h,c} >= this for --head")
    p.add_argument("--run", required=True, help="run id under $PARAM_DECOMP_OUT_DIR/runs (the model-wide ASPD run)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    a = p.parse_args()
    a.look = [int(x) for x in a.look.split(",") if x.strip()]
    torch.set_num_threads(a.threads)
    return a


class Probe:
    """One loaded run, traced prompt by prompt."""

    def __init__(self, args: argparse.Namespace):
        from aspd.analysis.prompt.engine import PromptEngine

        self.args = args
        self.t0 = time.time()
        from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

        self.eng = PromptEngine(PARAM_DECOMP_OUT_DIR / "runs" / args.run, device="cpu")
        self.eng._load()  # noqa: SLF001
        self.tok = self.eng._tok  # noqa: SLF001
        self.say(f"loaded {args.run} on {args.threads} threads")

    def say(self, msg: str) -> None:
        print(f"[{time.time() - self.t0:.0f}s] {msg}", flush=True)

    def trace(self, ids: torch.Tensor):
        from aspd.analysis.prompt.trace import build_trace

        pieces = [self.tok.decode([t]) for t in ids[0].tolist()]
        tr = build_trace(self.eng._model, self.eng._cfg, ids, pieces)  # noqa: SLF001
        self.say(f"traced {ids.shape[1]} tokens")
        return tr

    def o_module(self, tr) -> str:
        return tr.module_at(self.args.layer, "attn.o")

    def head_mass(self, tr) -> torch.Tensor:
        """`[n_heads, C]` -- share of each component's read-direction norm^2 in each head's block."""
        V = self.eng._model.components[self.o_module(tr)].V.detach().float()  # noqa: SLF001  [d, C]
        n_heads = tr.n_heads
        per_head = V.pow(2).unflatten(0, (n_heads, V.shape[0] // n_heads)).sum(1)
        return per_head / per_head.sum(0, keepdim=True).clamp_min(1e-12)


class Tally:
    """Pools probe/control positions ACROSS prompts, holding sums rather than traces."""

    def __init__(self, probe: Probe):
        self.p = probe
        self.mod: str | None = None
        self.mass: torch.Tensor | None = None
        self.d = self.c = self.fires = None
        self.n_hit = self.n_ctl = 0

    def add(self, tr, hit: list[int], ctl: list[int]) -> None:
        if self.mod is None:
            self.mod, self.mass = self.p.o_module(tr), self.p.head_mass(tr)
        e = tr.effective(self.mod).abs()
        d, c, f = e[hit].sum(0), e[ctl].sum(0), (e[hit] > 0).sum(0)
        self.d = d if self.d is None else self.d + d
        self.c = c if self.c is None else self.c + c
        self.fires = f if self.fires is None else self.fires + f
        self.n_hit += len(hit)
        self.n_ctl += len(ctl)

    def show(self, name: str) -> None:
        a = self.p.args
        assert self.d is not None and self.mass is not None, "nothing tallied"
        d, c = self.d / self.n_hit, self.c / self.n_ctl
        gap, mass, fires = d - c, self.mass, self.fires

        print(f"\n=== {name} ===  {self.n_hit} probe positions, {self.n_ctl} control  ({self.mod})")
        print(f"{'comp':>7} {'probe':>8} {'ctl':>8} {'gap':>8} {'fires':>11} "
              f"{'mass h' + str(a.head):>8} {'top head':>9}")

        def line(i: int, rank: int, of: int) -> None:
            print(f"{i:>7} {float(d[i]):>8.3f} {float(c[i]):>8.3f} {float(gap[i]):>8.3f}"
                  f" {int(fires[i]):>5}/{self.n_hit:<5} {float(mass[a.head, i]):>8.2f}"
                  f" {'h' + str(int(mass[:, i].argmax())):>9}   rank {rank}/{of}")

        for i in torch.argsort(gap, descending=True)[:a.top].tolist():
            line(i, int((gap > gap[i]).sum()) + 1, gap.numel())

        keep = (mass[a.head] >= a.min_mass).nonzero().flatten()
        print(f"\n-- of the {keep.numel()} components reading >= {a.min_mass:.0%} from head "
              f"{a.layer}.{a.head} --")
        for i in keep[torch.argsort(gap[keep], descending=True)][:a.top].tolist():
            line(i, int((gap[keep] > gap[i]).sum()) + 1, keep.numel())

        if a.look:
            print(f"{'':>7} {'-' * 52}")
            for i in a.look:
                line(i, int((gap > gap[i]).sum()) + 1, gap.numel())
                if mass[a.head, i] >= a.min_mass:
                    print(f"{'':>7} ^ rank {int((gap[keep] > gap[i]).sum()) + 1}/{keep.numel()} "
                          f"within head {a.layer}.{a.head}")

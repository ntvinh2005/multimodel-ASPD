"""Induction probe: [eos] r_1..r_n r_1..r_n with random tokens r_i.

Probe tokens are the second copy, control tokens the first.
"""

import torch

from aspd.analysis.probes.common import EOS, Probe, Tally, parse


def main() -> None:
    a = parse(__doc__.splitlines()[0], layer=6, head=9)
    p = Probe(a)
    tally = Tally(p)

    for seed in range(a.prompts):
        g = torch.Generator().manual_seed(seed)
        r = torch.randint(1000, 49000, (a.n,), generator=g)
        ids = torch.cat([torch.tensor([EOS]), r, r]).unsqueeze(0)
        tr = p.trace(ids)
        tally.add(tr, list(range(1 + a.n, 1 + 2 * a.n)), list(range(1, 1 + a.n)))

    tally.show(f"induction, layer {a.layer}")
    p.say("done")


if __name__ == "__main__":
    main()
